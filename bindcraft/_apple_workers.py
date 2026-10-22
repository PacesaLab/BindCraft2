"""Private Apple worker planning and persistent-pool supervision.

Only configured Metal entry points dispatch here. The scientific campaign loop,
length buckets, estimator and reserve policy are shared without alteration.
"""
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from bindcraft import _apple
from bindcraft import design_workers as shared
from bindcraft._worker_memory import MIB, pinned_settings, resource_capacity, validate_resource_budget
from bindcraft.design_workers import (TRAJECTORY_ONLY_WORKERS_PER_GPU, AUTOMATIC_WORKERS_PER_GPU,
                                      MAXIMUM_WORKERS_PER_GPU)
METAL_ALLOCATION_UNIT = MIB

# This bootstrap uses only the standard library. exec preserves the registered
# PID, and the pipe closes on parent death; no child imports a model before its
# supervisor has registered it. Q's worker_context performs the state-side gate.
_WORKER_STARTUP_GATE = r'''
import os, select, sys
fd, parent = int(sys.argv[1]), int(sys.argv[2])
ready, _, _ = select.select([fd], [], [], 30.0)
allowed = bool(ready) and os.read(fd, 1) == b'1' and os.getppid() == parent
os.close(fd)
if not allowed:
    print('campaign worker startup refused: supervisor registration unavailable', file=sys.stderr, flush=True)
    raise SystemExit(2)
os.execvpe(sys.argv[3], sys.argv[3:], os.environ)
'''
_WORKER_TERMINATE_SECONDS = 5.0
_WORKER_KILL_SECONDS = 5.0
_MAX_IDLE_REPLANS = 8

def _metal_worker_problems(settings: dict) -> list[str]:
    problems = []
    for name, environment in (('design_workers', 'BINDCRAFT_DESIGN_WORKERS'), ('workers_per_gpu', 'BINDCRAFT_WORKERS_PER_GPU')):
        requested = os.environ.get(environment, settings.get(name))
        if str(requested).strip().lower() not in ('none', '', 'auto'):
            try:
                valid = int(requested) >= 0 and str(int(requested)) == str(requested).strip()
            except (ValueError, TypeError):
                valid = False
            if not valid:
                problems.append(f'Metal {environment if environment in os.environ else name}={requested!r} must be auto or a nonnegative worker count')
    gpu_ids = os.environ.get('BINDCRAFT_GPU_IDS', settings.get('gpu_ids'))
    if gpu_ids is not None:
        named = gpu_ids.split(',') if isinstance(gpu_ids, str) else gpu_ids if isinstance(gpu_ids, (list, tuple)) else [gpu_ids]
        requested = [str(device).strip() for device in named if str(device).strip()]
        if requested not in ([], ['all'], ['0']):
            source = 'BINDCRAFT_GPU_IDS' if 'BINDCRAFT_GPU_IDS' in os.environ else 'gpu_ids'
            problems.append(f'Metal campaigns use the single Metal device; {source}={gpu_ids!r} is incompatible. Leave it unset or use 0')
    return problems


def plan_metal_design_workers(settings: dict, residue_count: int | None=None, trajectory_budget: int | None=None,
                              *, execution_profile: dict | None=None, resource_budget: dict | None=None) -> list[dict]:
    problems = _metal_worker_problems(settings)
    if problems:
        raise ValueError('\n'.join(problems))
    validate_resource_budget(resource_budget)
    managed = execution_profile is not None
    if managed:
        pinned_settings(settings, execution_profile)  # Validate before discovery/allocation.
    pool = _apple.metal_pool_budget()
    if pool['budget_bytes'] <= 0:
        raise ValueError('Metal memory budget leaves no positive campaign allowance under the reserve and native memory limits; free memory or review those limits before starting the campaign')
    requested = str(os.environ.get('BINDCRAFT_WORKERS_PER_GPU', settings.get('workers_per_gpu', 'auto'))).strip().lower()
    requested = 'auto' if requested in ('none', '') else requested
    packed = (TRAJECTORY_ONLY_WORKERS_PER_GPU if settings.get('trajectory_only') else AUTOMATIC_WORKERS_PER_GPU) if requested == 'auto' else int(requested)
    ceiling = max(0, int(os.environ.get('BINDCRAFT_MAX_WORKERS_PER_GPU', settings.get('max_workers_per_gpu', MAXIMUM_WORKERS_PER_GPU))))
    worker_limit = os.environ.get('BINDCRAFT_DESIGN_WORKERS', settings.get('design_workers'))
    worker_limit = 0 if str(worker_limit).strip().lower() in ('none', '', 'auto') else int(worker_limit)
    if trajectory_budget is not None and trajectory_budget <= 0:
        return []
    limits = [limit for limit in (worker_limit, trajectory_budget) if limit]
    count = min([packed, ceiling, *limits, *([] if settings.get('auto_multi_gpu', True) else [1])])
    if count <= 0:
        raise ValueError('Metal worker count ceilings leave no admissible design worker')
    length_buckets = shared.campaign_length_buckets(settings)
    bucket_size = shared.campaign_length_bucket(settings)
    requested_count = count
    for count in range(requested_count, 0, -1):
        # A managed pool of one still has a separate supervisor process.
        available_bytes = resource_capacity('metal:unified', pool['worker_pool_bytes' if managed or count > 1 else 'budget_bytes'], resource_budget)
        available = available_bytes // METAL_ALLOCATION_UNIT
        if available <= 0:
            raise ValueError('Metal memory budget leaves no positive child capacity after coordinator overhead and the campaign resource ceiling')
        budgets = {'metal:unified': available * METAL_ALLOCATION_UNIT}
        plan = shared.assign_worker_length_buckets([{'gpu': 'metal', 'card': 0, 'memory_fraction': 0.0} for _ in range(count)], length_buckets)
        for worker in plan:
            worker['residue_count'] = shared.worker_residue_count(settings, worker, residue_count)
            shapes = len({shared.padded_prediction_length(length, bucket_size) for length in worker['lengths']}) if worker.get('lengths') else len(length_buckets)
            profile_argument = {'execution_profile': execution_profile} if managed else {}
            worker['metal_estimate_bytes'] = _apple.estimate_metal_worker_bytes(settings, worker['residue_count'], shape_count=max(1, shapes), **profile_argument)
            worker['metal_pool'] = pool
        if any(worker['metal_estimate_bytes'] is None for worker in plan):
            if managed:
                raise ValueError('Supervised Metal worker memory is unknown: active configuration is outside the provisional worker-memory estimate envelope; admission requires a supported estimate')
            print('campaign Metal fan-out disabled: settings are outside the provisional worker-memory estimate envelope; using one persistent worker in the current process', flush=True)
            # Legacy current-process fallback is not evidence that this workload
            # fits. It retains the native governor, all original lengths and the
            # conservative unmanaged lifecycle; supervised workers refuse above.
            available_bytes = resource_capacity('metal:unified', pool['budget_bytes'], resource_budget) // METAL_ALLOCATION_UNIT * METAL_ALLOCATION_UNIT
            plan = [{'gpu': 'metal', 'card': 0, 'memory_fraction': 0.0, 'residue_count': residue_count or 0,
                     'metal_estimate_bytes': None, 'metal_pool': pool, 'memory_estimate_status': 'unknown'}]
            plan[0].update(metal_memory_budget_bytes=available_bytes, cpus=[],
                           resource_budgets={'metal:unified': available_bytes},
                           resource_reservations={'metal:unified': available_bytes})
            return plan
        needed = [max(1, (worker['metal_estimate_bytes'] + METAL_ALLOCATION_UNIT - 1) // METAL_ALLOCATION_UNIT) for worker in plan]
        if sum(needed) > available:
            if count > 1:
                continue
            raise ValueError(f"Metal memory budget {available / 1024:.2f} GiB cannot fit one estimated worker ({sum(needed) / 1024:.2f} GiB); free memory or reduce the campaign workload")
        spare, remainder = divmod(available - sum(needed), count)
        for index, (worker, requirement, cpus) in enumerate(zip(plan, needed, shared.worker_cpu_slices(count))):
            allowance = (requirement + spare + (index < remainder)) * METAL_ALLOCATION_UNIT
            worker.update(metal_memory_budget_bytes=allowance, cpus=cpus,
                          resource_budgets=dict(budgets), resource_reservations={'metal:unified': allowance})
        if count < requested_count:
            print(f'campaign Metal fan-out held to {count} of {requested_count} requested design worker(s) by the shared memory budget', flush=True)
        return plan
    raise AssertionError('Metal planning must return a plan or refuse an insufficient budget')


class _WorkerPoolCancelled(BaseException):
    def __init__(self, signum):
        self.signum = signum


def supervise_design_worker(process: subprocess.Popen, log_file, console: shared.TrajectoryOrderedConsole, worker_index: int, *, recycle_exit_code: int | None=None) -> None:
    try:
        shared.relay_worker_output(process.stdout, log_file, console, worker_index)
    finally:
        code = process.wait()
        if code and code != recycle_exit_code:
            console.aside(f'campaign lost design worker {worker_index}, {shared.worker_stop_reason(code)} | {log_file.name}')


def _stop_owned_workers(processes: list) -> None:
    # Signal only Popen-owned, unreaped children. Never act on stored state PIDs.
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + _WORKER_TERMINATE_SECONDS
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process in processes:
        if process.poll() is None:
            process.kill()
    deadline = time.monotonic() + _WORKER_KILL_SECONDS
    for process in processes:
        process.wait(timeout=max(0.0, deadline - time.monotonic()))


def _join_worker_output(relays: list) -> None:
    deadline = time.monotonic() + 2.0
    for relay in relays:
        relay.join(timeout=max(0.0, deadline - time.monotonic()))


def launch_design_workers(plan: list[dict], log_directory: str, worker_command: list[str], launch_stagger_seconds: float=0.0, project_folders: tuple[str, ...]=(), requested_designs: int=0, max_trajectories: int | None=None, *, context: dict | None=None, admission=None) -> int:
    if any(worker['gpu'] != 'metal' for worker in plan):
        raise ValueError('Apple supervision requires a Metal-only plan')
    if shared.running_as_design_worker():
        raise ValueError('A design worker cannot launch another worker pool')
    if (context is None) != (admission is None):
        raise ValueError('Managed workers require both generation context and supervisor admission')
    recycle_code = None
    if context is not None:
        from bindcraft._worker_lifecycle import RECYCLE_EXIT_CODE
        recycle_code = RECYCLE_EXIT_CODE
    os.makedirs(log_directory, exist_ok=True)
    processes, log_files, relays, console = [], [], [], shared.TrajectoryOrderedConsole()
    registration_attempted, recorded = set(), set()
    previous_handlers = {}
    outcome = None
    primary_error = None

    def cancelled(signum, frame):
        raise _WorkerPoolCancelled(signum)

    def record_exits():
        refusal = None
        for index, process in processes:
            if index in registration_attempted and index not in recorded and process.poll() is not None:
                # Admission must validate ownership and the persisted transition
                # for exit 75; a bare status cannot authorize recycling.
                try:
                    admission.record_worker_exit(context, index, process.pid, process.returncode)
                except Exception as error:
                    refusal = refusal or error
                finally:
                    recorded.add(index)
        if refusal is not None:
            raise refusal

    workers_on_card, launch_order = {}, []
    for worker_index, worker in enumerate(plan):
        rung = workers_on_card[worker['gpu']] = workers_on_card.get(worker['gpu'], -1) + 1
        launch_order.append((rung, worker_index, worker))
    launched_rung = 0
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, cancelled)
        for rung, worker_index, worker in sorted(launch_order):
            if rung != launched_rung and launch_stagger_seconds:
                time.sleep(launch_stagger_seconds)
            launched_rung = rung
            environment = {**os.environ, 'BINDCRAFT_WORKER_ID': str(worker_index), 'BINDCRAFT_WORKER_COUNT': str(len(plan))}
            # Do not retain a prior invocation's private child policy or slices.
            for name in ('BINDCRAFT_WORKER_CONTEXT', 'BINDCRAFT_BINDER_LENGTHS', 'BINDCRAFT_WORKER_CPUS', 'BINDCRAFT_METAL_WORKER_POLICY'):
                environment.pop(name, None)
            environment.pop('CUDA_VISIBLE_DEVICES', None)
            environment.pop('XLA_PYTHON_CLIENT_MEM_FRACTION', None)
            environment.update(_apple.metal_worker_environment(worker['metal_memory_budget_bytes'], pool=worker['metal_pool']))
            if worker.get('lengths'):
                environment['BINDCRAFT_BINDER_LENGTHS'] = ','.join(str(length) for length in worker['lengths'])
            if worker.get('cpus'):
                environment['BINDCRAFT_WORKER_CPUS'] = ','.join(str(cpu) for cpu in worker['cpus'])
            prefix = str(os.getpid())
            if context is not None:
                environment['BINDCRAFT_WORKER_CONTEXT'] = json.dumps(context, sort_keys=True)
                run_id = ''.join(char for char in str(context['run_id']) if char.isalnum() or char in '-_')
                prefix = f"{prefix}_run_{run_id}_generation_{int(context['generation']):04d}"
            log_file = open(os.path.join(log_directory, f'{prefix}_worker_{worker_index:02d}_gpu_{worker["card"]}.log'), 'x' if context is not None else 'a', buffering=1)
            log_files.append(log_file)
            if context is None:
                process = subprocess.Popen(worker_command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                processes.append((worker_index, process))
            else:
                read_gate, write_gate = os.pipe()
                try:
                    command = [sys.executable, '-u', '-c', _WORKER_STARTUP_GATE, str(read_gate), str(os.getpid()), *worker_command]
                    process = subprocess.Popen(command, pass_fds=(read_gate,), env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    processes.append((worker_index, process))
                    # Registration can persist its identity and then raise.
                    # Attempt to record every such child's confirmed exit;
                    # admission remains authoritative about ownership.
                    registration_attempted.add(worker_index)
                    admission.register_worker(context, worker_index, process.pid)
                    os.write(write_gate, b'1')
                finally:
                    os.close(read_gate)
                    os.close(write_gate)
            relay = threading.Thread(target=supervise_design_worker, args=(process, log_file, console, worker_index), kwargs={'recycle_exit_code': recycle_code}, daemon=True)
            relay.start()
            relays.append(relay)
            print(f'worker={worker_index} gpu={worker["card"]} log={log_file.name}', flush=True)
        original_failures = []
        while any(process.poll() is None for _, process in processes):
            record_exits()
            original_failures = [process.returncode for _, process in processes if process.returncode not in (None, 0, recycle_code)]
            if original_failures:
                break
            threading.Event().wait(0.05)
        _stop_owned_workers([process for _, process in processes])
        record_exits()
        _join_worker_output(relays)
        exit_codes = [process.returncode for _, process in processes]
        failures = [code for code in exit_codes if code and code != recycle_code]
        lost = len(failures)
        if lost:
            console.aside(f'campaign closed {lost} of {len(exit_codes)} design worker(s) short: their trajectories are abandoned, and each is named above with its log')
        if context is None:
            shared.report_campaign_close(project_folders, requested_designs, max_trajectories)
        # A genuine failure takes precedence even when another worker recycled.
        outcome = shared.campaign_exit_status(original_failures or failures or exit_codes)
        return outcome
    except _WorkerPoolCancelled as cancellation:
        outcome = 128 + cancellation.signum
        return outcome
    except BaseException as failure:
        primary_error = failure
        raise
    finally:
        cleanup_errors = []
        # Ignore further cancellation during this bounded shutdown. The original
        # signal handlers are restored before returning to the caller.
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        try:
            for cleanup in (lambda: _stop_owned_workers([process for _, process in processes]), record_exits,
                            lambda: _join_worker_output(relays)):
                try:
                    cleanup()
                except Exception as error:
                    cleanup_errors.append(error)
            for _, process in processes:
                if process.stdout is not None and not any(relay.is_alive() for relay in relays):
                    try:
                        process.stdout.close()
                    except Exception as error:
                        cleanup_errors.append(error)
            for log_file in log_files:
                if not any(relay.is_alive() for relay in relays):
                    try:
                        log_file.close()
                    except Exception as error:
                        cleanup_errors.append(error)
            if any(relay.is_alive() for relay in relays):
                cleanup_errors.append(RuntimeError('Worker output remained open after bounded cleanup; refusing another generation'))
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if cleanup_errors:
            if primary_error is not None:
                for error in cleanup_errors:
                    primary_error.add_note(f'Worker cleanup also failed: {type(error).__name__}: {error}')
            elif outcome not in (None, 0, recycle_code):
                for error in cleanup_errors:
                    console.aside(f'campaign worker cleanup failed after exit {outcome}: {type(error).__name__}: {error}')
            else:
                for error in cleanup_errors[1:]:
                    cleanup_errors[0].add_note(f'Worker cleanup also failed: {type(error).__name__}: {error}')
                raise cleanup_errors[0]


def _worker_command(settings: dict, log_directory: str, worker_command, worker_arguments) -> list[str]:
    if worker_command is not None:
        return worker_command
    os.makedirs(log_directory, exist_ok=True)
    worker_settings_path = os.path.join(log_directory, 'campaign_settings.json')
    handle, partial_path = tempfile.mkstemp(dir=log_directory, prefix='campaign_settings.', suffix='.partial')
    try:
        with os.fdopen(handle, 'w') as settings_file:
            settings_file.write(json.dumps(shared.json_compatible(settings), sort_keys=True))
        os.replace(partial_path, worker_settings_path)
    finally:
        if os.path.exists(partial_path):
            os.unlink(partial_path)
    return [sys.executable, '-u', '-m', 'bindcraft.cli', 'design', worker_settings_path, *worker_arguments]


def _report_worker_plan(plan, residue_count, trajectory_budget, context=None):
    if trajectory_budget and len(plan) >= trajectory_budget:
        print(f'campaign fan-out held to {len(plan)} design worker(s) by a budget of {trajectory_budget} trajectories: a worker beyond the budget claims none', flush=True)
    if context is not None:
        print(f"campaign generation {context['generation']} | profile {context['profile']['signature']} rung {context['profile']['rung']} | {len(plan)} worker(s) | resource budgets {json.dumps(plan[0]['resource_budgets'], sort_keys=True)}", flush=True)
    pool = plan[0]['metal_pool']
    print(f"campaign Metal fan-out: {len(plan)} persistent design workers on device 0 | budget {pool['budget_bytes'] / 1024 ** 3:.2f} GiB | coordinator {pool['coordinator_bytes'] / 1024 ** 3:.2f} GiB | worker pool {pool['worker_pool_bytes'] / 1024 ** 3:.2f} GiB | reserve {pool['reserve_bytes'] / 1024 ** 3:.2f} GiB | provisional sizing; hardware calibration pending", flush=True)
    for worker_index, worker in enumerate(plan):
        lengths_note = '' if worker.get('lengths') else ' | draws all configured binder lengths'
        estimate = worker.get('metal_estimate_bytes')
        estimate_note = 'unknown' if estimate is None else f'{estimate / 1024 ** 3:.2f} GiB'
        print(f"worker={worker_index} metal=0 allowance={worker['metal_memory_budget_bytes'] / 1024 ** 3:.2f} GiB estimate={estimate_note} | {worker['residue_count']} padded residues{lengths_note}", flush=True)
        if worker.get('lengths'):
            print(f"worker={worker_index} gpu={worker['card']} draws {len(worker['lengths'])} binder lengths from {min(worker['lengths'])} to {max(worker['lengths'])}, folded at {worker['residue_count']} padded residues", flush=True)


def _plan_resource_ceiling(plan, ceiling):
    if not plan:
        raise ValueError('Campaign resource budget cannot admit one design worker')
    budgets = plan[0].get('resource_budgets')
    if not budgets or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in budgets.values()):
        raise ValueError('Managed worker plan requires nonnegative integer resource budgets')
    if ceiling is not None and (not budgets.keys() <= ceiling.keys() or any(value > ceiling[name] for name, value in budgets.items())):
        raise ValueError('Worker replanning cannot enlarge the established campaign resource ceiling')
    totals = dict.fromkeys(budgets, 0)
    for worker in plan:
        reservations = worker.get('resource_reservations', {})
        if worker.get('resource_budgets') != budgets or not reservations or not reservations.keys() <= budgets.keys():
            raise ValueError('Managed worker plan has inconsistent resource maps')
        for name, value in reservations.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError('Managed worker reservations must be nonnegative integer bytes')
            totals[name] += value
        if worker['gpu'] != 'metal' or set(budgets) != {'metal:unified'}:
            raise ValueError('Apple admission requires one unified-memory resource')
        required = {'metal:unified'}
        if not required <= reservations.keys() or any(budgets[name] <= 0 or reservations[name] <= 0 for name in required):
            raise ValueError('An admitted worker requires positive capacity and reservations for each active resource')
    if any(value > budgets[name] for name, value in totals.items()):
        raise ValueError('Managed worker reservations exceed the campaign resource ceiling')
    return {**(ceiling or {}), **budgets}


def _dispatch_generations(settings, log_directory, residue_count, worker_command, project_folders, trajectory_budget, admission):
    from bindcraft._worker_lifecycle import RECYCLE_EXIT_CODE, WorkerRecycleRequested
    from bindcraft.campaign_output import CampaignProgress
    project_folder = project_folders[0]
    progress = CampaignProgress(project_folder, int(settings.get('number_of_final_designs', 1)), trajectory_budget)
    ceiling, idle_replans = None, 0
    while True:
        before = progress.campaign_status()
        if before[0] >= progress.requested_designs or (trajectory_budget and before[1] >= trajectory_budget):
            shared.report_campaign_close(project_folders, progress.requested_designs, trajectory_budget)
            return 0
        profile = admission.current_profile()
        plan = shared.plan_design_workers(settings, residue_count, trajectory_budget, execution_profile=profile, resource_budget=ceiling)
        ceiling = _plan_resource_ceiling(plan, ceiling)
        try:
            context = admission.start_generation(profile, plan)
        except WorkerRecycleRequested:
            idle_replans += 1
            if idle_replans > _MAX_IDLE_REPLANS:
                raise RuntimeError('Campaign execution profile repeatedly changed before workers could start')
            continue
        if context is None:
            shared.report_campaign_close(project_folders, progress.requested_designs, trajectory_budget)
            return 0
        _report_worker_plan(plan, residue_count, trajectory_budget, context)
        try:
            status = shared.launch_design_workers(plan, log_directory, worker_command, shared.design_worker_launch_stagger(settings), project_folders, progress.requested_designs, trajectory_budget, context=context, admission=admission)
        except BaseException as failure:
            # Partial launches can leave never-started reservations. Do not let
            # their finish refusal replace the launch/registration failure.
            try:
                admission.finish_generation(context)
            except Exception as cleanup_error:
                failure.add_note(f'Generation cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}')
            raise
        try:
            # launch records exits only after wait/poll confirms death. Admission
            # refuses this operation if any reservation is still resident.
            admission.finish_generation(context)
        except Exception as cleanup_error:
            if status in (0, RECYCLE_EXIT_CODE):
                raise
            print(f'campaign generation cleanup failed after exit {status}: {type(cleanup_error).__name__}: {cleanup_error}', flush=True)
        if status != RECYCLE_EXIT_CODE:
            shared.report_campaign_close(project_folders, progress.requested_designs, trajectory_budget)
            return status
        after = progress.campaign_status()
        idle_replans = idle_replans + 1 if before == after else 0
        if idle_replans > _MAX_IDLE_REPLANS:
            raise RuntimeError('Campaign workers repeatedly recycled without campaign progress')
        print(f"campaign generation {context['generation']} drained; reassessing campaign progress and execution profile", flush=True)


def dispatch_design_workers(settings: dict, log_directory: str, residue_count: int | None=None, worker_command: list[str] | None=None, project_folders: tuple[str, ...]=(), worker_arguments: tuple[str, ...]=(), trajectory_budget: int | None=None, *, admission=None) -> int | None:
    if shared.running_as_design_worker():
        return None
    from bindcraft.parameter_sweep import parameter_sweep_arms
    if parameter_sweep_arms(settings) or len(project_folders) > 1:
        # Sweep arms have independent campaign progress. Keep the conservative
        # legacy planner until there is an arm-level admission integration.
        print('campaign sweep uses legacy workers; coordinated execution-profile recycling is unavailable for sweep arms', flush=True)
        plan = shared.plan_design_workers(settings, residue_count, trajectory_budget)
        if len(plan) < 2:
            return None
        command = _worker_command(settings, log_directory, worker_command, worker_arguments)
        _report_worker_plan(plan, residue_count, trajectory_budget)
        return shared.launch_design_workers(plan, log_directory, command, shared.design_worker_launch_stagger(settings), project_folders, int(settings.get('number_of_final_designs', 1)), trajectory_budget)
    folders = project_folders or (str(settings.get('project_folder') or os.path.dirname(os.path.abspath(log_directory))),)
    if admission is not None:
        command = _worker_command(settings, log_directory, worker_command, worker_arguments)
        return _dispatch_generations(settings, log_directory, residue_count, command, folders, trajectory_budget, admission)
    from bindcraft._worker_lifecycle import SupervisorAdmission
    with SupervisorAdmission(settings, folders[0], max_trajectories=trajectory_budget) as supervisor:
        command = _worker_command(settings, log_directory, worker_command, worker_arguments)
        return _dispatch_generations(settings, log_directory, residue_count, command, folders, trajectory_budget, supervisor)
