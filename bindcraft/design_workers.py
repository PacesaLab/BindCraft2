import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from bindcraft.af2 import campaign_length_bucket, padded_prediction_length
from bindcraft.campaign_output import json_compatible
from bindcraft.protein_preparation import design_residue_count
from bindcraft.settings import build_design_settings

DESIGN_ACTIVATION_BYTES_PER_RESIDUE_PAIR = 38000
DESIGN_MODEL_RESIDENT_GB = 3.4
DESIGN_MEMORY_SAFETY_FACTOR = 2.0
GPU_MEMORY_HEADROOM_GB = 4.0
MAXIMUM_WORKERS_PER_GPU = 8
AUTOMATIC_WORKERS_PER_GPU = 7
TRAJECTORY_ONLY_WORKERS_PER_GPU = 1

def running_as_design_worker() -> bool:
    return 'BINDCRAFT_WORKER_ID' in os.environ

def visible_design_gpus() -> list[str]:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        return [device.strip() for device in visible.split(',') if device.strip()]
    try:
        listing = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in listing.splitlines() if line.strip()]

def selected_design_gpus(gpu_ids: str | list | None=None) -> list[str]:
    gpus = visible_design_gpus()
    if gpu_ids in (None, '', 'all'):
        return gpus
    requested = [str(device).strip() for device in (gpu_ids.split(',') if isinstance(gpu_ids, str) else gpu_ids) if str(device).strip()]
    missing = [device for device in requested if device not in gpus]
    if missing:
        raise ValueError(f'requested GPUs {missing} are not visible to this process')
    return requested

def design_gpu_memory_gb() -> dict[str, tuple[float, float]]:
    try:
        listing = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    fields = [[field.strip() for field in line.split(',')] for line in listing.splitlines() if line.strip()]
    return {index: (float(free_mib) / 1024, float(total_mib) / 1024) for index, free_mib, total_mib in fields}

def estimate_design_memory_gb(residue_count: int) -> float:
    return DESIGN_MEMORY_SAFETY_FACTOR * (DESIGN_MODEL_RESIDENT_GB + DESIGN_ACTIVATION_BYTES_PER_RESIDUE_PAIR * int(residue_count) ** 2 / 1e9)

def design_worker_memory_fractions(free_gb: float, total_gb: float, worker_memory_gb: list[float]) -> list[float]:
    if not total_gb or len(worker_memory_gb) < 2:
        return [0.0] * len(worker_memory_gb)
    packed_share = max(0.0, free_gb - GPU_MEMORY_HEADROOM_GB) / sum(worker_memory_gb)
    return [round(packed_share * memory_gb / total_gb, 3) for memory_gb in worker_memory_gb]

def design_workers_per_gpu(settings: dict, free_gb: float, residue_count: int | None) -> int:
    requested = str(os.environ.get('BINDCRAFT_WORKERS_PER_GPU', settings.get('workers_per_gpu', 'auto'))).lower()
    worker_ceiling = max(1, int(os.environ.get('BINDCRAFT_MAX_WORKERS_PER_GPU', settings.get('max_workers_per_gpu', MAXIMUM_WORKERS_PER_GPU))))
    if requested == 'auto':
        workers_per_gpu = TRAJECTORY_ONLY_WORKERS_PER_GPU if settings.get('trajectory_only') or not (residue_count and free_gb) else AUTOMATIC_WORKERS_PER_GPU
    else:
        workers_per_gpu = int(requested)
    if residue_count and free_gb:
        workers_per_gpu = min(workers_per_gpu, int(max(0.0, free_gb - GPU_MEMORY_HEADROOM_GB) // estimate_design_memory_gb(residue_count)))
    return max(1, min(worker_ceiling, workers_per_gpu))

HOST_MEMORY_PER_WORKER_GB = 4.0

def available_host_memory_gb() -> float:
    try:
        for line in open('/proc/meminfo'):
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 1e6
    except OSError:
        return 0.0
    return 0.0

def host_memory_worker_ceiling(gpu_count: int) -> int:
    available_gb = available_host_memory_gb()
    if not available_gb or gpu_count < 1:
        return MAXIMUM_WORKERS_PER_GPU
    return max(1, int(available_gb // HOST_MEMORY_PER_WORKER_GB) // gpu_count)

SUBBATCH_MEMORY_SHARE = 0.5
PACKED_LAUNCH_STAGGER_SECONDS = 0.0

def campaign_subbatch_size(settings: dict, residue_count: int | None) -> int | None | str:
    requested = settings.get('subbatch_size', 'auto')
    if requested != 'auto' or not residue_count:
        return requested
    free_gb = max((free for free, _ in design_gpu_memory_gb().values()), default=0.0)
    return None if free_gb and estimate_design_memory_gb(residue_count) <= SUBBATCH_MEMORY_SHARE * free_gb else requested

def design_worker_launch_stagger(settings: dict) -> float:
    return float(os.environ.get('BINDCRAFT_WORKER_LAUNCH_STAGGER', settings.get('worker_launch_stagger', PACKED_LAUNCH_STAGGER_SECONDS)))

def campaign_length_buckets(settings: dict) -> tuple[tuple[int, ...], ...]:
    design_settings = build_design_settings(settings)
    binder_lengths = None if design_settings.binder.scaffold else design_settings.binder.lengths  #fold conditioning case
    bucket_size = campaign_length_bucket(settings)
    buckets: dict[int, list[int]] = {}
    for length in sorted(set(binder_lengths or ())):
        buckets.setdefault(padded_prediction_length(length, bucket_size), []).append(length)
    return tuple(tuple(bucket_lengths) for _, bucket_lengths in sorted(buckets.items()))

def assign_worker_length_buckets(plan: list[dict], length_buckets: tuple[tuple[int, ...], ...]) -> list[dict]:
    if len(length_buckets) < 2 or len(plan) < 2:
        return plan
    widest_first = sorted(length_buckets, key=len, reverse=True)
    for worker_index, worker in enumerate(plan):
        share = (widest_first[worker_index % len(length_buckets)],) if len(plan) >= len(length_buckets) else length_buckets[worker_index::len(plan)]
        worker['lengths'] = tuple(length for bucket_lengths in share for length in bucket_lengths)
    return plan

def worker_residue_count(settings: dict, worker: dict, residue_count: int | None) -> int:
    return design_residue_count({**settings, 'binder_lengths': list(worker['lengths'])}) if worker.get('lengths') else residue_count or 0

def plan_design_workers(settings: dict, residue_count: int | None=None, trajectory_budget: int | None=None) -> list[dict]:
    gpus = selected_design_gpus(os.environ.get('BINDCRAFT_GPU_IDS', settings.get('gpu_ids')))
    gpu_memory = design_gpu_memory_gb()
    worker_limit = int(os.environ.get('BINDCRAFT_DESIGN_WORKERS', settings.get('design_workers') or 0))
    plan = []
    host_ceiling = host_memory_worker_ceiling(len(gpus))
    for gpu in gpus:
        packed = min(host_ceiling, design_workers_per_gpu(settings, gpu_memory.get(gpu, (0.0, 0.0))[0], residue_count))
        plan += [{'gpu': gpu, 'memory_fraction': 0.0} for _ in range(packed)]
    worker_limits = [limit for limit in (worker_limit, trajectory_budget) if limit]
    plan = assign_worker_length_buckets(plan[:min(worker_limits)] if worker_limits else plan, campaign_length_buckets(settings))
    for worker in plan:
        worker['residue_count'] = worker_residue_count(settings, worker, residue_count)
    for gpu in dict.fromkeys(worker['gpu'] for worker in plan):
        packed = [worker for worker in plan if worker['gpu'] == gpu]
        needed_gb = [estimate_design_memory_gb(worker['residue_count']) for worker in packed]
        for worker, memory_fraction in zip(packed, design_worker_memory_fractions(*gpu_memory.get(gpu, (0.0, 0.0)), needed_gb)):
            worker['memory_fraction'] = memory_fraction
    return plan

BLOCK_OPENING = '=== trajectory '

def trajectory_block_number(line: str) -> int | None:
    opening = line.strip()
    if not opening.startswith(BLOCK_OPENING):
        return None
    number, _, _ = opening[len(BLOCK_OPENING):].partition(' ')
    return int(number) if number.isdigit() else None

class TrajectoryOrderedConsole:

    def __init__(self) -> None:
        self.console_lock = threading.Lock()
        self.designing: dict[int, int] = {}
        self.finished: dict[int, str] = {}

    def hand_over(self, worker_index: int, trajectory_number: int | None, block: str, designing: int | None) -> None:
        with self.console_lock:
            if designing is None:
                self.designing.pop(worker_index, None)
            else:
                self.designing[worker_index] = designing
            if trajectory_number is None:
                print(block, end='', flush=True)
            elif block:
                self.finished[trajectory_number] = block
            oldest_designing = min(self.designing.values(), default=None)
            for number in sorted(self.finished):
                if oldest_designing is not None and number > oldest_designing:
                    break
                print(self.finished.pop(number), end='', flush=True)

def relay_worker_output(stream, log_file, console: TrajectoryOrderedConsole, worker_index: int=0) -> None:
    block, trajectory_number = [], None
    try:
        for line in stream:
            log_file.write(line)
            opening = trajectory_block_number(line)
            if opening is None:
                block.append(line)
                continue
            console.hand_over(worker_index, trajectory_number, ''.join(block), opening)
            block, trajectory_number = [line], opening
    finally:
        console.hand_over(worker_index, trajectory_number, ''.join(block), None)

def report_campaign_close(project_folders: tuple[str, ...], requested_designs: int=0, max_trajectories: int | None=None) -> None:
    from bindcraft.campaign_log import campaign_budget_exhausted, campaign_closed
    from bindcraft.campaign_output import CampaignProgress, RANKING_METRIC
    for project_folder in project_folders:
        accepted_design_count, trajectory_count = CampaignProgress(project_folder, 0).campaign_status()
        if requested_designs and accepted_design_count < requested_designs and max_trajectories and trajectory_count >= max_trajectories:
            print('\n' + campaign_budget_exhausted(max_trajectories, trajectory_count, accepted_design_count, requested_designs), flush=True)
        print(campaign_closed(accepted_design_count, trajectory_count, RANKING_METRIC), flush=True)

def launch_design_workers(plan: list[dict], log_directory: str, worker_command: list[str], launch_stagger_seconds: float=0.0, project_folders: tuple[str, ...]=(), requested_designs: int=0, max_trajectories: int | None=None) -> int:
    os.makedirs(log_directory, exist_ok=True)
    processes, log_files, relays, console = [], [], [], TrajectoryOrderedConsole()
    workers_on_card, launch_order = {}, []
    for worker_index, worker in enumerate(plan):
        rung = workers_on_card[worker['gpu']] = workers_on_card.get(worker['gpu'], -1) + 1
        launch_order.append((rung, worker_index, worker))
    launched_rung = 0
    try:
        for rung, worker_index, worker in sorted(launch_order):
            if rung != launched_rung and launch_stagger_seconds:
                time.sleep(launch_stagger_seconds)
            launched_rung = rung
            environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': worker['gpu'], 'BINDCRAFT_WORKER_ID': str(worker_index), 'BINDCRAFT_WORKER_COUNT': str(len(plan))}
            if worker.get('memory_fraction'):
                environment['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(worker['memory_fraction'])
            if worker.get('lengths'):
                environment['BINDCRAFT_BINDER_LENGTHS'] = ','.join(str(length) for length in worker['lengths'])
            log_file = open(os.path.join(log_directory, f'worker_{worker_index:02d}_gpu_{worker["gpu"]}.log'), 'a', buffering=1)
            log_files.append(log_file)
            process = subprocess.Popen(worker_command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            processes.append(process)
            relay = threading.Thread(target=relay_worker_output, args=(process.stdout, log_file, console, worker_index), daemon=True)
            relay.start()
            relays.append(relay)
            print(f'worker={worker_index} gpu={worker["gpu"]} log={log_file.name}', flush=True)
        exit_codes = [process.wait() for process in processes]
        for relay in relays:
            relay.join()
        report_campaign_close(project_folders, requested_designs, max_trajectories)
        return next((exit_code for exit_code in exit_codes if exit_code), 0)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for log_file in log_files:
            log_file.close()

def dispatch_design_workers(settings: dict, log_directory: str, residue_count: int | None=None, worker_command: list[str] | None=None, project_folders: tuple[str, ...]=(), worker_arguments: tuple[str, ...]=(), trajectory_budget: int | None=None) -> int | None:
    if running_as_design_worker() or not settings.get('auto_multi_gpu', True):
        return None
    plan = plan_design_workers(settings, residue_count, trajectory_budget)
    if trajectory_budget and len(plan) >= trajectory_budget:
        print(f'campaign fan-out held to {len(plan)} design worker(s) by a budget of {trajectory_budget} trajectories: a worker beyond the budget claims none', flush=True)
    if len(plan) < 2:
        return None
    if worker_command is None:
        os.makedirs(log_directory, exist_ok=True)
        worker_settings_path = os.path.join(log_directory, 'campaign_settings.json')
        Path(worker_settings_path).write_text(json.dumps(json_compatible(settings), sort_keys=True))
        worker_command = [sys.executable, '-u', '-m', 'bindcraft.cli', 'design', worker_settings_path, *worker_arguments]
    memory_note = f' at {estimate_design_memory_gb(residue_count):.1f} GB each' if residue_count and (not any(worker.get('lengths') for worker in plan)) else ''
    print(f"campaign fan-out: {len(plan)} design workers on GPUs {','.join(worker['gpu'] for worker in plan)}{memory_note}", flush=True)
    for worker_index, worker in enumerate(plan):
        if worker.get('lengths'):
            print(f"worker={worker_index} gpu={worker['gpu']} draws {len(worker['lengths'])} binder lengths from {min(worker['lengths'])} to {max(worker['lengths'])}, folded at {worker['residue_count']} padded residues at {estimate_design_memory_gb(worker['residue_count']):.1f} GB", flush=True)
    return launch_design_workers(plan, log_directory, worker_command, design_worker_launch_stagger(settings), project_folders or tuple(folder for folder in (settings.get('project_folder'),) if folder), int(settings.get('number_of_final_designs', 1)), trajectory_budget)
