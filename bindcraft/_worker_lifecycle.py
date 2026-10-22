"""Private Apple campaign admission. Importing this module does not import JAX/models.

The coordinator lease spans generations; the existing progress lock protects
only short state changes. Process start identities make orphan recovery fail
closed without signalling a possibly reused PID.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import uuid

RECYCLE_EXIT_CODE = 75
_STATE_KEY = '_worker_lifecycle'
_STARTUP_TIMEOUT = 10.0
_NON_EXECUTION_FIELDS = frozenset({
    'project_folder', 'campaign_name', 'binder_name', 'hash_design_names',
    'number_of_final_designs', 'max_trajectories', 'resume', 'parameter_sweep',
    'auto_multi_gpu', 'design_workers', 'workers_per_gpu', 'max_workers_per_gpu',
    'gpu_ids', 'worker_launch_stagger', 'desperation', 'desperation_trajectories',
    'autotune', 'autotune_loss_weights', 'archive_trajectories', 'sparse_output',
    'compile_next_length',
})


class WorkerRecycleRequested(Exception):
    """A persisted generation transition, never a spent trajectory attempt."""


def _profile_signature(settings: dict) -> str:
    from bindcraft.settings import DEFAULT_SETTINGS
    from bindcraft.campaign_output import json_compatible
    # Cover compiled loss parameters and future model settings as well as model
    # construction options. Ordinary dynamic weights and loop lengths remain
    # legal within a warm process and must not cause recycling.
    configuration = {name: value for name, value in {**DEFAULT_SETTINGS, **settings}.items()
                     if name not in _NON_EXECUTION_FIELDS and not name.startswith(('weights_', 'save_'))
                     and not name.endswith('_steps')}
    configuration['losses'] = {name: {key: value for key, value in entry.items() if key not in ('weight', 'weights')}
                                if isinstance(entry, dict) else {}
                                for name, entry in (configuration.get('losses') or {}).items()}
    # Normal CLI resolution represents inactive filter limits with infinities.
    # Use the campaign's existing canonical JSON spelling for those values;
    # leave the actual scientific settings untouched.
    return hashlib.sha256(json.dumps(json_compatible(configuration), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def execution_profile(settings: dict, project_folder: str) -> dict:
    from bindcraft.desperation import desperation_overrides, desperation_rungs, trajectories_since_accepted
    fruitless = trajectories_since_accepted(project_folder)
    overrides = desperation_overrides(settings, fruitless)
    return {'signature': _profile_signature({**settings, **overrides}),
            'overrides': overrides, 'rung': desperation_rungs(settings, fruitless),
            'fruitless_trajectories': fruitless}


def _validate_profile(profile: dict) -> None:
    if (not isinstance(profile, dict) or not isinstance(profile.get('signature'), str)
            or len(profile['signature']) != 64 or not isinstance(profile.get('overrides'), dict)
            or type(profile.get('rung')) is not int or not 0 <= profile['rung'] <= 7):
        raise ValueError('Malformed worker execution profile')
    try:
        int(profile['signature'], 16)
    except ValueError as error:
        raise ValueError('Malformed worker execution profile signature') from error
    # Restrict profiles to the existing scientific ladder, never arbitrary settings.
    from bindcraft.desperation import DESPERATION_SETTINGS
    if set(profile['overrides']) - set(DESPERATION_SETTINGS):
        raise ValueError('Unknown worker execution profile overrides')
    json.dumps(profile, allow_nan=False)


def apply_execution_profile(settings: dict, profile: dict) -> dict:
    _validate_profile(profile)
    return {**settings, **copy.deepcopy(profile['overrides'])}


def _project(project_folder: str) -> str:
    return os.path.realpath(os.path.abspath(project_folder))


def _process_identity(pid: int) -> dict | None:
    if type(pid) is not int or pid <= 0:
        raise ValueError('Invalid worker PID')
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError as error:
        raise RuntimeError(f'Cannot establish ownership of PID {pid}') from error
    # A zombie has released its allocations. Linux's boot ID/start tick and
    # macOS's kernel start timestamp distinguish reused PIDs without signalling
    # an unrelated process. No process-list command or JAX import is needed.
    if Path('/proc/self/stat').exists():
        try:
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            return None  # Process exited between kill(0) and the identity read.
        if fields[0] == 'Z':
            return None
        start = Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + fields[19]
    else:
        try:
            import psutil
            process = psutil.Process(pid)
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            start = repr(process.create_time())
        except ImportError as error:
            raise RuntimeError('Cannot verify PID identity without /proc or psutil') from error
        except psutil.NoSuchProcess:
            return None
        except psutil.Error as error:
            raise RuntimeError(f'Ambiguous process identity for PID {pid}') from error
    return {'pid': pid, 'start': start, 'host': socket.gethostname()}


def _alive(identity: dict) -> bool:
    if (not isinstance(identity, dict) or identity.get('host') != socket.gethostname()
            or not isinstance(identity.get('start'), str)):
        raise RuntimeError('Ambiguous or remote campaign process ownership')
    current = _process_identity(identity.get('pid'))
    return current == identity


def _validate_context(context: dict, project_folder: str) -> None:
    if (not isinstance(context, dict) or type(context.get('version')) is not int or context['version'] != 1
            or not isinstance(context.get('run_id'), str) or not context['run_id']
            or type(context.get('generation')) is not int or context['generation'] < 1
            or context.get('project_folder') != _project(project_folder)):
        raise ValueError('Malformed or mismatched BINDCRAFT_WORKER_CONTEXT')
    _validate_profile(context.get('profile'))


def _generation(state: dict, context: dict, project_folder: str) -> dict:
    _validate_context(context, project_folder)
    lifecycle = state.get(_STATE_KEY)
    if (not isinstance(lifecycle, dict) or lifecycle.get('version') != 1
            or any(lifecycle.get(name) != context[name] for name in ('run_id', 'generation', 'project_folder', 'profile'))):
        raise RuntimeError('Stale or unowned worker generation')
    return lifecycle


def _worker_record(lifecycle: dict, worker_index: int) -> dict:
    if type(worker_index) is not int or str(worker_index) not in lifecycle.get('workers', {}):
        raise RuntimeError('Worker is not reserved in this generation')
    return lifecycle['workers'][str(worker_index)]


def _worker_index() -> int:
    try:
        index = int(os.environ['BINDCRAFT_WORKER_ID'])
    except (KeyError, ValueError) as error:
        raise ValueError('Managed worker requires BINDCRAFT_WORKER_ID') from error
    return index


def worker_context(project_folder: str) -> dict | None:
    from bindcraft._apple import metal_requested
    if not metal_requested():
        return None
    raw = os.environ.get('BINDCRAFT_WORKER_CONTEXT')
    if raw is None:
        return None
    try:
        context = json.loads(raw)
    except ValueError as error:
        raise ValueError('Malformed BINDCRAFT_WORKER_CONTEXT JSON') from error
    _validate_context(context, project_folder)
    from bindcraft.campaign_output import CampaignProgress
    progress = CampaignProgress(project_folder, 0)
    index, deadline = _worker_index(), time.monotonic() + _STARTUP_TIMEOUT
    identity = _process_identity(os.getpid())
    while True:
        recycle = False
        with progress.locked_progress() as state:
            lifecycle = _generation(state, context, project_folder)
            if lifecycle.get('coordinator_released') or not _alive(lifecycle['coordinator']):
                raise RuntimeError('Worker coordinator exited before startup/admission')
            record = _worker_record(lifecycle, index)
            if os.environ.get('BINDCRAFT_WORKER_COUNT') != str(len(lifecycle['workers'])):
                raise ValueError('Managed worker count does not match its reservations')
            if record.get('identity') is not None:
                if record['identity'] != identity or record.get('returncode') is not None:
                    raise RuntimeError('Worker registration does not own this process')
                if lifecycle['status'] == 'draining':
                    recycle = True
                elif lifecycle['status'] != 'running':
                    raise RuntimeError('Worker generation is not running')
                else:
                    return context
        if recycle:
            raise WorkerRecycleRequested('Registered generation is draining')
        if time.monotonic() >= deadline:
            raise RuntimeError('Timed out waiting for worker registration')
        time.sleep(0.025)


def _claim_admission(state: dict, context: dict, settings: dict, project_folder: str) -> bool:
    """Called *inside* CampaignProgress's lock; return a recycle decision to save."""
    lifecycle = _generation(state, context, project_folder)
    if lifecycle.get('coordinator_released') or not _alive(lifecycle['coordinator']):
        raise RuntimeError('Worker coordinator is no longer alive')
    record = _worker_record(lifecycle, _worker_index())
    if record.get('identity') != _process_identity(os.getpid()) or record.get('returncode') is not None:
        raise RuntimeError('Unregistered worker cannot claim a trajectory')
    if lifecycle['status'] == 'draining':
        return False
    if lifecycle['status'] != 'running':
        raise RuntimeError('Worker generation is not running')
    current = execution_profile(settings, project_folder)
    if current['signature'] != context['profile']['signature']:
        lifecycle.update(status='draining', transition={'profile': current, 'requested_by': _worker_index()})
        return False
    return True


class SupervisorAdmission:
    def __init__(self, settings: dict, project_folder: str, *, max_trajectories: int | None = None):
        from bindcraft.campaign_output import CampaignProgress
        self.settings = copy.deepcopy(settings)
        self.project_folder = _project(project_folder)
        self.progress = CampaignProgress(self.project_folder, int(settings.get('number_of_final_designs', 1)),
                                         settings.get('max_trajectories') if max_trajectories is None else max_trajectories)
        self.run_id = uuid.uuid4().hex
        self._lease = None
        self._identity = None

    def __enter__(self):
        if self._lease is not None:
            raise RuntimeError('Supervisor admission is already entered')
        self._lease = open(os.path.join(self.project_folder, '.worker_supervisor.lock'), 'a+')
        try:
            fcntl.flock(self._lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            recorded_lease = bool(os.fstat(self._lease.fileno()).st_size)
            self._identity = _process_identity(os.getpid())
            # Never let progress's normal CSV recovery erase orphan ownership.
            if os.path.exists(self.progress.state_path):
                try:
                    json.loads(Path(self.progress.state_path).read_text())
                except (ValueError, OSError) as error:
                    raise RuntimeError('Cannot verify campaign ownership from unreadable state') from error
            with self.progress.locked_progress() as state:
                previous = state.get(_STATE_KEY)
                if recorded_lease and not previous:
                    raise RuntimeError('Campaign admission state is missing; cannot verify an earlier pool')
                if previous:
                    self._assert_no_workers(previous)
                    if not previous.get('coordinator_released') and _alive(previous['coordinator']):
                        raise RuntimeError('Campaign coordinator is still alive')
                state[_STATE_KEY] = {'version': 1, 'run_id': self.run_id, 'generation': 0,
                                     'project_folder': self.project_folder, 'coordinator': self._identity,
                                     'status': 'idle', 'workers': {}}
            # The lock file is also a tombstone: normal CSV state recovery must
            # not erase evidence of a potentially surviving pool after a crash.
            self._lease.seek(0)
            self._lease.truncate()
            self._lease.write(self.run_id)
            self._lease.flush()
            os.fsync(self._lease.fileno())
            return self
        except BaseException:
            self._lease.close()
            self._lease = None
            raise

    def __exit__(self, exc_type, exc, traceback):
        try:
            with self.progress.locked_progress() as state:
                lifecycle = self._owned(state)
                # A surviving child must stop admission, and its reservation is
                # retained for a later invocation to verify after process exit.
                lifecycle['status'] = 'draining' if any(record.get('identity') and _alive(record['identity']) for record in lifecycle['workers'].values()) else 'closed'
                lifecycle['coordinator_released'] = True
        finally:
            self._lease.close()
            self._lease = None

    def _owned(self, state: dict) -> dict:
        if self._lease is None:
            raise RuntimeError('Supervisor admission lease is not held')
        lifecycle = state.get(_STATE_KEY, {})
        if lifecycle.get('run_id') != self.run_id or lifecycle.get('coordinator') != self._identity:
            raise RuntimeError('Supervisor admission ownership was lost')
        return lifecycle

    @staticmethod
    def _assert_no_workers(lifecycle: dict, *, recorded: bool = False) -> None:
        if not isinstance(lifecycle.get('workers'), dict):
            raise RuntimeError('Ambiguous worker reservations')
        for record in lifecycle['workers'].values():
            if record.get('identity') and _alive(record['identity']):
                raise RuntimeError('Cannot replace a generation while an old worker remains alive')
            if recorded and record.get('returncode') is None and not record.get('cancelled_before_registration'):
                raise RuntimeError('Worker exit must be recorded before releasing its reservation')

    def current_profile(self) -> dict:
        with self.progress.locked_progress() as state:
            self._owned(state)
            return execution_profile(self.settings, self.project_folder)

    def start_generation(self, profile: dict, plan: list[dict]) -> dict | None:
        _validate_profile(profile)
        with self.progress.locked_progress() as state:
            lifecycle = self._owned(state)
            self._assert_no_workers(lifecycle, recorded=bool(lifecycle['workers']))
            if state['accepted'] >= self.progress.requested_designs or (self.progress.max_trajectories and state['trajectories'] >= self.progress.max_trajectories):
                return None
            current = execution_profile(self.settings, self.project_folder)
            if current['signature'] != profile['signature']:
                raise WorkerRecycleRequested('Execution profile changed before generation installation')
            if not plan:
                raise ValueError('A generation requires at least one worker reservation')
            context = {'version': 1, 'run_id': self.run_id, 'generation': lifecycle['generation'] + 1,
                       'project_folder': self.project_folder, 'profile': copy.deepcopy(current)}
            state[_STATE_KEY] = {**context, 'coordinator': self._identity, 'status': 'running',
                                 'workers': {str(index): {'identity': None, 'returncode': None,
                                                         'reservations': copy.deepcopy(worker.get('resource_reservations', {}))}
                                             for index, worker in enumerate(plan)}}
            return context

    def register_worker(self, context: dict, worker_index: int, pid: int):
        identity = _process_identity(pid)
        if identity is None:
            raise RuntimeError('Worker exited before registration')
        with self.progress.locked_progress() as state:
            self._owned(state)
            lifecycle = _generation(state, context, self.project_folder)
            record = _worker_record(lifecycle, worker_index)
            if record['identity'] is not None:
                raise RuntimeError('Worker reservation has already been registered')
            if lifecycle['status'] not in ('running', 'draining'):
                raise RuntimeError('Cannot register in a closed generation')
            record['identity'] = identity

    def record_worker_exit(self, context: dict, worker_index: int, pid: int, returncode: int):
        rejection = None
        with self.progress.locked_progress() as state:
            self._owned(state)
            lifecycle = _generation(state, context, self.project_folder)
            record = _worker_record(lifecycle, worker_index)
            if not record.get('identity') or record['identity']['pid'] != pid:
                raise RuntimeError('Exit does not match the registered worker')
            if _alive(record['identity']):
                raise RuntimeError('Cannot release a live worker reservation')
            if record['returncode'] is not None and record['returncode'] != returncode:
                raise RuntimeError('Conflicting worker exit record')
            if record['returncode'] is None:
                record['returncode'] = int(returncode)
                if returncode == RECYCLE_EXIT_CODE and not (lifecycle['status'] == 'draining' and lifecycle.get('transition')):
                    record['recycle_rejection'] = 'Worker exit 75 has no persisted transition request'
            rejection = record.get('recycle_rejection')
        # Death is an accounting fact even when its exit code is invalid. Save
        # both before raising so launch cleanup cannot mask the real rejection.
        if rejection:
            raise RuntimeError(rejection)

    def finish_generation(self, context: dict):
        with self.progress.locked_progress() as state:
            self._owned(state)
            lifecycle = _generation(state, context, self.project_folder)
            # The launcher closes its startup pipes and reaps all owned Popen
            # children before this call. An unregistered slot never passed the
            # allocation gate, so it has no process exit to record. Mark its
            # cancellation explicitly; start_generation still rejects pending
            # slots until this finalization has happened.
            for record in lifecycle['workers'].values():
                if record.get('identity') is None:
                    record['cancelled_before_registration'] = True
            self._assert_no_workers(lifecycle, recorded=True)
            lifecycle['status'] = 'finished'


def _python_worker_entry(payload_path: str) -> int:
    """Explicit command for Python/CLI callers; arguments remain private/local."""
    import pickle
    # The payload is created by this invocation, never by campaign input. Gate
    # before restoring a JAX key or importing campaign/model implementations.
    from bindcraft._apple import metal_requested
    raw_context = json.loads(os.environ['BINDCRAFT_WORKER_CONTEXT']) if metal_requested() and 'BINDCRAFT_WORKER_CONTEXT' in os.environ else None
    if raw_context is not None:
        worker_context(raw_context['project_folder'])
    with open(payload_path, 'rb') as source:
        arguments = pickle.load(source)
    from bindcraft.campaign import run_campaign
    run_campaign(**arguments)
    return 0


if __name__ == '__main__':
    import sys
    # run_campaign imports the canonical module name; keep exception identity.
    sys.modules['bindcraft._worker_lifecycle'] = sys.modules[__name__]
    try:
        sys.exit(_python_worker_entry(sys.argv[1]))
    except WorkerRecycleRequested:
        sys.exit(RECYCLE_EXIT_CODE)
