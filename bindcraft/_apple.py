"""Private Apple startup policy; importing this module never imports JAX.

The governor mapping is retained from the stock metaljax 0.11.9 runtime
and applies to the pinned 0.11.9+fbc3 build. Its
``plugin-native/metal/metal_client.cc`` multiplies the ``*_MB`` settings by
2**20; ``runtime/memory.cc`` treats nonpositive hard limits as disabled.
See https://github.com/cytokineking/metaljax/tree/d5962d9ba7340f8b7b893f9332851d34da31b7d7/plugin-native .
"""

from decimal import Decimal, InvalidOperation, ROUND_CEILING
import importlib.util
import json
import os
import platform
import re
import shlex
import sys
import threading


_MIB = 1 << 20
_GIB = 1 << 30
_configured = False
_requested = False
_budget_bytes: int | None = None
_pool_budget: dict[str, int] | None = None
_summary = ''
_WORKER_POLICY = 'BINDCRAFT_METAL_WORKER_POLICY'
_POOL_FIELDS = frozenset(('budget_bytes', 'coordinator_bytes', 'worker_pool_bytes',
                          'reserve_bytes', 'system_limit_bytes', 'free_floor_bytes'))
_COORDINATOR_BYTES = 2 * _GIB  # Provisional: imports/preflight/log relay, no model parameters.
_JAX_SELECTORS = ('jax_xla_backend', 'jax_platform_name', 'jax_platforms')
_verification_lock = threading.Lock()
_verified_client: object | None = None


def _native_apple() -> bool:
    return platform.system() == 'Darwin' and platform.machine().lower() == 'arm64'


def _plugin_installed() -> bool:
    # This top-level spec lookup does not import metaljax or its PJRT loader.
    try:
        return importlib.util.find_spec('metaljax') is not None
    except (ImportError, ValueError):
        return False


def _jax_choice() -> str:
    """Read an existing JAX config without importing JAX or discovering devices."""
    jax = sys.modules.get('jax')
    config = getattr(jax, 'config', None)
    values = getattr(config, 'values', {}) if config is not None else {}
    # These legacy selectors override the default platform even when
    # jax_platforms lists several backends (JAX 0.11.2 xla_bridge.py).
    # A live config.update() is authoritative over stale environment values.
    for choices in (values, {name: os.environ.get(name.upper(), '') for name in _JAX_SELECTORS}):
        for name in _JAX_SELECTORS:
            value = choices.get(name)
            if value and value.strip():
                return value.split(',')[0].strip().lower()
    return ''


def _backends_initialized() -> bool:
    bridge = sys.modules.get('jax._src.xla_bridge')
    if bridge is None:
        return False
    # Reading JAX's existing state is safe; importing its bridge or asking for
    # a backend here would initialize plugins before the policy is in place.
    check = getattr(bridge, 'backends_are_initialized', None)
    return bool(check()) if check is not None else bool(getattr(bridge, '_backends', {}))


def _sync_imported_jax(*, automatic: bool = False) -> None:
    jax = sys.modules.get('jax')
    if jax is None:
        return
    if automatic:
        jax.config.update('jax_platforms', 'metal')
    elif not any(jax.config.values.get(name) for name in _JAX_SELECTORS):
        explicit = {name: os.environ[name.upper()] for name in _JAX_SELECTORS
                    if os.environ.get(name.upper(), '').strip()}
        if explicit and _backends_initialized():
            raise RuntimeError('JAX backend selection is too late; restart Python and set JAX_PLATFORMS before importing JAX.')
        for name, value in explicit.items():
            jax.config.update(name, value)


def _env_integer(name: str) -> int | None:
    raw = os.environ.get(name, '')
    if not raw:
        return None  # metaljax's EnvInt also treats empty as the default.
    if not re.fullmatch(r'\s*[+-]?[0-9]+', raw):
        raise ValueError(f'{name} must be an integer (MiB for memory limits); got {raw!r}.')
    value = int(raw)
    if not -(1 << 63) <= value < (1 << 63):
        raise ValueError(f'{name} is outside metaljax\'s signed 64-bit integer range.')
    return value


def _governor_environment(budget_mb: int, system_mb: int, free_mb: int) -> dict[str, str]:
    if _env_integer('METALJAX_MEM_GOVERNOR') == 0:
        raise ValueError('BindCraft2 requires the Metal memory governor; unset METALJAX_MEM_GOVERNOR or set it to 1.')
    for name, limit in (('METALJAX_MEM_BUDGET_MB', budget_mb), ('METALJAX_MEM_SYS_MB', system_mb)):
        explicit = _env_integer(name)
        # Zero/negative disables a metaljax hard line, so it is not stricter.
        effective = min(limit, explicit) if explicit is not None and explicit > 0 else limit
        if name == 'METALJAX_MEM_BUDGET_MB':
            budget_mb = effective
        else:
            system_mb = effective
    explicit_floor = _env_integer('METALJAX_MEM_FREE_FLOOR_MB')
    if explicit_floor is not None:
        free_mb = max(free_mb, explicit_floor)
    if free_mb > ((1 << 63) - 1) // _MIB:
        raise ValueError('METALJAX_MEM_FREE_FLOOR_MB is too large for metaljax\'s byte counter.')
    return {
        'METALJAX_MEM_GOVERNOR': '1',
        'METALJAX_MEM_BUDGET_MB': str(budget_mb),
        'METALJAX_MEM_SYS_MB': str(system_mb),
        'METALJAX_MEM_FREE_FLOOR_MB': str(free_mb),
    }


def _memory_policy() -> tuple[dict[str, str], int, str, dict[str, int]]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError('Metal memory policy requires psutil; install BindCraft2 with the [metal] extra.') from exc

    try:
        vm = psutil.virtual_memory()
        total = int(vm.total)
        # psutil's macOS available snapshot is free + inactive, excluding the
        # reclaimable active file cache. Do not count swap or assume all RAM is free.
        available = min(int(vm.available), total)
    except (AttributeError, OSError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError('Cannot read a valid available-memory snapshot for the Metal memory policy.') from exc
    if not 0 < total < (1 << 63) or available < 0:
        raise RuntimeError('Cannot read a valid available-memory snapshot for the Metal memory policy.')
    raw = os.environ.get('BINDCRAFT_METAL_RESERVE_GIB', '16' if total >= 64 * _GIB else '8')
    try:
        reserve_gib = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError('BINDCRAFT_METAL_RESERVE_GIB must be finite and nonnegative.') from exc
    if not reserve_gib.is_finite() or reserve_gib < 0:
        raise ValueError('BINDCRAFT_METAL_RESERVE_GIB must be finite and nonnegative.')
    if reserve_gib >= Decimal(available) / _GIB:
        raise ValueError(
            f'Metal available memory ({available / _GIB:.2f} GiB) minus '
            f'BINDCRAFT_METAL_RESERVE_GIB={raw} must be positive. '
            'Free memory or lower the reserve before starting BindCraft2.'
        )
    reserve = int((reserve_gib * _GIB).to_integral_value(rounding=ROUND_CEILING))
    automatic_mb = (available - reserve) // _MIB
    if automatic_mb <= 0:
        raise ValueError('Metal available memory minus reserve must leave at least 1 MiB for the governor budget.')

    env = _governor_environment(automatic_mb, (total - reserve) // _MIB, (reserve + _MIB - 1) // _MIB)
    budget_mb = int(env['METALJAX_MEM_BUDGET_MB'])
    system_mb = int(env['METALJAX_MEM_SYS_MB'])
    free_mb = int(env['METALJAX_MEM_FREE_FLOOR_MB'])
    # Reuse this one startup snapshot. A stricter system ceiling or free floor
    # can leave less campaign headroom than the native per-process hard limit.
    allowance = max(0, min(budget_mb * _MIB, system_mb * _MIB - (total - available),
                           available - free_mb * _MIB)) // _MIB * _MIB
    coordinator = min(_COORDINATOR_BYTES, allowance)
    pool = {
        'budget_bytes': allowance,
        'coordinator_bytes': coordinator,
        'worker_pool_bytes': allowance - coordinator,
        'reserve_bytes': reserve,
        'system_limit_bytes': system_mb * _MIB,
        'free_floor_bytes': free_mb * _MIB,
    }
    summary = (
        f'Metal memory: reserve {reserve / _GIB:g} GiB; effective process budget '
        f'{budget_mb / 1024:.3f} GiB (automatic {automatic_mb / 1024:.3f} GiB '
        f'from {available / _GIB:.3f} GiB available); system ceiling '
        f'{system_mb / 1024:.3f} GiB; free-memory soft floor {free_mb / 1024:.3f} GiB. '
        'Best-effort headroom, not an OS reservation or preallocation.'
    )
    return env, budget_mb * _MIB, summary, pool


def metal_pool_budget() -> dict[str, int]:
    """Return the campaign's startup snapshot; never probe JAX or resample RAM.

    Coordinator overhead is deducted only here, for a subprocess pool. The
    current-process path retains ``memory_budget_bytes()`` in full.
    """
    configure_before_jax()
    if _pool_budget is None:
        raise RuntimeError('The Metal campaign budget requires Metal selection before importing bindcraft.')
    return dict(_pool_budget)


def _validate_pool(pool: dict[str, int]) -> None:
    if not isinstance(pool, dict) or pool.keys() != _POOL_FIELDS:
        raise ValueError('Metal worker pool must contain exactly the six campaign budget fields.')
    for name, value in pool.items():
        if type(value) is not int or not 0 <= value < (1 << 63):
            raise ValueError(f'Metal worker pool {name} must be a nonnegative signed-64-bit byte count.')
        if name != 'reserve_bytes' and value % _MIB:
            raise ValueError(f'Metal worker pool {name} must be MiB-aligned.')
    if (pool['coordinator_bytes'] + pool['worker_pool_bytes'] > pool['budget_bytes']
            or pool['budget_bytes'] > pool['system_limit_bytes']
            or pool['system_limit_bytes'] <= 0
            or pool['free_floor_bytes'] < pool['reserve_bytes']):
        raise ValueError('Metal worker pool has inconsistent campaign, coordinator, system or reserve limits.')


def metal_worker_environment(worker_budget_bytes: int, *, pool: dict[str, int]) -> dict[str, str]:
    """Serialize a validated child allowance; worker identity is set by the launcher."""
    _validate_pool(pool)
    if (type(worker_budget_bytes) is not int or worker_budget_bytes <= 0
            or worker_budget_bytes % _MIB or worker_budget_bytes > pool['worker_pool_bytes']):
        raise ValueError('Metal worker allowance must be positive, MiB-aligned and within worker_pool_bytes.')
    return {
        'METALJAX_MEM_GOVERNOR': '1',
        'METALJAX_MEM_BUDGET_MB': str(worker_budget_bytes // _MIB),
        'METALJAX_MEM_SYS_MB': str(pool['system_limit_bytes'] // _MIB),
        'METALJAX_MEM_FREE_FLOOR_MB': str(pool['free_floor_bytes'] // _MIB),
        _WORKER_POLICY: json.dumps({'version': 1, 'worker_budget_bytes': worker_budget_bytes, **pool},
                                  sort_keys=True, separators=(',', ':')),
    }


def _worker_memory_policy() -> tuple[dict[str, str], int, str, dict[str, int]]:
    def unique_fields(items):
        result = dict(items)
        if len(result) != len(items):
            raise ValueError('Duplicate Metal worker policy fields.')
        return result

    try:
        policy = json.loads(os.environ[_WORKER_POLICY], object_pairs_hook=unique_fields)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{_WORKER_POLICY} must contain version-1 JSON from the Metal worker launcher.') from exc
    if (not isinstance(policy, dict) or policy.keys() != _POOL_FIELDS | {'version', 'worker_budget_bytes'}
            or type(policy['version']) is not int or policy['version'] != 1):
        raise ValueError(f'{_WORKER_POLICY} must contain exactly the version-1 worker and pool fields.')
    pool = {name: policy[name] for name in _POOL_FIELDS}
    env = metal_worker_environment(policy['worker_budget_bytes'], pool=pool)
    identity, count = os.environ.get('BINDCRAFT_WORKER_ID', ''), os.environ.get('BINDCRAFT_WORKER_COUNT', '')
    if (not re.fullmatch(r'[0-9]+', identity) or not re.fullmatch(r'[0-9]+', count)
            or not 0 <= int(identity) < int(count)
            or int(count) > pool['worker_pool_bytes'] // _MIB):
        raise ValueError(f'{_WORKER_POLICY} requires consistent BINDCRAFT_WORKER_ID and BINDCRAFT_WORKER_COUNT.')
    # Transported reserve/system policy is already resolved. Do not consult
    # psutil or subtract the reserve again in a child; retain stricter limits.
    env.update(_governor_environment(policy['worker_budget_bytes'] // _MIB,
                                    pool['system_limit_bytes'] // _MIB, pool['free_floor_bytes'] // _MIB))
    budget = int(env['METALJAX_MEM_BUDGET_MB']) * _MIB
    summary = (f'Metal worker {identity}/{count}: effective process budget {budget / _GIB:.3f} GiB '
               f'from campaign allowance {pool["budget_bytes"] / _GIB:.3f} GiB; '
               f'inherited reserve {pool["reserve_bytes"] / _GIB:g} GiB (already accounted for). '
               f'System ceiling {int(env["METALJAX_MEM_SYS_MB"]) / 1024:.3f} GiB; '
               f'free-memory soft floor {int(env["METALJAX_MEM_FREE_FLOOR_MB"]) / 1024:.3f} GiB. '
               'Best-effort headroom, not an OS reservation or preallocation.')
    return env, budget, summary, pool


def estimate_metal_worker_memory(settings: dict, residue_count: int, *, shape_count: int = 1,
                                 execution_profile: dict | None = None) -> dict:
    """CPU-only worker reservation, or an unsupported-configuration result."""
    from bindcraft._metal_memory import estimate_worker_memory
    return estimate_worker_memory(settings, residue_count, shape_count=shape_count,
                                  execution_profile=execution_profile)


def estimate_metal_worker_bytes(settings: dict, residue_count: int, *, shape_count: int = 1,
                                execution_profile: dict | None = None) -> int | None:
    """Compatible integer/None adapter; an unknown estimate is never zero."""
    return estimate_metal_worker_memory(settings, residue_count, shape_count=shape_count,
                                       execution_profile=execution_profile)['total_bytes']


def metal_live_worker_capacity(pool: dict[str, int]) -> int:
    """Recheck pressure only after draining; never enlarge the startup pool.

    The free floor already includes the OS reserve. Subtract coordinator
    overhead once, and never credit old workers or presumed allocator caches.
    """
    _validate_pool(pool)
    import psutil
    try:
        vm = psutil.virtual_memory()
        total, available = int(vm.total), int(vm.available)
    except (AttributeError, OSError, TypeError, ValueError, OverflowError) as error:
        raise RuntimeError('Cannot reassess Metal worker capacity after draining') from error
    if not 0 <= available <= total < (1 << 63) or total <= 0:
        raise RuntimeError('Cannot reassess Metal worker capacity from an invalid memory snapshot')
    capacity = min(pool['worker_pool_bytes'],
                   available - pool['free_floor_bytes'] - pool['coordinator_bytes'],
                   pool['system_limit_bytes'] - (total - available) - pool['coordinator_bytes'])
    return max(0, capacity) // _MIB * _MIB


def configure_before_jax() -> None:
    """Select Metal and configure its governor before backend initialization."""
    global _configured, _requested, _budget_bytes, _pool_budget, _summary
    if _configured:
        return
    choice = _jax_choice()
    automatic = not choice and _native_apple() and _plugin_installed()
    requested = choice == 'metal' or automatic
    if not requested:
        _configured = True
        return
    if not _native_apple() or not _plugin_installed():
        raise RuntimeError('Metal requires native Apple Silicon macOS and the BindCraft2 [metal] extra.')
    if _backends_initialized():
        raise RuntimeError(
            'BindCraft2 cannot safely configure Metal after JAX initialized a backend. '
            'Restart Python and import bindcraft before calling jax.devices() or creating JAX arrays; '
            'set JAX_PLATFORMS=cpu before starting Python for intentional CPU execution.'
        )
    for name in ('XLA_FLAGS', 'BC2_XLA_EXTRA'):
        if any(flag.startswith(('--xla_gpu_', '--xla_cuda_')) for flag in shlex.split(os.environ.get(name, ''))):
            raise ValueError(f'Remove CUDA/ROCm XLA flags from {name} before selecting Metal.')
    if _WORKER_POLICY in os.environ:
        env, budget, summary, pool = _worker_memory_policy()
    else:
        if any(name in os.environ for name in ('BINDCRAFT_WORKER_ID', 'BINDCRAFT_WORKER_COUNT')):
            raise ValueError(f'Metal design workers require {_WORKER_POLICY} from the Metal worker launcher.')
        env, budget, summary, pool = _memory_policy()
    # If JAX was imported but no backend was created, its environment config
    # has already been read. Set the automatic choice in that existing config,
    # or copy an explicit environment choice only when the config has none.
    _sync_imported_jax(automatic=automatic)
    if automatic:
        env['JAX_PLATFORMS'] = 'metal'
    os.environ.update(env)
    _requested, _budget_bytes, _pool_budget, _summary = True, budget, pool, summary
    _configured = True


def metal_requested() -> bool:
    """Whether startup selected/requested Metal; this does not probe a device."""
    choice = _jax_choice()
    return choice == 'metal' if choice else _requested


def is_metal_backend() -> bool:
    """Check the actual default JAX backend, initializing it if necessary."""
    import jax
    return jax.default_backend().lower() == 'metal'


def verify_metal_backend() -> None:
    """Verify Metal once per initialized client; always guard against fallback."""
    global _verified_client
    if not metal_requested():
        return
    try:
        import jax
        if not is_metal_backend():
            raise RuntimeError(f'JAX selected {jax.default_backend()!r} instead of Metal.')
        devices = jax.devices()
        if not devices or any(device.platform.lower() != 'metal' for device in devices):
            raise RuntimeError('JAX did not select Metal devices.')
        client = devices[0].client
        with _verification_lock:
            if _verified_client is client:
                return
            import jax.numpy as jnp
            result = jax.jit(lambda value: value + 1)(jnp.asarray(1, dtype=jnp.int32))
            result.block_until_ready()
            if not result.devices() or any(device.platform.lower() != 'metal' for device in result.devices()):
                raise RuntimeError('The verification result did not execute on a Metal device.')
            if int(result) != 2:
                raise RuntimeError('The Metal verification computation returned an incorrect result.')
            # Do not cache a failed/asynchronous attempt as a successful check.
            _verified_client = client
    except Exception as exc:
        raise RuntimeError(
            'Metal was requested but BindCraft2 could not verify Metal execution. '
            'Check the native Apple Silicon [metal] installation and import bindcraft before initializing JAX. '
            f'Details: {exc}'
        ) from exc


def memory_budget_bytes() -> int | None:
    """The effective automatic process budget, after stricter operator limits."""
    return _budget_bytes


def memory_policy_summary() -> str:
    """Describe the applied governor policy without initializing JAX."""
    return _summary
