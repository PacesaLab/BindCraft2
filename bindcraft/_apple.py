"""Private Apple startup policy; importing this module never imports JAX.

The governor mapping is for the pinned metaljax 0.11.9 runtime. Its
``plugin-native/metal/metal_client.cc`` multiplies the ``*_MB`` settings by
2**20; ``runtime/memory.cc`` treats nonpositive hard limits as disabled.
See https://github.com/eterevsky/metaljax/tree/v0.11.9/plugin-native .
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


def estimate_metal_worker_bytes(settings: dict, residue_count: int, *, shape_count: int = 1,
                                execution_profile: dict | None = None) -> int | None:
    """Provisional campaign allowance, NOT a hardware-calibrated fit claim.

    Envelope: <=512 padded residues, <=16 assigned buckets, one structural
    target/one binder chain, rematerialized chunked attention, <=3 recycles.
    The finite default desperation ladder is included without changing science:
    flexibility 0/0.5, validation initial coordinates, design recycles up to 3,
    narrowing the active design pool and replacing monomer/multimer validation.
    Sweeps, extra design states and relaxation remain outside this envelope.

    In GiB: 2 runtime + 0.5 per resident AF parameter set + 0.25 per retained
    compiled entry; active workspace 4/6 plus compilation 3/4 at <=256/512
    residues; 2 for sequential MPNN/validation outputs; then 25% margin. Charge
    all design parameters even when the active pool narrows. With desperation,
    include the outer campaign's original validator, a current replacement and
    an incoming replacement's parameters. The two populated validators and the
    design predictor retain independent 16-entry caches (campaign.py/af2.py).
    Bound design forward/gradient variants by recycle x flexibility x bucket;
    validation also has model families, bound/unbound shapes and initial-guess
    variants. All entries are charged at the largest assigned shape. Workspace
    covers three recycles/initial coordinates throughout the envelope. Wave 4
    must calibrate footprints, overlapping compilation and delayed reclamation.
    No CUDA coefficients, device queries or executable disk cache assumptions.

    A supervised execution_profile removes only cross-configuration retention:
    size the currently constructed design/validation parameters and their two
    design LRUs and one validation LRU, retaining beta-sheet recycle variants.
    Keep the original workspace/compilation coefficients and 25% margin. Add
    1 GiB per active design recycle above one, 0.5 GiB each for flexible target
    and initial coordinates, 0.5 GiB MPNN parameters, up to 16 cached functions
    with each candidate batch variant, and 2 GiB downstream buffers per ten
    sequence candidates (rounded up). A configured next-length compiler adds
    another 3/4 GiB cold peak. These extra stress allowances are uncalibrated. They do not modify
    settings or reserve future ladder rungs. Unsupported profiles return None.
    """
    managed = execution_profile is not None
    if managed:
        from bindcraft._worker_memory import pinned_settings
        settings = pinned_settings(settings, execution_profile)
    if (not isinstance(settings, dict) or type(residue_count) is not int or not 1 <= residue_count <= 512
            or type(shape_count) is not int or not 1 <= shape_count <= 16):
        return None
    targets = settings.get('targets', [])
    if (not isinstance(targets, (list, tuple)) or len(targets) != 1 or not isinstance(targets[0], dict)
            or not targets[0].get('target_path')
            or str(targets[0]['target_path']).lower().endswith(('.fasta', '.fa', '.faa'))
            or targets[0].get('objective', 'target') != 'target' or targets[0].get('weight', 1) != 1
            or type(settings.get('copies', 1)) is not int or settings.get('copies', 1) != 1
            or settings.get('binder_scaffold') or settings.get('binder_shapes')
            or settings.get('parameter_sweep')
            or settings.get('relax_accepted_designs') or not settings.get('use_remat', True)
            or settings.get('attention_backend', 'auto') not in ('auto', 'chunked')
            or settings.get('subbatch_size', 'auto') != 'auto' or settings.get('use_cueq')
            or (settings.get('compile_next_length') and not managed) or settings.get('cyclize_peptide')
            or settings.get('target_flexibility', 0) not in (0, 0.5)
            or type(settings.get('large_complex_initialization', False)) is not bool
            or type(settings.get('hard_target', False)) is not bool):
        return None
    losses = settings.get('losses') or {}
    ordinary = {'plddt_loss', 'ptm_loss', 'iptm_loss', 'binder_pae', 'interface_pae',
                'binder_contacts', 'interface_contacts', 'binder_helicity', 'compactness'}
    if any(name.startswith('weights_') and name.removeprefix('weights_') not in ordinary and value
           for name, value in settings.items()):
        return None
    if not isinstance(losses, dict) or any(
            name not in ordinary or not isinstance(entry, dict)
            or not isinstance(entry.get('params', {}), dict)
            or entry.get('prediction_state', 'complex') != 'complex'
            or entry.get('params', {}).get('prediction_state', 'complex') != 'complex'
            for name, entry in losses.items()):
        return None
    recycles = (settings.get('design_recycles', 1), settings.get('validation_recycles', 3),
                settings.get('betasheet_reopt_recycles', settings.get('design_recycles', 1)))
    if any(type(value) is not int or not 0 <= value <= 3 for value in recycles):
        return None
    validation = settings.get('validation_model') or 'monomer'
    if validation not in ('monomer', 'multimer'):
        return None
    models = []
    for name, default in (('validation_models', 2), ('design_models', None)):
        default = (5 - models[0] if validation == 'multimer' else 5) if default is None else default
        value = settings.get(name)
        value = default if value is None else value
        count = value if type(value) is int else len(value) if isinstance(value, (list, tuple)) else 0
        if type(count) is not int or not 1 <= count <= 5:
            return None
        models.append(count)
    if managed:
        # The selector rejects explicit null design, but null validation counts
        # zero reserved models when deriving the default design pool. That can
        # select all five and then use the held-out monomer fallback.
        if 'design_models' in settings and settings['design_models'] is None:
            return None
        if (validation == 'multimer' and 'design_models' not in settings
                and 'validation_models' in settings and settings['validation_models'] is None):
            models[1] = 5
        if validation == 'multimer':
            held_out = 5 - models[1]
            if settings.get('validation_models') is None:
                models[0] = held_out or 2
            # The constructor falls back to the two monomer checkpoints when
            # the requested held-out multimer count is exhausted.
            if models[0] > held_out and models[0] > 2:
                return None
        elif models[0] > 2:
            return None
    else:
        if validation == 'multimer' and settings.get('validation_models') is None:
            models[0] = 5 - models[1] or 2  # Legacy all-ladder held-out bound.
        if validation == 'multimer' and sum(models) > 5:
            return None
    redesign = not settings.get('trajectory_only')
    desperate = not managed and bool(settings.get('desperation', True)) and redesign
    resident = models[1] + (models[0] if redesign else 0)
    # The ladder changes active models, not the original resident design pool.
    # Implicit multimer validation uses the remaining held-out models; with an
    # explicit design count it can exceed the initial two monomer validators.
    replacement_models = (5 - models[1] if settings.get('design_models') is not None else 2)
    if settings.get('validation_models') is not None:
        replacement_models = models[0]
    replacement_models = max(models[0], replacement_models or 2)
    if desperate:
        resident += 2 * replacement_models  # Current + incoming, original already counted.
    recycle_variants = len(set(recycles[::2]) | ({3} if desperate else set()))
    flexibility_variants = len({settings.get('target_flexibility', 0)} | ({0.5} if desperate else set()))
    design_entries = 2 * min(16, shape_count * recycle_variants * flexibility_variants)
    validation_entries = 0
    if redesign:
        initial_variants = 1 if settings.get('hard_target') or settings.get('large_complex_initialization') else 1 + desperate
        # Managed constructors use either held-out multimers (one family) or
        # MONOMER_POOL's model_1_ptm/model_2_ptm, whose CONFIG_DIFFS are identical.
        # Parameters remain distinct, but the prediction LRU key is the family,
        # not the checkpoint name. Preserve the old unmanaged ladder bound.
        validation_families = 1 if managed else models[0]
        validation_entries = min(16, 2 * shape_count * flexibility_variants * initial_variants * validation_families)
        if desperate:
            validation_entries += min(16, 2 * shape_count * flexibility_variants * initial_variants * replacement_models)
    units = 8 + 2 * resident + design_entries + validation_entries  # Units are 0.25 GiB.
    units += 28 if residue_count <= 256 else 40  # Live workspace + concurrent compilation.
    units += 8 if redesign else 0
    if managed:
        units += 4 * max(0, max(recycles[::2]) - 1)
        units += 2 * bool(settings.get('target_flexibility'))
        units += 2 * bool(settings.get('hard_target') or settings.get('large_complex_initialization'))
        if settings.get('compile_next_length'):
            units += 12 if residue_count <= 256 else 16  # A second cold compiler can overlap.
        if redesign:
            candidates = settings.get('sequence_candidates', 10)
            if type(candidates) is not int or candidates < 1:
                return None
            # Redraws can use each remaining batch size, all in this profile.
            # The LRU bounds Python functions by chain shape; each function's
            # JIT can retain all remaining-candidate batch sizes independently.
            units += 2 + min(16, shape_count) * candidates
            units += 8 * ((candidates + 9) // 10 - 1)
    return units * 5 * (_GIB // 16)  # 25% margin; 1/16 GiB is still MiB-aligned.


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
