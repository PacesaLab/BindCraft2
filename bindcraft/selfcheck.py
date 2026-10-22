import importlib.metadata
import importlib.util
import platform
import sys
from bindcraft.model_weights import missing_model_weights, missing_shipped_weights, model_weights

REQUIRED_MODULES = ('jax', 'haiku', 'optax', 'biotite', 'matplotlib', 'numpy', 'scipy', 'ml_collections', 'absl')
ACCELERATOR_MODULES = {'cuda13': ('jax_plugins.xla_cuda13', 'cuequivariance_jax'), 'cuda12': ('jax_plugins.xla_cuda12', 'cuequivariance_jax'), 'rocm': ('jax_plugins.xla_rocm',), 'oneapi': ('jax_plugins.xla_oneapi',),
                       'metal': ('jaxlib', 'metaljax', 'jax_plugins.metal', 'psutil')}
METAL_CORE_VERSIONS = {'jax': '0.11.2', 'jaxlib': '0.11.2', 'metaljax': '0.11.9'}

def module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False

def missing_modules(accelerator: str='') -> tuple[str, ...]:
    wanted = REQUIRED_MODULES + ACCELERATOR_MODULES.get(accelerator, ())
    return tuple(name for name in wanted if not module_present(name))

def _metal_installation_problems() -> tuple[str, ...]:
    """Inspect wheel compatibility and metadata without importing or starting JAX."""
    problems = []
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        problems.append('Metal requires native arm64 macOS and arm64 Python; use an Apple Silicon terminal/environment, not Rosetta')
    else:
        try:
            macos_major = int(platform.mac_ver()[0].split('.')[0])
        except ValueError:
            macos_major = 0
        if macos_major < 14:
            problems.append('metaljax 0.11.9 requires macOS 14 or newer')
    if sys.implementation.name != 'cpython' or sys.version_info < (3, 12):
        problems.append('Metal requires CPython 3.12 or newer')
    for distribution, expected in METAL_CORE_VERSIONS.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f'distribution not installed: {distribution}=={expected}; install ".[metal]"')
            continue
        if installed != expected:
            problems.append(f'Metal requires {distribution}=={expected}, found {installed}; reinstall ".[metal]"')
    return tuple(problems)

def _verify_backend(accelerator: str) -> None:
    if accelerator != 'metal':
        return
    from bindcraft._apple import memory_policy_summary, metal_requested, verify_metal_backend
    if not metal_requested():
        raise ValueError('Metal was not selected; run with JAX_PLATFORMS=metal for this check (an explicit CPU/backend selection is preserved)')
    verify_metal_backend()
    if metal_requested():
        print('bindcraft: JAX Metal backend executed a small compiled operation')
        summary = memory_policy_summary()
        if summary:
            print(f'bindcraft: {summary}')

def shadowed_api_names() -> tuple[str, ...]:
    from bindcraft import shadowed_names
    return tuple(f'{name} defined in {" and ".join(modules)}, reached from {modules[0]}' for name, modules in sorted(shadowed_names().items()))

def _report_metal(shipped_only: bool=False, check_backend: bool=False) -> int:
    accelerator = 'metal'
    modules = missing_modules(accelerator)
    installation_problems = _metal_installation_problems()
    parameters, redesign_weights = model_weights(download=False)
    checkpoints = missing_shipped_weights(redesign_weights) if shipped_only else missing_model_weights(parameters, redesign_weights)
    problems = [f'module not installed: {name}' for name in modules] + list(installation_problems) + list(checkpoints)
    if check_backend and not (modules or installation_problems):
        try:
            _verify_backend(accelerator)
        except (RuntimeError, ValueError, ImportError) as failure:
            problems.append(f'backend check failed: {failure}')
    for problem in problems:
        print(f'  {problem}', file=sys.stderr)
    for shadowed in shadowed_api_names():
        print(f'  shadowed API name: {shadowed}', file=sys.stderr)
    return 1 if problems else 0

def report(accelerator: str='', shipped_only: bool=False, check_backend: bool=False) -> int:
    if accelerator == 'metal':
        return _report_metal(shipped_only, check_backend)
    modules = missing_modules(accelerator)
    parameters, redesign_weights = model_weights(download=False)
    checkpoints = missing_shipped_weights(redesign_weights) if shipped_only else missing_model_weights(parameters, redesign_weights)
    for problem in [f'module not installed: {name}' for name in modules] + list(checkpoints):
        print(f'  {problem}', file=sys.stderr)
    for shadowed in shadowed_api_names():
        print(f'  shadowed API name: {shadowed}', file=sys.stderr)
    return 1 if modules or checkpoints else 0

if __name__ == '__main__':
    requested = [argument for argument in sys.argv[1:] if not argument.startswith('-')]
    raise SystemExit(report(requested[0] if requested else '', '--shipped-only' in sys.argv, '--check-backend' in sys.argv))
