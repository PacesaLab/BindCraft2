import importlib.util
import sys
from bindcraft.model_weights import missing_model_weights, missing_shipped_weights, model_weights

REQUIRED_MODULES = ('jax', 'haiku', 'optax', 'biotite', 'matplotlib', 'numpy', 'scipy', 'ml_collections', 'absl')
ACCELERATOR_MODULES = {'cuda13': ('jax_plugins.xla_cuda13', 'cuequivariance_jax'), 'cuda12': ('jax_plugins.xla_cuda12', 'cuequivariance_jax'), 'rocm': ('jax_plugins.xla_rocm',), 'oneapi': ('jax_plugins.xla_oneapi',)}

def module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False

def missing_modules(accelerator: str='') -> tuple[str, ...]:
    wanted = REQUIRED_MODULES + ACCELERATOR_MODULES.get(accelerator, ())
    return tuple(name for name in wanted if not module_present(name))

def shadowed_api_names() -> tuple[str, ...]:
    from bindcraft import shadowed_names
    return tuple(f'{name} defined in {" and ".join(modules)}, reached from {modules[0]}' for name, modules in sorted(shadowed_names().items()))

def report(accelerator: str='', shipped_only: bool=False) -> int:
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
    raise SystemExit(report(requested[0] if requested else '', '--shipped-only' in sys.argv))
