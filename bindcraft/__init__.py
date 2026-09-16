import ast
import functools
import importlib.util
import os
import pkgutil

def append_xla_flags(flags: str) -> None:
    os.environ['XLA_FLAGS'] = f"{os.environ.get('XLA_FLAGS', '')} {flags}".strip()

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
OPERATOR_COMPILATION_CACHE = os.environ.get('JAX_COMPILATION_CACHE_DIR')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', os.path.join(os.environ.get('TMPDIR') or '/tmp', 'bindcraft_xla_cache'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES', 'none')
if 'triton_gemm' not in os.environ.get('XLA_FLAGS', ''):
    append_xla_flags('--xla_gpu_enable_triton_gemm=false')
if os.environ.get('BC2_XLA_EXTRA', '').strip():
    append_xla_flags(os.environ['BC2_XLA_EXTRA'].strip())

COMMAND_ENTRY = 'main'
DEFINITION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
DISPATCHER_MODULE = 'cli'

@functools.cache
def package_modules() -> tuple[str, ...]:
    return tuple(sorted(name for _, name, is_package in pkgutil.iter_modules(__path__) if not is_package and (not name.startswith('_'))))

def module_source(module_name: str) -> str:
    spec = importlib.util.find_spec(f'{__name__}.{module_name}')
    read_source = getattr(spec.loader, 'get_source', None) if spec else None
    return (read_source(spec.name) or '') if read_source else ''

def assigned_names(node: ast.stmt) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
    return [target.id for target in targets if isinstance(target, ast.Name)]

def defined_names(source: str) -> tuple[str, ...]:
    body = ast.parse(source).body
    declared = [node.value for node in body if '__all__' in assigned_names(node)]
    if declared:
        return tuple(ast.literal_eval(declared[-1]))
    defined = [name for node in body for name in ([node.name] if isinstance(node, DEFINITION_NODES) else assigned_names(node))]
    return tuple(name for name in dict.fromkeys(defined) if not name.startswith('_'))

@functools.cache
def module_definitions() -> dict[str, tuple[str, ...]]:
    definitions = {}
    for module_name in package_modules():
        try:
            definitions[module_name] = defined_names(module_source(module_name))
        except (OSError, SyntaxError, ValueError):
            definitions[module_name] = ()
    return definitions

@functools.cache
def public_api() -> dict[str, str]:
    api: dict[str, str] = {}
    for module_name, names in module_definitions().items():
        for name in names if module_name != DISPATCHER_MODULE else ():
            if name != COMMAND_ENTRY:
                api.setdefault(name, module_name)
    return api

@functools.cache
def shadowed_names() -> dict[str, tuple[str, ...]]:
    contributing = {module: names for module, names in module_definitions().items() if module != DISPATCHER_MODULE}
    defining = {name: tuple(module for module, names in contributing.items() if name in names) for name in public_api()}
    return {name: modules for name, modules in defining.items() if len(modules) > 1}

@functools.cache
def command_modules() -> tuple[str, ...]:
    return tuple(name for name, names in module_definitions().items() if COMMAND_ENTRY in names and name != DISPATCHER_MODULE)

def reload_api() -> None:
    for reading in (package_modules, module_definitions, public_api, shadowed_names, command_modules):
        reading.cache_clear()

def __getattr__(name: str):
    if name == '__all__':
        return tuple(sorted(public_api()))
    if name in public_api():
        return getattr(importlib.import_module(f'{__name__}.{public_api()[name]}'), name)
    if name in package_modules():
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

def __dir__() -> list[str]:
    return sorted({*globals(), *public_api(), *package_modules()})
