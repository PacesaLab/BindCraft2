import os
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

MPNN_WEIGHT_VARIANTS = {'neutral': 'weights_neutral', 'negative': 'weights_negative', 'positive': 'weights_positive'}
DEFAULT_MPNN_MODEL = 'v_48_020'
DEFAULT_MPNN_VARIANT = 'negative'
CAMPAIGN_MODELS = tuple(f'model_{index}_multimer_v3' for index in range(1, 6)) + ('model_1_ptm', 'model_2_ptm')
MPNN_CHECKPOINT_FLOOR, ALPHAFOLD_CHECKPOINT_FLOOR = 1 << 20, 100 << 20
ALPHAFOLD_PARAMETER_URL = 'https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar'
ALPHAFOLD_PARAMETER_GIGABYTES = 5.3
SHIPPED_WEIGHTS = Path(__file__).resolve().parent / 'weights'

def weight_cache() -> Path:
    return Path(os.environ.get('BINDCRAFT_WEIGHTS') or Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache') / 'bindcraft')

def proteinmpnn_weights() -> str:
    return os.environ.get('BINDCRAFT_MPNN_WEIGHTS') or str(SHIPPED_WEIGHTS / 'proteinmpnn' / MPNN_WEIGHT_VARIANTS['neutral'])

def mpnn_variant_directory(data_dir: str, variant: str) -> str:
    variant = str(variant).lower()
    if variant not in MPNN_WEIGHT_VARIANTS:
        raise ValueError(f'unknown mpnn_variant {variant!r}; use one of {", ".join(sorted(MPNN_WEIGHT_VARIANTS))}')
    return os.path.join(os.path.dirname(data_dir.rstrip(os.sep)), MPNN_WEIGHT_VARIANTS[variant])

def alphafold_parameter_file(parameters: str, model_name: str) -> str | None:
    candidate_paths = (os.path.join(parameters, 'params', f'params_{model_name}.npz'), os.path.join(parameters, f'params_{model_name}.npz'), os.path.join(parameters, 'params', f'{model_name}.npz'), os.path.join(parameters, f'{model_name}.npz'))
    return next((path for path in candidate_paths if os.path.isfile(path)), None)

def whole_checkpoint(path: str | None, floor_bytes: int) -> bool:
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) >= floor_bytes

def missing_shipped_weights(redesign_weights: str) -> tuple[str, ...]:
    return tuple(f'ProteinMPNN {variant} weights: no checkpoint at {checkpoint}' for variant in sorted(MPNN_WEIGHT_VARIANTS) for checkpoint in (os.path.join(mpnn_variant_directory(redesign_weights, variant), f'{DEFAULT_MPNN_MODEL}.npz'),) if not whole_checkpoint(checkpoint, MPNN_CHECKPOINT_FLOOR))

def missing_model_weights(parameters: str, redesign_weights: str) -> tuple[str, ...]:
    problems = [f'ProteinMPNN {variant} weights: no checkpoint at {checkpoint}' for variant in sorted(MPNN_WEIGHT_VARIANTS) for checkpoint in (os.path.join(mpnn_variant_directory(redesign_weights, variant), f'{DEFAULT_MPNN_MODEL}.npz'),) if not whole_checkpoint(checkpoint, MPNN_CHECKPOINT_FLOOR)]
    for model_name in CAMPAIGN_MODELS:
        checkpoint = alphafold_parameter_file(parameters, model_name) if parameters else None
        if not whole_checkpoint(checkpoint, ALPHAFOLD_CHECKPOINT_FLOOR):
            problems.append(f'AlphaFold {model_name}: {checkpoint or f"no params_{model_name}.npz under {parameters or chr(39)+chr(39)}"}')
    return tuple(problems)

def holds_alphafold_parameters(directory: Path) -> bool:
    return any(directory.glob('params_*.npz')) or any((directory / 'params').glob('params_*.npz'))

def alphafold_parameters(download: bool=True) -> str:
    configured = os.environ.get('BINDCRAFT_AF2_PARAMS')
    if configured:
        return configured
    cached = weight_cache() / 'alphafold'
    found = next((directory for directory in (SHIPPED_WEIGHTS / 'alphafold', cached) if directory.is_dir() and holds_alphafold_parameters(directory)), None)
    if found is not None:
        return str(found)
    return download_alphafold_parameters(cached) if download else ''

def download_alphafold_parameters(destination: Path) -> str:
    print(f'bindcraft: downloading AlphaFold parameters ({ALPHAFOLD_PARAMETER_GIGABYTES:.1f} GB, once) to {destination}', flush=True)
    staging = Path(f'{destination}.incoming.{os.getpid()}')
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True)
        archive = staging / 'alphafold_params.tar'
        with urllib.request.urlopen(ALPHAFOLD_PARAMETER_URL, timeout=60) as response, open(archive, 'wb') as archive_file:
            while chunk := response.read(1 << 22):
                archive_file.write(chunk)
                print(f'\r  {archive_file.tell() / 1e9:.1f} of {ALPHAFOLD_PARAMETER_GIGABYTES:.1f} GB', end='', flush=True)
        print(f'\r  unpacking {archive.name}', flush=True)
        with tarfile.open(archive) as parameters:
            parameters.extractall(staging, filter='data')
        archive.unlink()
    except (OSError, urllib.error.URLError, tarfile.TarError) as failure:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(f'AlphaFold parameters could not be downloaded from {ALPHAFOLD_PARAMETER_URL}: {failure}\nrun "bindcraft fetch-weights" where the network reaches it, or set BINDCRAFT_AF2_PARAMS to a directory holding params_<model>.npz')
    if destination.is_dir():
        shutil.rmtree(staging, ignore_errors=True)
    else:
        staging.rename(destination)
    return str(destination)

def model_weights(download: bool=True) -> tuple[str, str]:
    os.environ['BINDCRAFT_MPNN_WEIGHTS'] = redesign_weights = proteinmpnn_weights()
    parameters = alphafold_parameters(download)
    if parameters:
        os.environ['BINDCRAFT_AF2_PARAMS'] = parameters
    return parameters, redesign_weights
