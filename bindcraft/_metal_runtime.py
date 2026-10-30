"""Installed Metal runtime identity, without importing JAX or loading native code.

Controls describe the current environment for subsequent compilation. Callers
must keep it fixed for a predictor's lifetime; this is not a backend probe or a
claim that a particular graph uses the optimized kernels.
"""

from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


METAL_CORE_VERSIONS = {'jax': '0.11.2', 'jaxlib': '0.11.2', 'metaljax': '0.11.9+fbc3'}
# From the exact published wheel above, not the installed package's RECORD.
_NATIVE_SHA256 = {
    'metaljax/lib/libmetal_pjrt_native.dylib': '25230c9a0a5b342c6a14313a06db0844f0e9cf3ce562ac5b8543a3edfd5449e4',
    'metaljax/lib/mlx/lib/libmlx_metaljax.dylib': '72dc8d9a4b1e07df2f6bc91383e0567acf3ce292398ae6ce4ed1d19fce0b1374',
    'metaljax/lib/mlx/lib/libjaccl_metaljax.dylib': '48f5790b66a173170527e6ba7fac3e445787550851439870019581032d5448f9',
}
_REWRITE_CONTROLS = ('METALJAX_EVO_ATTN', 'METALJAX_SLICE_BEFORE_CONVERT')
_ATTENTION_CONTROLS = ('METALJAX_EVO_ATTN_LOGITS', 'METALJAX_EVO_ATTN_KERNEL')


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _file_stamp(path: Path) -> tuple[int, ...]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


@lru_cache(maxsize=16)
def _native_sha256(path: Path, stamp: tuple[int, ...]) -> str:
    # Stat on each lookup, hash only on first use or replacement/modification.
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if _file_stamp(path) != stamp:
        raise OSError('native file changed during verification')
    return digest


def get_runtime_identity() -> dict:
    """Return a JSON-compatible metadata/files/environment snapshot.

    Missing or unrecognized builds remain explicitly unknown. Only matching
    published native bytes establish the fbc3 control semantics; version metadata
    alone does not verify them. No installation paths or timestamps are exposed.
    """
    versions = {name: _version(name) for name in METAL_CORE_VERSIONS}
    native_hashes, problems = {}, []
    try:
        distribution = importlib.metadata.distribution('metaljax')
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    try:
        spec = importlib.util.find_spec('metaljax')
        locations = getattr(spec, 'submodule_search_locations', None)
        package_root = Path(next(iter(locations))) if locations else None
    except (ImportError, ValueError):
        package_root = None
    if distribution is not None:
        for relative, expected in _NATIVE_SHA256.items():
            try:
                # Match the loader's top-level package lookup, including a
                # shadowing source tree. Distribution metadata alone can refer
                # to different files from those Python would actually select.
                path = (package_root / relative.removeprefix('metaljax/')
                        if package_root is not None else Path(distribution.locate_file(relative)))
                native_hashes[relative] = _native_sha256(path, _file_stamp(path))
                if native_hashes[relative] != expected:
                    problems.append(f'native hash mismatch: {relative}')
            except OSError:
                problems.append(f'native file missing, unreadable or changing: {relative}')
    recognized = versions['metaljax'] == METAL_CORE_VERSIONS['metaljax']
    verification = ('verified' if not problems else 'mismatch') if recognized and distribution is not None else 'unknown'
    if package_root is None:
        problems.append('metaljax package location is unknown')
        verification = 'unknown'
    # The loader's explicit override wins over the installed plugin. Its linked
    # runtime is not established by the wheel's hashes, even if this file matches.
    override = os.environ.get('METALJAX_PLUGIN_PATH')
    if override:
        try:
            path = Path(override)
            native_hashes['plugin_override'] = _native_sha256(path, _file_stamp(path))
        except OSError:
            problems.append('native plugin override missing, unreadable or changing')
        problems.append('METALJAX_PLUGIN_PATH overrides the published wheel; unset it to verify the installed runtime')
        verification = 'unknown'
    controls = {}
    for name in _REWRITE_CONTROLS + _ATTENTION_CONTROLS:
        raw = os.environ.get(name)
        # fbc3's C++ gates distinguish unset from empty, and do not trim/parse
        # booleans. Rewrites require exactly "1" when set; subgates reject "0".
        enabled = (raw is None or raw == '1') if name in _REWRITE_CONTROLS else raw != '0'
        controls[name] = {'raw': raw, 'enabled': enabled if verification == 'verified' else None}
    identity = {
        'schema_version': 1,
        'metaljax_version': versions['metaljax'],
        'jax_version': versions['jax'],
        'jaxlib_version': versions['jaxlib'],
        'optimization_controls': controls,
        # Mirror the loader's setdefault rules. This remains stable before and
        # after its precision defaults are exported, without importing the loader.
        'precision_controls': {
            'matmul_precision': (os.environ.get('METALJAX_MATMUL_PRECISION')
                                 if os.environ.get('METALJAX_MATMUL_PRECISION') in ('highest', 'default') else 'high'),
            'mlx_enable_tf32': os.environ.get('MLX_ENABLE_TF32',
                                             None if os.environ.get('METALJAX_MATMUL_PRECISION') == 'highest'
                                             else '1' if os.environ.get('METALJAX_MATMUL_PRECISION') == 'default' else '0'),
            'mlx_metal_gpu_arch': os.environ.get('MLX_METAL_GPU_ARCH',
                                                'applegpu_g16g' if os.environ.get('METALJAX_MATMUL_PRECISION') == 'highest' else None),
            'f64': os.environ.get('METALJAX_F64'),
        },
        'native_verification': verification,
        'native_hashes': native_hashes,
        'native_problems': problems,
    }
    return identity


def runtime_identity_key(identity: dict) -> str:
    """Stable execution identity, independent of install/output paths and time.

    Equivalent effective controls share a key; unknown semantics retain the raw
    value so an unrecognized runtime cannot silently alias a verified profile.
    """
    controls = identity.get('optimization_controls', {})
    effective = {name: (value['enabled'] if value.get('enabled') is not None else {'unknown': value.get('raw')})
                 for name, value in controls.items()}
    relevant = {name: identity.get(name) for name in
                ('schema_version', 'metaljax_version', 'jax_version', 'jaxlib_version', 'native_verification', 'native_hashes', 'precision_controls')}
    relevant['optimization_controls'] = effective
    serialized = json.dumps(relevant, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return 'metal-runtime-v1:' + hashlib.sha256(serialized.encode()).hexdigest()
