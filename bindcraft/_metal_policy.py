"""Resolve Metal AlphaFold execution settings without loading a backend."""

SCHEMA_VERSION = 1
SUBBATCH_RESIDUE_THRESHOLD = 384
LARGE_COMPLEX_SUBBATCH_SIZE = 256
QUERY_CHUNK_SIZE = 256


def resolve_policy(padded_length: int, *, subbatch_size: int | None | str='auto',
                   attention_backend: str='auto', use_remat: bool=True) -> dict:
    """Resolve settings for an actual traced residue length without loading JAX."""
    if type(padded_length) is not int or padded_length <= 0:
        raise ValueError('Metal padded_length must be a positive integer')
    if subbatch_size == 'auto':
        subbatch_size = LARGE_COMPLEX_SUBBATCH_SIZE if padded_length > SUBBATCH_RESIDUE_THRESHOLD else None
    elif subbatch_size is not None and (type(subbatch_size) is not int or subbatch_size <= 0):
        raise ValueError('Metal subbatch_size must be auto, None, or a positive integer')
    if attention_backend == 'auto':
        attention_backend = 'chunked'
    elif attention_backend not in ('stock', 'chunked'):
        raise ValueError(f'attention backend {attention_backend!r} is unsupported on Metal; use chunked or stock attention')
    if type(use_remat) is not bool:
        raise ValueError('Metal use_remat must be a boolean')
    return dict(schema_version=SCHEMA_VERSION, padded_length=padded_length,
                global_subbatch_size=subbatch_size, attention_backend=attention_backend,
                query_chunk_size=QUERY_CHUNK_SIZE if attention_backend == 'chunked' else None,
                rematerialization=use_remat, dtype='bfloat16')


def policy_key(policy: dict) -> tuple:
    """Return the effective execution identity, independent of requested aliases."""
    return tuple(policy[name] for name in ('schema_version', 'padded_length',
                 'global_subbatch_size', 'attention_backend', 'query_chunk_size',
                 'rematerialization', 'dtype'))
