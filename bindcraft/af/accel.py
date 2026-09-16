import functools

import jax
import jax.numpy as jnp

LOGIT_CLIP = 1e8

MIN_FUSED_LENGTH = 16

FUSED_DTYPES = (jnp.bfloat16, jnp.float16)


def _stock_logits(q, k, bias, nonbatched_bias):
    logits = jnp.einsum('bqhc,bkhc->bhqk', q, k) + bias
    if nonbatched_bias is not None:
        logits = logits + jnp.expand_dims(nonbatched_bias, axis=0)
    return jnp.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)


def _attend_stock(q, k, v, bias, nonbatched_bias):
    weights = jax.nn.softmax(_stock_logits(q, k, bias, nonbatched_bias))
    return jnp.einsum('bhqk,bkhc->bqhc', weights, v)


def _slice_queries(tensor, start, end):
    if tensor is None or tensor.shape[-2] == 1:
        return tensor
    return tensor[..., start:end, :]


def _attend_chunked(q, k, v, bias, nonbatched_bias, chunk_size):
    query_count = q.shape[1]
    if not chunk_size or chunk_size >= query_count:
        return _attend_stock(q, k, v, bias, nonbatched_bias)

    @functools.partial(jax.checkpoint, prevent_cse=False)
    def attend_chunk(query_chunk, bias_chunk, nonbatched_chunk):
        return _attend_stock(query_chunk, k, v, bias_chunk, nonbatched_chunk)

    chunks = [attend_chunk(q[:, start:start + chunk_size],
                           _slice_queries(bias, start, start + chunk_size),
                           _slice_queries(nonbatched_bias, start, start + chunk_size))
              for start in range(0, query_count, chunk_size)]
    return jnp.concatenate(chunks, axis=1)


def _attend_fused(q, k, v, bias, nonbatched_bias, implementation):
    batch, query_count, heads, _ = q.shape
    key_count = k.shape[1]
    combined_bias = bias if nonbatched_bias is None else bias + nonbatched_bias[None]
    combined_bias = jnp.broadcast_to(combined_bias.astype(q.dtype), (batch, heads, query_count, key_count))

    query_pad = query_count % 2
    #cuDNN's fused-attention backward re-materialises the softmax from the saved log-sum-exp, and a query
    #row whose real keys are all masked at -1e9 (a padded, dead residue) has no finite reference: fp32
    #cannot resolve it and the garbage d(bias) it returns leaks into every live pair activation. Always
    #give cuDNN at least one pad key column at -LOGIT_CLIP so those rows have a well-defined backward; two
    #columns where key_count is even keep the padded length even. xla is unaffected and keeps the minimal pad.
    key_pad = (2 - key_count % 2) if implementation == 'cudnn' else key_count % 2
    if query_pad or key_pad:
        q = jnp.pad(q, ((0, 0), (0, query_pad), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, key_pad), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, key_pad), (0, 0), (0, 0)))
        combined_bias = jnp.pad(combined_bias, ((0, 0), (0, 0), (0, query_pad), (0, key_pad)))
        if key_pad:
            combined_bias = combined_bias.at[:, :, :, key_count:].set(jnp.asarray(-LOGIT_CLIP, combined_bias.dtype))

    attended = jax.nn.dot_product_attention(q, k, v, bias=combined_bias, scale=1.0, implementation=implementation)
    return attended[:, :query_count]


def attend(q, k, v, bias, nonbatched_bias=None, backend='stock', chunk_size=256):
    if backend in (None, 'stock'):
        return _attend_stock(q, k, v, bias, nonbatched_bias)
    if backend == 'chunked':
        return _attend_chunked(q, k, v, bias, nonbatched_bias, chunk_size)
    if backend not in ('cudnn', 'xla'):
        raise ValueError(f'unknown attention backend {backend!r}')
    unsupported_dtype = backend == 'cudnn' and q.dtype not in FUSED_DTYPES
    too_short = min(q.shape[1], k.shape[1]) < MIN_FUSED_LENGTH
    if unsupported_dtype or too_short:
        return _attend_chunked(q, k, v, bias, nonbatched_bias, chunk_size)
    return _attend_fused(q, k, v, bias, nonbatched_bias, backend)


def supported_attention_backend(requested: str='auto') -> str:
    if requested != 'auto':
        return requested
    probe_shape = (1, MIN_FUSED_LENGTH, 1, 8)
    probe_query = jnp.zeros(probe_shape, dtype=jnp.bfloat16)
    try:
        jax.block_until_ready(jax.nn.dot_product_attention(probe_query, probe_query, probe_query, scale=1.0, implementation='cudnn'))
    except Exception:
        return 'stock'
    return 'cudnn'


def cuequivariance_available() -> bool:
    try:
        import cuequivariance_jax  # noqa: F401
    except Exception:
        return False
    return True


def fused_triangle_multiplicative_update(activations, mask, parameters, outgoing: bool):
    import cuequivariance_jax as cuequivariance

    original_dtype = activations.dtype

    norm_in, projection, gate = parameters['left_norm_input'], parameters['projection'], parameters['gate']
    center_norm, out_projection, gating = parameters['center_norm'], parameters['output_projection'], parameters['gating_linear']
    projection_weight, projection_bias = projection['weights'].T, projection['bias']
    gate_weight, gate_bias = gate['weights'].T, gate['bias']
    if not outgoing:
        channels = activations.shape[-1]
        swap_halves = lambda tensor: jnp.concatenate([tensor[channels:], tensor[:channels]], axis=0)
        projection_weight, projection_bias = swap_halves(projection_weight), swap_halves(projection_bias)
        gate_weight, gate_bias = swap_halves(gate_weight), swap_halves(gate_bias)
    updated = cuequivariance.triangle_multiplicative_update(
        activations, direction='outgoing' if outgoing else 'incoming', mask=mask,
        norm_in_weight=norm_in['scale'], norm_in_bias=norm_in['offset'],
        p_in_weight=projection_weight, p_in_bias=projection_bias,
        g_in_weight=gate_weight, g_in_bias=gate_bias,
        norm_out_weight=center_norm['scale'], norm_out_bias=center_norm['offset'],
        p_out_weight=out_projection['weights'].T, p_out_bias=out_projection['bias'],
        g_out_weight=gating['weights'].T, g_out_bias=gating['bias'])
    return jnp.asarray(updated).astype(original_dtype)
