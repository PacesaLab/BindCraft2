"""Small alignment decompositions using device arithmetic on Metal.

The cyclic Jacobi solver, eigenvector JVP and quaternion matrix are adapted
from FreeBindCraft's ``metal_utils.py``, published at commit
``c88c0b4f99302ad6d878f446441c51cb384a818e``. The adaptation preserves BC2's
matrix-product precision and column-vector Kabsch rotation convention. There
are no backend probes or changes to JAX's global lowering registrations here.

The reused arithmetic is distributed under the following license:

MIT License

Copyright (c) 2024 Martin Pacesa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import jax
import jax.numpy as jnp


def _matmul(left, right):
    return jnp.matmul(left, right, precision=jax.lax.Precision.HIGHEST)


def _jacobi4(matrix):
    """Diagonalize symmetric (..., 4, 4) matrices in eight cyclic sweeps."""
    eye = jnp.broadcast_to(jnp.eye(4, dtype=matrix.dtype), matrix.shape)

    def sweep(_, state):
        a, vectors = state
        for p, q in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
            off = a[..., p, q]
            theta = 0.5 * jnp.arctan2(2 * off, a[..., q, q] - a[..., p, p])
            theta = jnp.where(off == 0, 0, theta)
            c, s = jnp.cos(theta), jnp.sin(theta)
            rotation = eye.at[..., p, p].set(c).at[..., q, q].set(c)
            rotation = rotation.at[..., p, q].set(s).at[..., q, p].set(-s)
            a = _matmul(_matmul(rotation.swapaxes(-1, -2), a), rotation)
            vectors = _matmul(vectors, rotation)
        return a, vectors

    diagonal, vectors = jax.lax.fori_loop(0, 8, sweep, (matrix, eye))
    return jnp.diagonal(diagonal, axis1=-2, axis2=-1), vectors


@jax.custom_jvp
def largest_eigenvector(matrix):
    """Return a unit leading eigenvector of a symmetric (..., 4, 4) matrix.

    A repeated leading eigenvalue has no unique eigenvector or derivative.
    Choose the first maximum and use zero derivative within its exactly
    repeated eigenspace. Small nonzero gaps remain unregularized, as in eigh;
    gradients near a degenerate optimum are consequently ill-conditioned.
    """
    values, vectors = _jacobi4(matrix)
    index = jnp.argmax(values, axis=-1)
    return jnp.take_along_axis(vectors, index[..., None, None], axis=-1)[..., 0]


@largest_eigenvector.defjvp
def _largest_eigenvector_jvp(primals, tangents):
    (matrix,), (tangent,) = primals, tangents
    values, vectors = _jacobi4(matrix)
    index = jnp.argmax(values, axis=-1)
    vector = jnp.take_along_axis(vectors, index[..., None, None], axis=-1)[..., 0]
    value = jnp.take_along_axis(values, index[..., None], axis=-1)
    gap = value - values
    valid = gap != 0
    inverse_gap = jnp.where(valid, 1 / jnp.where(valid, gap, 1), 0)
    projected = _matmul(_matmul(vectors.swapaxes(-1, -2), tangent), vector[..., None])[..., 0]
    derivative = _matmul(vectors, (inverse_gap * projected)[..., None])[..., 0]
    return vector, derivative


def quaternion_matrix(matrix):
    """Build the symmetric quaternion matrix for (..., 3, 3) inputs."""
    xx, xy, xz = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    yx, yy, yz = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    zx, zy, zz = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    rows = ((xx + yy + zz, zy - yz, xz - zx, yx - xy),
            (zy - yz, xx - yy - zz, xy + yx, xz + zx),
            (xz - zx, xy + yx, yy - xx - zz, yz + zy),
            (yx - xy, xz + zx, yz + zy, zz - xx - yy))
    return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)


def rotation_from_covariance(covariance):
    """Return BC2's proper column rotation from a weighted A.T @ B matrix.

    The transpose is required because BC2 applies the returned rotation as
    ``(coordinates - center) @ rotation.T``. Quaternion normalization follows
    from the orthogonal Jacobi eigenvectors, including rank-deficient inputs.
    """
    q = largest_eigenvector(quaternion_matrix(covariance.swapaxes(-1, -2)))
    w, x, y, z = (q[..., i] for i in range(4))
    rows = ((w*w + x*x - y*y - z*z, 2*(x*y - w*z), 2*(x*z + w*y)),
            (2*(x*y + w*z), w*w - x*x + y*y - z*z, 2*(y*z - w*x)),
            (2*(x*z - w*y), 2*(y*z + w*x), w*w - x*x - y*y + z*z))
    return jnp.stack([jnp.stack(row, axis=-1) for row in rows], axis=-2)
