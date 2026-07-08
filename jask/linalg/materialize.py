"""materialize(a) - bridge from disk-backed to in-memory JAX computation.

Reads a DiskArray fully into memory as a real jax.Array. Backward writes
the incoming cotangent to <a.filename>.grad, so gradients flow upstream.

Only use when the value is known to fit in memory - typically at the end
of a disk-backed pipeline, right before computing a scalar loss.
"""

import jax
import numpy as np
from jax.experimental import io_callback

from ..base import DiskArray


def _run_forward(a: DiskArray) -> jax.Array:
    return a.to_jax()


def _run_backward(g, a: DiskArray) -> DiskArray:
    grad_path = a.filename + ".grad"
    mm = np.memmap(grad_path, dtype=a.dtype, mode="w+", shape=a.full_shape)
    mm[...] = np.asarray(g)
    mm.flush()
    return a  # io_callback return, unused


@jax.custom_vjp
def _materialize(a: DiskArray) -> jax.Array:
    return io_callback(
        _run_forward,
        jax.ShapeDtypeStruct(a.full_shape, a.dtype),
        a,
    )


def _materialize_fwd(a):
    return _materialize(a), (a,)


def _materialize_bwd(residuals, g):
    (a,) = residuals
    io_callback(_run_backward, a, g, a)
    return (a,)  # placeholder cotangent - real gradient lives at <a.filename>.grad


_materialize.defvjp(_materialize_fwd, _materialize_bwd)


def materialize(a: DiskArray) -> jax.Array:
    """Read a DiskArray into memory as a real jax.Array (differentiable)."""
    return _materialize(a)
