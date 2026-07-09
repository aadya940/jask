"""materialize(a) - bridge from disk-backed to in-memory JAX computation.

Reads a DiskArray fully into memory as a real jax.Array. Backward writes
the incoming cotangent to <a.filename>.grad, so gradients flow upstream.

Only use when the value is known to fit in memory - typically at the end
of a disk-backed pipeline, right before computing a scalar loss.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.experimental.hijax import VJPHiPrimitive
from jax.core import ShapedArray

from ..base.disk_array import DiskArray, DiskArrayType, BlockedArray


def _run_forward(a: BlockedArray) -> jax.Array:
    return a.to_jax()


def _run_backward(g, a: BlockedArray) -> BlockedArray:
    grad_path = a.filename + ".grad"
    mm = np.memmap(grad_path, dtype=a.dtype, mode="w+", shape=a.full_shape)
    mm[...] = np.asarray(g)
    mm.flush()
    return a  # io_callback return, unused


@jax.custom_vjp
def _materialize(a: BlockedArray) -> jax.Array:
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
    return (a,)


_materialize.defvjp(_materialize_fwd, _materialize_bwd)


def materialize(a: BlockedArray) -> jax.Array:
    """Internal: read a BlockedArray fully into memory as a jax.Array."""
    return _materialize(a)


# HiJax version


class HiMaterialize(VJPHiPrimitive):
    """DiskArray -> jax.Array of the same shape/dtype."""

    def __init__(self, x_aval: DiskArrayType):
        self.in_avals = (x_aval,)
        self.out_aval = ShapedArray(x_aval.shape, x_aval.dtype)
        self.params = {}
        super().__init__()

    def expand(self, x):
        return jnp.asarray(np.asarray(x.to_memmap()))

    def vjp_fwd(self, nzs_in, x):
        return self(x), x

    def vjp_bwd_retval(self, res, g):
        # Cotangent for x = write g (jax.Array) to a fresh DiskArray.
        return (DiskArray.from_numpy(np.asarray(g)),)


def hi_materialize(x: DiskArray) -> jax.Array:
    """Bridge from disk-backed to in-memory JAX computation."""
    op = HiMaterialize(DiskArrayType(x.shape, x.dtype))
    return op(x)
