"""materialize(a) - bridge from disk-backed to in-memory JAX computation.

Reads a DiskArray fully into memory as a real jax.Array. Backward writes
the incoming cotangent to a deterministic <filename>.grad DiskArray, so
gradients flow upstream.

Only use when the value is known to fit in memory - typically at the end
of a disk-backed pipeline, right before computing a scalar loss.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback
from jax.experimental.hijax import VJPHiPrimitive
from jax.core import ShapedArray

from ..base.disk_array import DiskArray, DiskArrayType, _as_lo, _is_tracing


class HiMaterializeBackward(VJPHiPrimitive):
    """Writes g to x's deterministic <filename>.grad path, as its own
    hi-primitive - vjp_bwd_retval's body only runs once, abstractly (bare
    tracers, no concrete data yet), so the actual write must be deferred
    here instead, the same way make_op's HiOpBackward works."""

    def __init__(self, x_ty: DiskArrayType):
        self.in_avals = (ShapedArray(x_ty.shape, x_ty.dtype),)
        self.out_aval = x_ty.to_tangent_aval()
        self._grad_ty = self.out_aval
        self.params = {}
        super().__init__()

    def expand(self, g):
        grad_ty = self._grad_ty

        if not _is_tracing(_as_lo(g)):
            mm = np.memmap(
                grad_ty.filename, dtype=grad_ty.dtype, mode="w+", shape=grad_ty.shape
            )
            mm[...] = np.asarray(g)
            mm.flush()
            return DiskArray(grad_ty.filename, grad_ty.shape, grad_ty.dtype)

        def run(g_val):
            mm = np.memmap(
                grad_ty.filename, dtype=grad_ty.dtype, mode="w+", shape=grad_ty.shape
            )
            mm[...] = np.asarray(g_val)
            mm.flush()
            return np.float32(0.0)

        marker = io_callback(run, jax.ShapeDtypeStruct((), grad_ty.dtype), g)
        return DiskArray(
            grad_ty.filename, grad_ty.shape, grad_ty.dtype, _lo_tracer=marker
        )

    def vjp_fwd(self, nzs_in, g):
        raise NotImplementedError("second-order grad not supported for materialize")


class HiMaterialize(VJPHiPrimitive):
    """DiskArray -> jax.Array of the same shape/dtype."""

    def __init__(self, x_aval: DiskArrayType):
        self.in_avals = (x_aval,)
        self.out_aval = ShapedArray(x_aval.shape, x_aval.dtype)
        self._x_aval = x_aval
        self.params = {}
        super().__init__()

    def expand(self, x):
        filename, shape, dtype = x.filename, self._x_aval.shape, self._x_aval.dtype

        if not _is_tracing(_as_lo(x)):
            return jnp.asarray(np.asarray(x.to_memmap()))

        def run(marker):
            return np.asarray(np.memmap(filename, dtype=dtype, mode="r", shape=shape))

        return io_callback(run, jax.ShapeDtypeStruct(shape, dtype), _as_lo(x))

    def vjp_fwd(self, nzs_in, x):
        return self(x), None

    def vjp_bwd_retval(self, res, g):
        # Bind a separate primitive instead of writing here directly - see
        # HiMaterializeBackward's docstring.
        backward_op = HiMaterializeBackward(self._x_aval)
        return (backward_op(g),)


def hi_materialize(x: DiskArray) -> jax.Array:
    """Bring a disk-backed array fully into memory as a real jax.Array.

    Reads the whole array in one shot - only use this when `x` is known
    to fit comfortably in memory, typically at the end of a disk-backed
    pipeline (e.g. right before handing a small result to `jax.pmap`, or
    computing a scalar loss). Differentiable: the backward pass writes
    the incoming cotangent back to a disk-backed gradient.

    Parameters
    ----------
    x : DiskArray
        The disk-backed array to materialize. Its full contents will be
        loaded into memory.

    Returns
    -------
    jax.Array
        An ordinary, in-memory `jax.Array` with the same shape, dtype,
        and values as `x`.

    Examples
    --------
    >>> import numpy as np
    >>> import jask
    >>> jask.set_memory_budget("1GB")
    >>> a = jask.DiskArray.from_numpy(np.ones((4, 4), dtype=np.float32))
    >>> real = jask.materialize(a)
    >>> type(real).__name__
    'ArrayImpl'
    """
    op = HiMaterialize(DiskArrayType(x.shape, x.dtype, x.filename))
    return op(x)
