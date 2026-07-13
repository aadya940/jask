import os
import tempfile

import numpy as np
import jax
from jax.experimental import io_callback
from jax.experimental.hijax import VJPHiPrimitive, ShapedArray

from ..base import BlockParallelOp
from ..base.base_algo import make_op
from ..base.base_page import get_default_policy, derive_page_shape
from ..base.disk_array import (
    BlockedArray,
    DiskArray,
    DiskArrayType,
    _as_lo,
    _ensure_on_disk,
    _is_tracing,
    _own_fresh_file,
)


class Mul(BlockParallelOp):
    """a * b elementwise, both DiskArrays."""

    def forward_block(self, a_block, b_block):
        return a_block * b_block

    def index_map(self, out_idx):
        return [(out_idx, out_idx)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        return (d_out_block * b_block, d_out_block * a_block)

    def output_shape(self, a_shape, b_shape):
        return a_shape


_elementwise_mul = make_op(Mul)


# ScalarMul is hand-written, not routed through make_op: the scalar is a
# genuine traced input (so it works when passed as a real jit argument,
# e.g. a learning rate - not just a Python literal closed over in the
# function body), and its gradient is a REDUCTION across every block
# (d(scalar*a)/d(scalar) = sum(a * d_out)), structurally unlike a normal
# per-block array gradient. make_op's contract assumes every gradient
# output is spatial/tiled like its input; this needs one tiled loop
# producing a spatial gradient (for a) and a scalar reduction (for the
# scalar) together, which is genuinely different machinery.


def _tiled_scalar_mul_forward(a_path, a_shape, a_dtype, page_shape, scalar, out_path):
    a_blocked = BlockedArray(a_path, a_shape, a_dtype, page_shape)
    out_blocked = BlockedArray.create(out_path, a_shape, a_dtype, page_shape)
    for idx in a_blocked.block_grid():
        block = a_blocked.read_block(idx)
        out_blocked.write_block(idx, np.asarray(scalar * block))
    return np.float32(0.0)


def _tiled_scalar_mul_backward(
    a_path, a_shape, a_dtype, page_shape, scalar, g_path, grad_a_path
):
    a_blocked = BlockedArray(a_path, a_shape, a_dtype, page_shape)
    g_blocked = BlockedArray(g_path, a_shape, a_dtype, page_shape)
    grad_a_blocked = BlockedArray.create(grad_a_path, a_shape, a_dtype, page_shape)
    scalar_grad_total = 0.0
    for idx in a_blocked.block_grid():
        a_block = a_blocked.read_block(idx)
        g_block = g_blocked.read_block(idx)
        grad_a_blocked.write_block(idx, np.asarray(scalar * g_block))
        scalar_grad_total += float(np.sum(a_block * g_block))
    return np.float32(scalar_grad_total)


class HiScalarMulBackward(VJPHiPrimitive):
    """Backward for scalar * a: grad_a (spatial, deterministic .grad path)
    and grad_scalar (a real reduction across every block), as its own
    primitive - vjp_bwd_retval's body only runs once, abstractly, so the
    actual tiled loop must be deferred here, same reasoning as
    make_op's HiOpBackward."""

    def __init__(self, a_ty: DiskArrayType, scalar_ty, g_ty):
        self.in_avals = (a_ty, scalar_ty, g_ty)
        self.out_aval = (a_ty.to_tangent_aval(), scalar_ty)
        self._a_ty = a_ty
        self._scalar_ty = scalar_ty
        self.params = {}
        super().__init__()

    def expand(self, a, scalar, g):
        a_ty = self._a_ty
        grad_a_path = a_ty.to_tangent_aval().filename
        # num_inputs=1: only `a` is an array input - the scalar isn't
        # tiled/block-resident the way an array operand is.
        page_shape = derive_page_shape(
            get_default_policy(), a_ty.dtype, a_ty.shape, num_inputs=1, phase="backward"
        )

        if not _is_tracing(_as_lo(a), _as_lo(scalar), _as_lo(g)):
            a_blocked = _ensure_on_disk(a, page_shape)
            g_blocked = _ensure_on_disk(g, page_shape)
            scalar_grad = _tiled_scalar_mul_backward(
                a_blocked.filename,
                a_ty.shape,
                a_ty.dtype,
                page_shape,
                float(scalar),
                g_blocked.filename,
                grad_a_path,
            )
            return (DiskArray(grad_a_path, a_ty.shape, a_ty.dtype), float(scalar_grad))

        a_path, g_path = a.filename, g.filename

        def run(a_lo, scalar_lo, g_lo):
            return _tiled_scalar_mul_backward(
                a_path,
                a_ty.shape,
                a_ty.dtype,
                page_shape,
                float(scalar_lo),
                g_path,
                grad_a_path,
            )

        scalar_grad = io_callback(
            run,
            jax.ShapeDtypeStruct((), a_ty.dtype),
            _as_lo(a),
            _as_lo(scalar),
            _as_lo(g),
        )
        grad_a = DiskArray(grad_a_path, a_ty.shape, a_ty.dtype, _lo_tracer=scalar_grad)
        return (grad_a, scalar_grad)


class HiScalarMul(VJPHiPrimitive):
    """scalar * a, with the scalar as a genuine traced input - works when
    passed as a real jit argument (e.g. a learning rate), not just a
    Python literal closed over in the function body."""

    def __init__(self, a_ty: DiskArrayType, scalar_ty):
        self.in_avals = (a_ty, scalar_ty)
        fd, out_path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        self._out_path = out_path
        self.out_aval = DiskArrayType(a_ty.shape, a_ty.dtype, out_path)
        self._a_ty = a_ty
        self._scalar_ty = scalar_ty
        self.params = {}
        super().__init__()

    def expand(self, a, scalar):
        a_ty = self._a_ty
        out_path = self._out_path
        page_shape = derive_page_shape(
            get_default_policy(), a_ty.dtype, a_ty.shape, num_inputs=1, phase="forward"
        )

        if not _is_tracing(_as_lo(a), _as_lo(scalar)):
            a_blocked = _ensure_on_disk(a, page_shape)
            _tiled_scalar_mul_forward(
                a_blocked.filename,
                a_ty.shape,
                a_ty.dtype,
                page_shape,
                float(scalar),
                out_path,
            )
            return _own_fresh_file(DiskArray(out_path, a_ty.shape, a_ty.dtype))

        a_path = a.filename

        def run(a_lo, scalar_lo):
            return _tiled_scalar_mul_forward(
                a_path, a_ty.shape, a_ty.dtype, page_shape, float(scalar_lo), out_path
            )

        marker = io_callback(
            run, jax.ShapeDtypeStruct((), a_ty.dtype), _as_lo(a), _as_lo(scalar)
        )
        return DiskArray(out_path, a_ty.shape, a_ty.dtype, _lo_tracer=marker)

    def vjp_fwd(self, nzs_in, a, scalar):
        return self(a, scalar), (a, scalar)

    def vjp_bwd_retval(self, res, g):
        a, scalar = res
        g_ty = self.out_aval.to_tangent_aval()
        backward_op = HiScalarMulBackward(self._a_ty, self._scalar_ty, g_ty)
        return backward_op(a, scalar, g)


def _scalar_mul(a, scalar):
    a_ty = DiskArrayType(a.shape, a.dtype, a.filename)
    scalar_ty = ShapedArray((), a.dtype)
    op = HiScalarMul(a_ty, scalar_ty)
    return op(a, scalar)


def _is_disk_array(x) -> bool:
    """True if x is DiskArray-typed - checked via its ABSTRACT type
    (jax.typeof), not isinstance(x, DiskArray): when this is called from
    inside a function being jax.jit-traced, x may still be a bare abstract
    tracer (not yet a concrete DiskArray Python object) at the point this
    dispatch runs, and isinstance would wrongly say no."""
    return isinstance(jax.typeof(x), DiskArrayType)


def mul(a, b):
    """Elementwise or scalar multiplication of disk-backed arrays.

    Computes ``a * b`` one tile at a time, never materializing either
    input or the output in full. If either argument is not a
    `DiskArray` (a Python number or a jax scalar, concrete or traced -
    e.g. a learning rate passed as a jitted argument), the other is
    scaled by it elementwise. Order-independent: ``mul(a, 3.5)`` and
    ``mul(3.5, a)`` both work. Equivalent to ``a * b`` via
    :class:`DiskArray`'s ``__mul__``/``__rmul__``.

    Parameters
    ----------
    a : DiskArray or scalar
        First operand. At least one of `a`, `b` must be a `DiskArray`.
    b : DiskArray or scalar
        Second operand.

    Returns
    -------
    DiskArray
        A new disk-backed array of the same shape as whichever operand
        is a `DiskArray`.

    Raises
    ------
    TypeError
        If neither `a` nor `b` is a `DiskArray`.

    Examples
    --------
    >>> import numpy as np
    >>> import jask
    >>> jask.set_memory_budget("1GB")
    >>> a = jask.DiskArray.from_numpy(np.full((4, 4), 2.0, dtype=np.float32))

    Elementwise, both operands `DiskArray`:

    >>> b = jask.DiskArray.from_numpy(np.full((4, 4), 3.0, dtype=np.float32))
    >>> np.asarray(jask.mul(a, b).to_memmap())[0, 0]
    6.0

    Scalar, either order - the scalar can be a Python literal or a real
    (even jit-traced) jax value:

    >>> np.asarray(jask.mul(a, 2.5).to_memmap())[0, 0]
    5.0
    >>> np.asarray(jask.mul(2.5, a).to_memmap())[0, 0]
    5.0
    """
    a_is_scalar = not _is_disk_array(a)
    b_is_scalar = not _is_disk_array(b)

    if a_is_scalar and b_is_scalar:
        raise TypeError("mul: at least one argument must be a DiskArray")
    if a_is_scalar:
        return _scalar_mul(b, a)
    if b_is_scalar:
        return _scalar_mul(a, b)
    return _elementwise_mul(a, b)
