import jax

from ..base import BlockParallelOp
from ..base.base_algo import make_op
from ..base.disk_array import DiskArrayType


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


class ScalarMul(BlockParallelOp):
    """scalar * a - multiply a DiskArray by a Python scalar.

    Routed through make_op like every other op (the scalar is forwarded as
    a from_inputs kwarg, same pattern Transpose uses for axes=) so it gets
    the same jit-compatible expand/backward wiring for free, instead of a
    hand-written HiPrimitive that has to duplicate that machinery.
    """

    def __init__(self, scalar: float):
        self.scalar = float(scalar)

    def forward_block(self, a_block):
        return self.scalar * a_block

    def index_map(self, out_idx):
        return [(out_idx,)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        return (self.scalar * d_out_block,)

    def output_shape(self, a_shape):
        return a_shape

    @classmethod
    def from_inputs(cls, a, scalar=1.0):
        return cls(scalar)


_scalar_mul_op = make_op(ScalarMul)


def _scalar_mul(a, scalar):
    return _scalar_mul_op(a, scalar=scalar)


def _is_disk_array(x) -> bool:
    """True if x is DiskArray-typed - checked via its ABSTRACT type
    (jax.typeof), not isinstance(x, DiskArray): when this is called from
    inside a function being jax.jit-traced, x may still be a bare abstract
    tracer (not yet a concrete DiskArray Python object) at the point this
    dispatch runs, and isinstance would wrongly say no."""
    return isinstance(jax.typeof(x), DiskArrayType)


def mul(a, b):
    """a * b - elementwise (both DiskArrays) or scalar (one is not a
    DiskArray - a Python number or a jax scalar, concrete or traced).
    Order-independent: mul(a, 3.5) and mul(3.5, a) both work."""
    a_is_scalar = not _is_disk_array(a)
    b_is_scalar = not _is_disk_array(b)

    if a_is_scalar and b_is_scalar:
        raise TypeError("mul: at least one argument must be a DiskArray")
    if a_is_scalar:
        return _scalar_mul(b, a)
    if b_is_scalar:
        return _scalar_mul(a, b)
    return _elementwise_mul(a, b)
