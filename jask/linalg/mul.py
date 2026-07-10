from ..base import BlockParallelOp
from ..base.base_algo import make_op
from ..base.disk_array import DiskArray, DiskArrayType

from jax.experimental.hijax import VJPHiPrimitive


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


# ScalarMul is a separate manual HiPrimitive because it takes a Python scalar
# alongside the DiskArray - not something make_op handles today (make_op only
# knows how to build ops from actual array inputs).


class ScalarMul(BlockParallelOp):
    """scalar * a - multiply a DiskArray by a Python scalar."""

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


class HiScalarMul(VJPHiPrimitive):
    def __init__(self, x_aval: DiskArrayType, scalar: float):
        self.in_avals = (x_aval,)
        self.out_aval = x_aval
        self.params = {"scalar": float(scalar)}
        super().__init__()

    def expand(self, x):
        from ..base.base_page import get_default_policy
        from ..base.base_algo import OOCAlgorithm

        blocked = x._to_blocked()
        algo = OOCAlgorithm(ScalarMul(self.scalar), get_default_policy())
        result_ba = algo.run_forward([blocked], blocked.page_shape)
        return DiskArray._from_blocked(result_ba)

    def vjp_fwd(self, nzs_in, x):
        return self(x), None

    def vjp_bwd_retval(self, res, g):
        return (self.scalar * g,)


def _scalar_mul(a, scalar):
    op = HiScalarMul(DiskArrayType(a.shape, a.dtype), scalar)
    return op(a)


def mul(a, b):
    """a * b - elementwise (both DiskArrays) or scalar (one is a Python number).
    Order-independent: mul(a, 3.5) and mul(3.5, a) both work."""
    a_is_scalar = isinstance(a, (int, float))
    b_is_scalar = isinstance(b, (int, float))

    if a_is_scalar and b_is_scalar:
        raise TypeError("mul: at least one argument must be a DiskArray")
    if a_is_scalar:
        return _scalar_mul(b, a)
    if b_is_scalar:
        return _scalar_mul(a, b)
    return _elementwise_mul(a, b)
