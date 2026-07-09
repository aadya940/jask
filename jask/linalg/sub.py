from ..base import BlockParallelOp, make_jax_op, get_default_policy
from ..base.disk_array import DiskArray, DiskArrayType

from jax.experimental.hijax import VJPHiPrimitive

_SUB_OP_CACHE: dict[tuple, tuple] = {}


class Sub(BlockParallelOp):
    """a - b elementwise."""

    def forward_block(self, a_block, b_block):
        return a_block - b_block

    def index_map(self, out_idx):
        # Elementwise - output block (i,j) needs a's (i,j) and b's (i,j).
        return [(out_idx, out_idx)]

    def combine(self, acc, partial):
        # Never called (single-entry index_map), but the ABC requires it.
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        # d(a - b)/da = 1, d(a - b)/db = -1
        return (d_out_block, -d_out_block)

    def output_shape(self, a_shape, b_shape):
        return a_shape


def sub(a, b):
    """a - b elementwise. a and b must share full_shape and page_shape."""
    assert a.full_shape == b.full_shape, "sub: shape mismatch"
    assert a.page_shape == b.page_shape, "sub: page_shape mismatch"

    policy = get_default_policy()
    page_shape = a.page_shape

    cache_key = (
        page_shape,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _SUB_OP_CACHE.get(cache_key)
    if cached is None:
        op = Sub()
        jax_op = make_jax_op(op, policy, page_shape)
        _SUB_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a, b)


# HiJax version


class HiSub(VJPHiPrimitive):
    def __init__(self, x_aval: DiskArrayType, y_aval: DiskArrayType):
        assert x_aval.shape == y_aval.shape, "hi_sub: shape mismatch"
        assert x_aval.dtype == y_aval.dtype, "hi_sub: dtype mismatch"
        self.in_avals = (x_aval, y_aval)
        self.out_aval = x_aval
        self.params = {}
        super().__init__()

    def expand(self, x, y):
        result = sub(x._to_blocked(), y._to_blocked())
        return DiskArray._from_blocked(result)

    def vjp_fwd(self, nzs_in, x, y):
        return self(x, y), None

    def vjp_bwd_retval(self, res, g):
        # d(a - b)/da = 1, d(a - b)/db = -1
        return (g, -1.0 * g)


def hi_sub(x: DiskArray, y: DiskArray) -> DiskArray:
    """Disk-backed elementwise subtraction."""
    op = HiSub(
        DiskArrayType(x.shape, x.dtype),
        DiskArrayType(y.shape, y.dtype),
    )
    return op(x, y)
