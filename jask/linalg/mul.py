from ..base import BlockParallelOp, make_jax_op, get_default_policy
from ..base.disk_array import DiskArray, DiskArrayType, BlockedArray

from jax.experimental.hijax import VJPHiPrimitive

_MUL_OP_CACHE: dict[tuple, tuple] = {}
_SCALAR_MUL_OP_CACHE: dict[tuple, tuple] = {}


class Mul(BlockParallelOp):
    """a * b elementwise, both DiskArrays."""

    def forward_block(self, a_block, b_block):
        return a_block * b_block

    def index_map(self, out_idx):
        return [(out_idx, out_idx)]

    def combine(self, acc, partial):
        # Never called (single-entry index_map), but the ABC requires it.
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        # d(a * b)/da = b, d(a * b)/db = a
        return (d_out_block * b_block, d_out_block * a_block)

    def output_shape(self, a_shape, b_shape):
        return a_shape


class ScalarMul(BlockParallelOp):
    """scalar * a - multiply a DiskArray by a Python scalar."""

    def __init__(self, scalar: float):
        self.scalar = float(scalar)

    def forward_block(self, a_block):
        return self.scalar * a_block

    def index_map(self, out_idx):
        return [(out_idx,)]

    def combine(self, acc, partial):
        # Never called (single-entry index_map), but the ABC requires it.
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        # d(scalar * a)/da = scalar
        return (self.scalar * d_out_block,)

    def output_shape(self, a_shape):
        return a_shape


def _elementwise_mul(a, b):
    assert a.full_shape == b.full_shape, "mul: shape mismatch"
    assert a.page_shape == b.page_shape, "mul: page_shape mismatch"

    policy = get_default_policy()
    page_shape = a.page_shape

    cache_key = (
        page_shape,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _MUL_OP_CACHE.get(cache_key)
    if cached is None:
        op = Mul()
        jax_op = make_jax_op(op, policy, page_shape)
        _MUL_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a, b)


def _scalar_mul(a, scalar):
    policy = get_default_policy()
    page_shape = a.page_shape

    cache_key = (
        float(scalar),
        page_shape,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _SCALAR_MUL_OP_CACHE.get(cache_key)
    if cached is None:
        op = ScalarMul(scalar)
        jax_op = make_jax_op(op, policy, page_shape)
        _SCALAR_MUL_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a)


def mul(a, b):
    """a * b - either elementwise (both DiskArrays) or scalar (one is a Python
    number). Order-independent: mul(a, 3.5) and mul(3.5, a) both work."""
    a_is_disk = isinstance(a, BlockedArray)
    b_is_disk = isinstance(b, BlockedArray)

    if a_is_disk and b_is_disk:
        return _elementwise_mul(a, b)
    if a_is_disk and not b_is_disk:
        return _scalar_mul(a, b)
    if b_is_disk and not a_is_disk:
        return _scalar_mul(b, a)
    raise TypeError("mul: at least one argument must be a DiskArray")


# HiJax version


class HiMul(VJPHiPrimitive):
    """Elementwise a * b, both DiskArray."""

    def __init__(self, x_aval: DiskArrayType, y_aval: DiskArrayType):
        assert x_aval.shape == y_aval.shape, "hi_mul: shape mismatch"
        assert x_aval.dtype == y_aval.dtype, "hi_mul: dtype mismatch"
        self.in_avals = (x_aval, y_aval)
        self.out_aval = x_aval
        self.params = {}
        super().__init__()

    def expand(self, x, y):
        result = _elementwise_mul(x._to_blocked(), y._to_blocked())
        return DiskArray._from_blocked(result)

    def vjp_fwd(self, nzs_in, x, y):
        return self(x, y), (x, y)

    def vjp_bwd_retval(self, res, g):
        x, y = res
        # d(a*b)/da = b, d(a*b)/db = a — use dunders (routes to jask.mul).
        return (g * y, g * x)


class HiScalarMul(VJPHiPrimitive):
    """scalar * a, scalar is a Python number stored as a hi-primitive param."""

    def __init__(self, x_aval: DiskArrayType, scalar: float):
        self.in_avals = (x_aval,)
        self.out_aval = x_aval
        self.params = {"scalar": float(scalar)}
        super().__init__()

    def expand(self, x):
        result = _scalar_mul(x._to_blocked(), self.scalar)
        return DiskArray._from_blocked(result)

    def vjp_fwd(self, nzs_in, x):
        return self(x), None

    def vjp_bwd_retval(self, res, g):
        # d(scalar * a)/da = scalar
        return (self.scalar * g,)


def hi_mul(a, b):
    """Disk-backed elementwise mul or scalar mul (order-independent).

    Dispatches on scalar-ness (Python number vs disk-array-like). Uses
    hasattr(x, 'shape') so it works on both real DiskArrays and
    JAX tracers (which also expose .shape/.dtype).
    """
    a_is_scalar = isinstance(a, (int, float))
    b_is_scalar = isinstance(b, (int, float))

    if a_is_scalar and b_is_scalar:
        raise TypeError("hi_mul: at least one argument must be a DiskArray")
    if a_is_scalar:
        op = HiScalarMul(DiskArrayType(b.shape, b.dtype), a)
        return op(b)
    if b_is_scalar:
        op = HiScalarMul(DiskArrayType(a.shape, a.dtype), b)
        return op(a)
    op = HiMul(
        DiskArrayType(a.shape, a.dtype),
        DiskArrayType(b.shape, b.dtype),
    )
    return op(a, b)
