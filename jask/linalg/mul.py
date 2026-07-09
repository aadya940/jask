from ..base import BlockParallelOp, make_jax_op, get_default_policy, DiskArray


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
    a_is_disk = isinstance(a, DiskArray)
    b_is_disk = isinstance(b, DiskArray)

    if a_is_disk and b_is_disk:
        return _elementwise_mul(a, b)
    if a_is_disk and not b_is_disk:
        return _scalar_mul(a, b)
    if b_is_disk and not a_is_disk:
        return _scalar_mul(b, a)
    raise TypeError("mul: at least one argument must be a DiskArray")
