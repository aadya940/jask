from ..base import BlockParallelOp, make_jax_op, get_default_policy

_SQUARE_OP_CACHE: dict[tuple, tuple] = {}


class Square(BlockParallelOp):
    """a**2 elementwise."""

    def forward_block(self, a_block):
        return a_block**2

    def index_map(self, out_idx):
        return [(out_idx,)]

    def combine(self, acc, partial):
        # Never called (single-entry index_map), but the ABC requires it.
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        # d(a**2)/da = 2*a
        return (2 * a_block * d_out_block,)

    def output_shape(self, a_shape):
        return a_shape


def square(a):
    """a**2 elementwise."""
    policy = get_default_policy()
    page_shape = a.page_shape

    cache_key = (
        page_shape,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _SQUARE_OP_CACHE.get(cache_key)
    if cached is None:
        op = Square()
        jax_op = make_jax_op(op, policy, page_shape)
        _SQUARE_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a)
