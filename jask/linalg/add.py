from ..base import BlockParallelOp, make_jax_op, get_default_policy


_ADD_OP_CACHE: dict[tuple, tuple] = {}


class Add(BlockParallelOp):
    """a + b elementwise."""

    def forward_block(self, a_block, b_block):
        return a_block + b_block

    def index_map(self, out_idx):
        # Elementwise - output block (i,j) needs a's (i,j) and b's (i,j).
        return [(out_idx, out_idx)]

    def combine(self, acc, partial):
        # Never called (single-entry index_map), but the ABC requires it.
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        # d(a + b)/da = 1, d(a + b)/db = 1
        return (d_out_block, d_out_block)

    def output_shape(self, a_shape, b_shape):
        return a_shape


def add(a, b):
    """a + b elementwise. a and b must share full_shape and page_shape."""
    assert a.full_shape == b.full_shape, "add: shape mismatch"
    assert a.page_shape == b.page_shape, "add: page_shape mismatch"

    policy = get_default_policy()
    page_shape = a.page_shape

    cache_key = (
        page_shape,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _ADD_OP_CACHE.get(cache_key)
    if cached is None:
        op = Add()
        jax_op = make_jax_op(op, policy, page_shape)
        _ADD_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a, b)
