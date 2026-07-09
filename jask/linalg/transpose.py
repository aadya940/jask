from ..base import BlockParallelOp, make_jax_op, get_default_policy


_TRANSPOSE_OP_CACHE: dict[tuple, tuple] = {}


class Transpose(BlockParallelOp):
    """a.T - swap the two dimensions of a 2D DiskArray."""

    def forward_block(self, a_block):
        return a_block.T

    def index_map(self, out_idx):
        # Output block (i, j) comes from input block (j, i), then transposed.
        i, j = out_idx
        return [((j, i),)]

    def combine(self, acc, partial):
        # Never called (single-entry index_map), but the ABC requires it.
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        # d(a.T)/da is the transpose operation itself - route the cotangent
        # back the same way the forward did (transpose the block).
        return (d_out_block.T,)

    def output_shape(self, a_shape):
        return (a_shape[1], a_shape[0])


def transpose(a):
    """a.T - swap the two dimensions of a 2D DiskArray."""
    assert len(a.full_shape) == 2, "transpose: only 2D arrays supported"

    policy = get_default_policy()
    # Output blocks are (n_page, m_page) if input blocks are (m_page, n_page).
    page_shape = (a.page_shape[1], a.page_shape[0])

    cache_key = (
        a.page_shape,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _TRANSPOSE_OP_CACHE.get(cache_key)
    if cached is None:
        op = Transpose()
        jax_op = make_jax_op(op, policy, page_shape)
        _TRANSPOSE_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a)
