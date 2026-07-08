from ..base import BlockParallelOp, make_jax_op, get_default_policy

import jax.numpy as jnp

_SUM_OP_CACHE: dict[tuple, tuple] = {}


class Sum(BlockParallelOp):
    """sum(A) - reduce all elements to a single scalar."""

    def __init__(self, input_block_indices: list[tuple]):
        # Every input block contributes to the single scalar output, the
        # op needs to know how many blocks the input has to enumerate them.
        self.input_block_indices = input_block_indices

    def forward_block(self, a_block):
        return jnp.sum(a_block)

    def index_map(self, out_idx):
        # Single output (out_idx == ()) - every input block feeds it.
        return [(idx,) for idx in self.input_block_indices]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        # d(sum)/d(a_i) = 1, so cotangent of a = broadcast(d_out, a.shape).
        return (jnp.full(a_block.shape, d_out_block),)

    def output_shape(self, a_shape):
        return ()


def sum(a):
    """Sum all elements of a. Returns a shape-() DiskArray."""
    policy = get_default_policy()
    input_block_indices = tuple(a.block_grid())

    cache_key = (
        input_block_indices,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _SUM_OP_CACHE.get(cache_key)
    if cached is None:
        op = Sum(list(input_block_indices))
        jax_op = make_jax_op(op, policy, output_page_shape=())
        _SUM_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a)
