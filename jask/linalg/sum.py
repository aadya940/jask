from ..base import BlockParallelOp
from ..base.base_algo import make_op

import jax.numpy as jnp


class Sum(BlockParallelOp):
    """sum(A) - reduce all elements to a single scalar."""

    def __init__(self, input_block_indices: list[tuple]):
        # Every input block contributes to the single scalar output; the op
        # needs to know how many blocks the input has to enumerate them.
        self.input_block_indices = input_block_indices

    def forward_block(self, a_block):
        return jnp.sum(a_block)

    def index_map(self, out_idx):
        return [(idx,) for idx in self.input_block_indices]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        # d(sum)/d(a_i) = 1, so cotangent = broadcast(d_out, block_shape).
        return (jnp.full(a_block.shape, d_out_block),)

    def output_shape(self, a_shape):
        return ()

    @classmethod
    def from_inputs(cls, a):
        # a is a BlockedArray (or DummyBlocked during aval-inference)
        try:
            return cls(list(a.block_grid()))
        except AttributeError:
            # Dummy input at __init__ time - block indices don't matter yet.
            return cls([])


sum = make_op(Sum)
