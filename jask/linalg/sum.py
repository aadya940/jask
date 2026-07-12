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


sum = make_op(
    Sum,
    doc="""Sum all elements of a disk-backed array to a scalar.

    Reduces the whole array one tile at a time, never materializing it
    in full. Unlike every other jask op, the result is small enough to
    return as a real ``jax.Array`` directly rather than a `DiskArray`.

    Parameters
    ----------
    a : DiskArray
        The array to reduce.

    Returns
    -------
    jax.Array
        A real (not disk-backed) scalar `jax.Array` holding the sum of
        every element of `a`.

    Examples
    --------
    >>> import numpy as np
    >>> import jask
    >>> jask.set_memory_budget("1GB")
    >>> a = jask.DiskArray.from_numpy(np.ones((4, 4), dtype=np.float32))
    >>> float(jask.sum(a))
    16.0
    """,
)
