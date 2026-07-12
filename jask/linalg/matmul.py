import jax.numpy as jnp

from ..base import BlockParallelOp
from ..base.base_algo import make_op


class Dot(BlockParallelOp):
    """Batched matmul: (*batch, M, K) @ (*batch, K, N) -> (*batch, M, N).

    Generalizes to any number of leading batch dims (including zero, the
    plain 2D case) because `@`/jnp.matmul already broadcasts correctly
    over leading dims on its own - the tiled block loop just needs to
    index the last two axes generically instead of assuming exactly 2D.
    Batch dims must match exactly between a and b (no broadcasting yet).
    """

    def __init__(self, k_blocks: int):
        self.k_blocks = k_blocks

    def forward_block(self, a_block, b_block):
        return a_block @ b_block

    def index_map(self, out_idx):
        *batch_idx, i, j = out_idx
        return [
            ((*batch_idx, i, k), (*batch_idx, k, j)) for k in range(self.k_blocks)
        ]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        d_a = d_out_block @ jnp.swapaxes(b_block, -1, -2)
        d_b = jnp.swapaxes(a_block, -1, -2) @ d_out_block
        return (d_a, d_b)

    def output_shape(self, a_shape, b_shape):
        if a_shape[:-2] != b_shape[:-2]:
            raise ValueError(
                f"dot: batch dims must match exactly (no broadcasting yet), "
                f"got {a_shape[:-2]} and {b_shape[:-2]}"
            )
        return a_shape[:-2] + (a_shape[-2], b_shape[-1])

    @classmethod
    def from_inputs(cls, a, b):
        try:
            k_blocks = -(-a.full_shape[-1] // a.page_shape[-1])
        except AttributeError:
            k_blocks = 1
        return cls(k_blocks=k_blocks)


dot = make_op(
    Dot,
    doc="""Matrix multiplication of two disk-backed arrays, with batching.

    Computes ``a @ b`` one tile at a time along the contraction
    dimension, never materializing either input or the output in full.
    Matches ``jnp.matmul``/``@`` semantics: for 2D inputs this is plain
    matrix multiplication; for higher-rank inputs, leading dimensions
    are treated as batch dimensions (broadcast, not contracted).
    Equivalent to ``a @ b`` via :class:`DiskArray`'s ``__matmul__``.

    Note this differs from `numpy.dot`/`jnp.dot`'s own N-D contraction
    rule (which sums over `a`'s last axis and `b`'s second-to-last axis
    without batching) - `jask.dot` always matches `@`.

    Parameters
    ----------
    a : DiskArray
        Left operand, shape ``(*batch, M, K)``.
    b : DiskArray
        Right operand, shape ``(*batch, K, N)``. Batch dimensions must
        match `a`'s exactly (no broadcasting between different batch
        shapes yet).

    Returns
    -------
    DiskArray
        A new disk-backed array of shape ``(*batch, M, N)``.

    Examples
    --------
    >>> import numpy as np
    >>> import jask
    >>> jask.set_memory_budget("1GB")
    >>> a = jask.DiskArray.from_numpy(np.ones((4, 6), dtype=np.float32))
    >>> b = jask.DiskArray.from_numpy(np.ones((6, 3), dtype=np.float32))
    >>> c = jask.dot(a, b)
    >>> np.asarray(c.to_memmap())[0, 0]
    6.0

    Batched (leading dim broadcasts, only the last two axes multiply):

    >>> a3 = jask.DiskArray.from_numpy(np.ones((5, 4, 6), dtype=np.float32))
    >>> b3 = jask.DiskArray.from_numpy(np.ones((5, 6, 3), dtype=np.float32))
    >>> jask.dot(a3, b3).shape
    (5, 4, 3)
    """,
)
