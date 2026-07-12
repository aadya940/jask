from ..base import BlockParallelOp
from ..base.base_algo import make_op


class Sub(BlockParallelOp):
    """a - b elementwise."""

    def forward_block(self, a_block, b_block):
        return a_block - b_block

    def index_map(self, out_idx):
        return [(out_idx, out_idx)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        return (d_out_block, -d_out_block)

    def output_shape(self, a_shape, b_shape):
        return a_shape


sub = make_op(
    Sub,
    doc="""Elementwise difference of two disk-backed arrays.

    Computes ``a - b`` one tile at a time, never materializing either
    input or the output in full. Equivalent to ``a - b`` via
    :class:`DiskArray`'s ``__sub__``.

    Parameters
    ----------
    a : DiskArray
        First operand (the minuend).
    b : DiskArray
        Second operand (the subtrahend). Must have the same shape and
        dtype as `a`.

    Returns
    -------
    DiskArray
        A new disk-backed array of the same shape as `a`, holding
        ``a - b``.

    Examples
    --------
    >>> import numpy as np
    >>> import jask
    >>> jask.set_memory_budget("1GB")
    >>> a = jask.DiskArray.from_numpy(np.full((4, 4), 5.0, dtype=np.float32))
    >>> b = jask.DiskArray.from_numpy(np.ones((4, 4), dtype=np.float32))
    >>> c = jask.sub(a, b)
    >>> np.asarray(c.to_memmap())[0, 0]
    4.0
    """,
)
