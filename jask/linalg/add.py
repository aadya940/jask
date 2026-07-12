from ..base import BlockParallelOp
from ..base.base_algo import make_op


class Add(BlockParallelOp):
    """a + b elementwise."""

    def forward_block(self, a_block, b_block):
        return a_block + b_block

    def index_map(self, out_idx):
        return [(out_idx, out_idx)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        return (d_out_block, d_out_block)

    def output_shape(self, a_shape, b_shape):
        return a_shape


add = make_op(
    Add,
    doc="""Elementwise sum of two disk-backed arrays.

    Computes ``a + b`` one tile at a time, never materializing either
    input or the output in full. Equivalent to ``a + b`` via
    :class:`DiskArray`'s ``__add__``.

    Parameters
    ----------
    a : DiskArray
        First operand.
    b : DiskArray
        Second operand. Must have the same shape and dtype as `a`.

    Returns
    -------
    DiskArray
        A new disk-backed array of the same shape as `a`, holding
        ``a + b``.

    Examples
    --------
    >>> import numpy as np
    >>> import jask
    >>> jask.set_memory_budget("1GB")
    >>> a = jask.DiskArray.from_numpy(np.ones((4, 4), dtype=np.float32))
    >>> b = jask.DiskArray.from_numpy(np.ones((4, 4), dtype=np.float32))
    >>> c = jask.add(a, b)
    >>> np.asarray(c.to_memmap())[0, 0]
    2.0
    """,
)
