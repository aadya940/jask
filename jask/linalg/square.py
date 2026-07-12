from ..base import BlockParallelOp
from ..base.base_algo import make_op


class Square(BlockParallelOp):
    """a**2 elementwise."""

    def forward_block(self, a_block):
        return a_block**2

    def index_map(self, out_idx):
        return [(out_idx,)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        return (2 * a_block * d_out_block,)

    def output_shape(self, a_shape):
        return a_shape


square = make_op(
    Square,
    doc="""Elementwise square of a disk-backed array.

    Computes ``a ** 2`` one tile at a time, never materializing the
    input or output in full.

    Parameters
    ----------
    a : DiskArray
        The array to square.

    Returns
    -------
    DiskArray
        A new disk-backed array of the same shape as `a`, holding
        ``a ** 2``.

    Examples
    --------
    >>> import numpy as np
    >>> import jask
    >>> jask.set_memory_budget("1GB")
    >>> a = jask.DiskArray.from_numpy(np.full((4, 4), 3.0, dtype=np.float32))
    >>> b = jask.square(a)
    >>> np.asarray(b.to_memmap())[0, 0]
    9.0
    """,
)
