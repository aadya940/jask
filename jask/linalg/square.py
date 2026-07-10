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


square = make_op(Square)
