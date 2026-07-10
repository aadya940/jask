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


sub = make_op(Sub)
