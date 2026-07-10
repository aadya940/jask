from ..base import BlockParallelOp
from ..base.base_algo import make_op


class Dot(BlockParallelOp):
    """C = A @ B, tiled over the shared (contraction) dimension."""

    def __init__(self, k_blocks: int):
        self.k_blocks = k_blocks

    def forward_block(self, a_block, b_block):
        return a_block @ b_block

    def index_map(self, out_idx):
        i, j = out_idx
        return [((i, k), (k, j)) for k in range(self.k_blocks)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        d_a = d_out_block @ b_block.T
        d_b = a_block.T @ d_out_block
        return (d_a, d_b)

    def output_shape(self, a_shape, b_shape):
        return (a_shape[0], b_shape[1])

    @classmethod
    def from_inputs(cls, a, b):
        try:
            k_blocks = -(-a.full_shape[1] // a.page_shape[1])
        except AttributeError:
            k_blocks = 1
        return cls(k_blocks=k_blocks)


dot = make_op(Dot)
