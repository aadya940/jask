from ..base import BlockParallelOp
from ..base.base_algo import make_op

import jax.numpy as jnp


class Transpose(BlockParallelOp):
    """a.T (default) or arbitrary axis permutation via `axes`."""

    def __init__(self, axes=None):
        # axes=None means reverse all dims (numpy default).
        self.axes = axes

    def forward_block(self, a_block):
        return jnp.transpose(a_block, self.axes)

    def index_map(self, out_idx):
        # Input block coordinate = axes-inverse-permuted output coordinate.
        if self.axes is None:
            src = tuple(reversed(out_idx))
        else:
            # inverse permutation: given axes=(2,0,1), out_idx[i] comes from in[axes[i]]
            src = tuple(out_idx[self.axes.index(i)] for i in range(len(out_idx)))
        return [(src,)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        # d(transpose)/d = inverse permutation applied to the cotangent.
        if self.axes is None:
            inv = None  # full-reverse is self-inverse
        else:
            n = len(self.axes)
            inv = tuple(sorted(range(n), key=lambda i: self.axes[i]))
        return (jnp.transpose(d_out_block, inv),)

    def output_shape(self, a_shape):
        if self.axes is None:
            return tuple(reversed(a_shape))
        return tuple(a_shape[i] for i in self.axes)

    @classmethod
    def from_inputs(cls, a, axes=None):
        return cls(axes=axes)


transpose = make_op(Transpose)
