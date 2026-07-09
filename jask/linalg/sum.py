from ..base import BlockParallelOp, make_jax_op, get_default_policy
from ..base.disk_array import DiskArray, DiskArrayType

import numpy as np
import jax
import jax.numpy as jnp
import tempfile
import os
from jax.experimental.hijax import VJPHiPrimitive
from jax.core import ShapedArray

_SUM_OP_CACHE: dict[tuple, tuple] = {}


def _broadcast_scalar_to_disk(scalar: float, shape: tuple, dtype) -> DiskArray:
    """Write `scalar` broadcast to `shape` directly to a new memmap file.
    Never allocates the full array in RAM - np.memmap assignment from a
    scalar streams to disk block by block."""
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    mm = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    mm.fill(scalar)  # scalar fill, no temp array allocated
    mm.flush()
    return DiskArray(path, shape, dtype)


class Sum(BlockParallelOp):
    """sum(A) - reduce all elements to a single scalar."""

    def __init__(self, input_block_indices: list[tuple]):
        # Every input block contributes to the single scalar output, the
        # op needs to know how many blocks the input has to enumerate them.
        self.input_block_indices = input_block_indices

    def forward_block(self, a_block):
        return jnp.sum(a_block)

    def index_map(self, out_idx):
        # Single output (out_idx == ()) - every input block feeds it.
        return [(idx,) for idx in self.input_block_indices]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block):
        # d(sum)/d(a_i) = 1, so cotangent of a = broadcast(d_out, a.shape).
        return (jnp.full(a_block.shape, d_out_block),)

    def output_shape(self, a_shape):
        return ()


def sum(a):
    """Sum all elements of a. Returns a shape-() DiskArray."""
    policy = get_default_policy()
    input_block_indices = tuple(a.block_grid())

    cache_key = (
        input_block_indices,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _SUM_OP_CACHE.get(cache_key)
    if cached is None:
        op = Sum(list(input_block_indices))
        jax_op = make_jax_op(op, policy, output_page_shape=())
        _SUM_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a)


# HiJax Version.


class HiSum(VJPHiPrimitive):
    def __init__(self, x_aval: DiskArrayType):
        self.in_avals = (x_aval,)
        self.out_aval = ShapedArray((), x_aval.dtype)
        self.params = {}
        super().__init__()

    def expand(self, x):
        # Bridge to jask.sum via DiskArray, materialize shape-() result.
        from .materialize import materialize
        return materialize(sum(x._to_blocked()))

    def vjp_fwd(self, nzs_in, x):
        return self(x), x

    def vjp_bwd_retval(self, res, g):
        # d(sum(x))/dx = broadcast(g, x.shape) - streamed directly to disk,
        # never allocates full-shape array in RAM.
        return (_broadcast_scalar_to_disk(float(g), res.shape, res.dtype),)


def hi_sum(x: DiskArray) -> jax.Array:
    """Disk-backed sum: reduce all elements to a scalar jax.Array."""
    op = HiSum(DiskArrayType(x.shape, x.dtype))
    return op(x)
