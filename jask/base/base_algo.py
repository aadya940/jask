from .base_ops import *
from .base_page import *
from .disk_array import *
from .utils import _ReusingBlockReader

from typing import Union
import os
import tempfile

import numpy as np
import jax
from jax.experimental import io_callback


class OOCAlgorithm:
    """Takes in Op implementation and implements complete forward,
    backward and gradient pass.

    BlockParallelOp will be used to write a complete loop,
    CustomOp will be called as it is without much change (it must define
    its own `forward(algo, *inputs)` / `backward(algo, inputs, d_out)`
    methods, using `algo` as a toolkit for block I/O and policy/cost
    tracking).
    """

    def __init__(self, op: Union[BlockParallelOp, CustomOp], policy: Policy):
        """Store the op/policy and JIT-compile the op's block functions once."""
        self._op = op
        self._policy = policy
        self._io_tracker = IOCost(total_pages=0)

        if isinstance(op, BlockParallelOp):
            # Plain per-call JIT, no vmap batching: profiling showed each
            # forward_block call here does real, substantial compute (single
            # large matmul), so per-call dispatch overhead was never the
            # bottleneck - vmap's batched-matmul shape turned out harder for
            # XLA's CPU backend to parallelize well, plus jnp.stack itself
            # added real cost. This is the faster, simpler path for this
            # block size; vmap batching would only pay off for many small
            # blocks where dispatch overhead genuinely dominates.
            self._jit_forward_block = jax.jit(op.forward_block)
            # Tried donate_argnums=(1, 2) here (d_a/d_b's shapes match
            # a_block/b_block's exactly, a real donation candidate) - but it
            # conflicts with _ReusingBlockReader's caching: a donated block
            # later pulled from the reuse cache into another call crashes
            # with "Buffer has been deleted or donated", confirmed by
            # actually running it. Reverted; not compatible with the reuse
            # optimization as currently structured.
            self._jit_backward_block = jax.jit(op.backward_block)

    def run_forward(
        self,
        inputs: list[BlockedArray],
        output_page_shape: tuple,
        output_filename: str | None = None,
    ) -> BlockedArray:
        """Compute the full disk-backed output, one block at a time."""
        if isinstance(self._op, CustomOp):
            return self._op.forward(self, *inputs)

        out = self._allocate_output(inputs, output_page_shape, output_filename)
        reader = _ReusingBlockReader(len(inputs), self._io_tracker)

        for out_idx in out.block_grid():
            acc = self._compute_output_block(inputs, out_idx, reader)
            out.write_block(out_idx, np.asarray(acc), self._io_tracker)

        return out

    def _allocate_output(
        self,
        inputs: list[BlockedArray],
        output_page_shape: tuple,
        output_filename: str | None,
    ) -> BlockedArray:
        """Create the (empty) output BlockedArray at the op's declared shape."""
        out_shape = self._op.output_shape(*(a.full_shape for a in inputs))
        if output_filename is None:
            fd, output_filename = tempfile.mkstemp(suffix=".ooc")
            os.close(fd)
        return BlockedArray.create(
            output_filename, out_shape, inputs[0].dtype, output_page_shape
        )

    def _compute_output_block(
        self, inputs: list[BlockedArray], out_idx: tuple, reader: _ReusingBlockReader
    ):
        """Load one output tile's contributing blocks and fold them into a result."""
        block_idx_groups = list(self._op.index_map(out_idx))  # K entries

        per_input_blocks = [
            reader.read(pos, arr, tuple(idxs[pos] for idxs in block_idx_groups))
            for pos, arr in enumerate(inputs)
        ]
        self._policy.mark_loaded()

        acc = None
        for k in range(len(block_idx_groups)):
            partial = self._jit_forward_block(
                *(blocks[k] for blocks in per_input_blocks)
            )
            acc = partial if acc is None else self._op.combine(acc, partial)

        self._policy.mark_evicted()
        return acc

    def run_backward(
        self,
        inputs: list[BlockedArray],
        d_out: BlockedArray,
        grad_handles: list[SpillFile] | None = None,
    ) -> tuple:
        """Compute and accumulate gradients for every input, one block at a time."""
        if isinstance(self._op, CustomOp):
            return self._op.backward(self, inputs, d_out)

        grads = self._allocate_grads(inputs, grad_handles)
        reader = _ReusingBlockReader(len(inputs), self._io_tracker)

        for out_idx in d_out.block_grid():
            self._accumulate_grad_block(inputs, grads, d_out, out_idx, reader)

        return tuple(grads)

    def _allocate_grads(
        self, inputs: list[BlockedArray], grad_handles: list[SpillFile] | None
    ) -> list[SpillFile]:
        """Create (or zero-init pre-decided) gradient SpillFiles, one per input."""
        if grad_handles is None:
            return [
                SpillFile.create(a.full_shape, a.dtype, a.page_shape) for a in inputs
            ]
        # Pre-decided filenames (e.g. from make_jax_op, so io_callback's
        # declared and actual return structures match) - allocate them here.
        for g in grad_handles:
            np.memmap(g.filename, dtype=g.dtype, mode="w+", shape=g.full_shape)
        return grad_handles

    def _accumulate_grad_block(
        self,
        inputs: list[BlockedArray],
        grads: list[SpillFile],
        d_out: BlockedArray,
        out_idx: tuple,
        reader: _ReusingBlockReader,
    ):
        """Compute one output tile's input gradients and accumulate them into grads."""
        d_out_block = jax.device_put(d_out.read_block(out_idx, self._io_tracker))
        block_idx_groups = list(self._op.index_map(out_idx))  # K entries

        per_input_blocks = [
            reader.read(pos, arr, tuple(idxs[pos] for idxs in block_idx_groups))
            for pos, arr in enumerate(inputs)
        ]
        self._policy.mark_loaded()

        for k, idxs in enumerate(block_idx_groups):
            grad_blocks = self._jit_backward_block(
                d_out_block, *(blocks[k] for blocks in per_input_blocks)
            )
            # Scatter-writes to K different grad-array locations - read-
            # modify-write, since a block may receive contributions from
            # multiple output tiles (e.g. matmul's dB_kj summed over i).
            for grad_arr, grad_block, idx in zip(grads, grad_blocks, idxs):
                existing = grad_arr.read_block(idx, self._io_tracker)
                grad_arr.write_block(
                    idx, existing + np.asarray(grad_block), self._io_tracker
                )

        self._policy.mark_evicted()


def make_jax_op(
    op: Union[BlockParallelOp, CustomOp], policy: Policy, output_page_shape: tuple
):
    """Wraps an Op + Policy into a JAX-callable, differentiable function.

    The returned function is safe to call inside a user's own jax.jit'd
    training step: forward/backward run as io_callback (JIT-compatible,
    guaranteed to execute on every real call, unlike pure_callback), with
    the gradient rule supplied manually via custom_vjp since JAX cannot
    trace through the disk-spanning block loop itself.
    """
    algo = OOCAlgorithm(op, policy)

    @jax.custom_vjp
    def f(*handles: BlockedArray):
        """Run the forward pass via a guaranteed-execution host callback."""
        out_shape = op.output_shape(*(h.full_shape for h in handles))
        fd, out_filename = tempfile.mkstemp(suffix=".ooc")
        os.close(fd)
        out_handle = BlockedArray(
            out_filename, out_shape, handles[0].dtype, output_page_shape
        )

        return io_callback(
            lambda *hs: algo.run_forward(list(hs), output_page_shape, out_filename),
            out_handle,  # zero-leaf static handle, not array data - see base_page's register_dataclass
            *handles,
        )

    def f_fwd(*handles: BlockedArray):
        """Run forward, saving the input handles as backward's residuals."""
        return f(*handles), handles

    def f_bwd(residuals, d_out):
        """Compute gradients as a side effect, return placeholder cotangents."""
        handles = residuals
        # Real gradient goes to a deterministic path derived from each input's
        # own filename - NOT a filename returned through JAX's cotangent
        # machinery. Confirmed empirically: custom_vjp requires bwd's return
        # to match the primal args' pytree structure *exactly* (including
        # meta fields like filename), so any *new* filename either errors
        # (strict match) or gets silently discarded in favor of the primal's
        # own aux (loosened match) - there is no way to hand back a genuinely
        # different filename through the return value itself. So: write the
        # real gradient as a side effect at `<input filename>.grad`, and
        # return the primal handles themselves as the placeholder cotangent
        # structure (trivially matches, since they *are* that structure).
        grad_handles = tuple(
            BlockedArray(h.filename + ".grad", h.full_shape, h.dtype, h.page_shape)
            for h in handles
        )
        # d_out's filename slot points at the value file (JAX's treedef
        # match constraint); real cotangent data lives at <that>.grad,
        # written by the downstream op's f_bwd. Swap here before reading.
        d_out_grad = SpillFile(
            d_out.filename + ".grad", d_out.full_shape, d_out.dtype, d_out.page_shape
        )
        io_callback(
            lambda *args: algo.run_backward(list(args[:-1]), args[-1], grad_handles),
            grad_handles,
            *handles,
            d_out_grad,
        )
        return handles

    f.defvjp(f_fwd, f_bwd)
    return f


def gradient_of(handle: SpillFile) -> SpillFile:
    """After jax.grad, the returned handle is a placeholder (same identity as
    the primal input) - the real gradient lives at `<filename>.grad`. Use
    this to get a BlockedArray pointing at the actual gradient data.
    """
    return SpillFile(
        handle.filename + ".grad", handle.full_shape, handle.dtype, handle.page_shape
    )
