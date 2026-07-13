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
from jax.experimental.hijax import VJPHiPrimitive
from jax.core import ShapedArray


class OOCAlgorithm:
    """Takes in Op implementation and implements complete forward,
    backward and gradient pass.

    BlockParallelOp will be used to write a complete loop,
    CustomOp will be called as it is without much change (it must define
    its own `forward(algo, *inputs)` / `backward(algo, inputs, d_out)`
    methods, using `algo` as a toolkit for block I/O and policy/cost
    tracking).
    """

    def __init__(self, op: Union[BlockParallelOp, CustomOp]):
        """Store the op and JIT-compile its block functions once."""
        self._op = op
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
        """Load one output tile's contributing blocks and fold them into a
        result - one k at a time, never pre-loading the whole K-sequence
        (an op like Sum has K equal to every block in the input, so batching
        the reads would defeat the memory budget entirely)."""
        block_idx_groups = list(self._op.index_map(out_idx))  # K entries

        acc = None
        for idxs in block_idx_groups:
            blocks_k = [
                reader.read_one(pos, arr, idxs[pos]) for pos, arr in enumerate(inputs)
            ]
            partial = self._jit_forward_block(*blocks_k)
            acc = partial if acc is None else self._op.combine(acc, partial)
            # Force this iteration's computation to actually complete before
            # moving on - JAX's dispatch is async by default, so without
            # this, buffers from many iterations can queue up unexecuted
            # (and unfreed) instead of being bounded to the current block.
            jax.block_until_ready(acc)

        return acc

    def run_backward(
        self,
        inputs: list[BlockedArray],
        d_out: BlockedArray,
        target_paths: list[str] | None = None,
    ) -> tuple:
        """Compute and accumulate gradients for every input, one block at a
        time. When target_paths is given (the BlockParallelOp path), each
        gradient is written directly there - no separate scratch SpillFile
        plus a full-array copy afterward. CustomOp keeps its existing
        contract (returns wherever it put the result) since no CustomOp
        currently exists to need target_paths support.
        """
        if isinstance(self._op, CustomOp):
            return self._op.backward(self, inputs, d_out)

        grads = self._allocate_grads(inputs, target_paths)
        reader = _ReusingBlockReader(len(inputs), self._io_tracker)

        for out_idx in d_out.block_grid():
            self._accumulate_grad_block(inputs, grads, d_out, out_idx, reader)

        return tuple(grads)

    def _allocate_grads(
        self, inputs: list[BlockedArray], target_paths: list[str] | None
    ) -> list[BlockedArray]:
        """Create each gradient's BlockedArray - directly at its
        deterministic target path when given, else a fresh SpillFile."""
        if target_paths is not None:
            return [
                BlockedArray.create(path, a.full_shape, a.dtype, a.page_shape)
                for a, path in zip(inputs, target_paths)
            ]
        return [SpillFile.create(a.full_shape, a.dtype, a.page_shape) for a in inputs]

    def _accumulate_grad_block(
        self,
        inputs: list[BlockedArray],
        grads: list[SpillFile],
        d_out: BlockedArray,
        out_idx: tuple,
        reader: _ReusingBlockReader,
    ):
        """Compute one output tile's input gradients and accumulate them into
        grads - one k at a time, same reasoning as _compute_output_block."""
        d_out_block = jax.device_put(d_out.read_block(out_idx, self._io_tracker))
        block_idx_groups = list(self._op.index_map(out_idx))  # K entries

        for idxs in block_idx_groups:
            blocks_k = [
                reader.read_one(pos, arr, idxs[pos]) for pos, arr in enumerate(inputs)
            ]
            grad_blocks = self._jit_backward_block(d_out_block, *blocks_k)
            jax.block_until_ready(grad_blocks)
            # Scatter-writes to K different grad-array locations - read-
            # modify-write, since a block may receive contributions from
            # multiple output tiles (e.g. matmul's dB_kj summed over i).
            for grad_arr, grad_block, idx in zip(grads, grad_blocks, idxs):
                existing = grad_arr.read_block(idx, self._io_tracker)
                grad_arr.write_block(
                    idx, existing + np.asarray(grad_block), self._io_tracker
                )


# make_op: auto-generate a public op from a BlockParallelOp class


def make_op(op_class, doc: str | None = None):
    """Given a BlockParallelOp subclass, return the public function to call it.
    Handles hijax registration, forward/backward wiring, DiskArray bridging.

    `doc`, if given, becomes the returned function's `__doc__` - the public
    op modules pass a real NumPy-style docstring here, since `public_fn`
    itself is a generic wrapper with nothing op-specific to say.
    Users write only the block-level math; this function does the rest.

    Under jax.jit, values flowing through XLA's traced graph are a TRIVIAL
    marker (see DiskArrayType.lo_ty), not the real array - real data only
    ever gets touched by OOCAlgorithm's tiled loop, run as an io_callback
    side effect that writes straight to disk. A scalar op output (e.g. Sum)
    is the one exception: tiny enough to flow as real data like any other
    jax scalar.

    HiPrimitive method bodies run once, abstractly, while JAX is still
    building the jaxpr, so they can't touch concrete data directly - that's
    why forward/backward computation always happens inside io_callback, and
    why backward is its own primitive (HiOpBackward) that vjp_bwd_retval
    only binds rather than executing inline.

    A bare eager call (no jit, no grad) skips io_callback and stays lazy.
    """
    from .base_page import get_default_policy, derive_page_shape
    from .disk_array import (
        DiskArray,
        DiskArrayType,
        _as_lo,
        _ensure_on_disk,
        _is_tracing,
        _own_fresh_file,
    )

    def _input_blocked(filename, shape, dtype, page_shape):
        """Block-addressable view of an input's existing file - never
        writes, since the file already holds the correct data (the
        DiskArray's lo_val is only ever a trivial marker, never real data).
        page_shape must be derived by the caller using this op call's own
        num_inputs/phase, not re-derived independently here."""
        return BlockedArray(filename, shape, dtype, page_shape)

    def _write_scalar(path, dtype, lo_val):
        """Write a scalar cotangent value to its own file."""
        mm = np.memmap(path, dtype=dtype, mode="w+", shape=())
        mm[()] = np.asarray(lo_val)
        mm.flush()
        return BlockedArray(path, (), dtype, ())

    class HiOpBackward(VJPHiPrimitive):
        """Backward computation for one HiOpGenerated call, as its own
        hi-primitive - see make_op's docstring.

        Each output's file is `<input's own filename>.grad` - deterministic
        (matches DiskArrayType.to_tangent_aval), not a fresh path per call,
        so a jit-compiled loop feeding gradients back in via
        DiskArray.update_ reuses one compiled executable.
        """

        def __init__(self, x_avals, g_ty, op_kwargs, g_is_scalar):
            self.in_avals = (*x_avals, g_ty)
            self.out_aval = tuple(ty.to_tangent_aval() for ty in x_avals)
            self._x_avals = x_avals
            self._g_ty = g_ty
            self._g_is_scalar = g_is_scalar
            self._op_kwargs = op_kwargs
            self.params = {}
            super().__init__()

        def expand(self, *args):
            *xs, g = args
            x_avals = self._x_avals
            dtype = x_avals[0].dtype
            op_kwargs = self._op_kwargs
            g_is_scalar = self._g_is_scalar
            grad_paths = [ty.to_tangent_aval().filename for ty in x_avals]
            num_inputs = len(xs)

            if not _is_tracing(*[_as_lo(x) for x in xs], _as_lo(g)):
                # Bare eager call - stay fully lazy, no io_callback needed.
                policy = get_default_policy()
                blockeds = [
                    _ensure_on_disk(
                        x, derive_page_shape(policy, dtype, ty.shape, num_inputs, "backward")
                    )
                    for x, ty in zip(xs, x_avals)
                ]
                op_impl = op_class.from_inputs(*blockeds, **op_kwargs)
                if g_is_scalar:
                    fd, path = tempfile.mkstemp(suffix=".dat")
                    os.close(fd)
                    mm = np.memmap(path, dtype=dtype, mode="w+", shape=())
                    mm[()] = float(g)
                    mm.flush()
                    d_out_ba = BlockedArray(path, (), dtype, ())
                else:
                    g_page = derive_page_shape(
                        policy, dtype, self._g_ty.shape, num_inputs, "backward"
                    )
                    d_out_ba = _ensure_on_disk(g, g_page)
                algo = OOCAlgorithm(op_impl)
                grads_ba = algo.run_backward(blockeds, d_out_ba, grad_paths)
                return tuple(
                    DiskArray(path, x.shape, x.dtype) for x, path in zip(xs, grad_paths)
                )

            filenames = [x.filename for x in xs]
            shapes = [ty.shape for ty in x_avals]

            def run(*los):
                # g's marker is always passed, even when its value is
                # unused (non-scalar case, real data read from its
                # filename instead) - needed so JAX sees the data
                # dependency and runs this after whatever wrote g's file.
                *xlos, g_lo = los
                policy = get_default_policy()
                blockeds = [
                    _input_blocked(
                        f, s, dtype, derive_page_shape(policy, dtype, s, num_inputs, "backward")
                    )
                    for f, s in zip(filenames, shapes)
                ]
                op_impl = op_class.from_inputs(*blockeds, **op_kwargs)
                out_shape = op_impl.output_shape(*(b.full_shape for b in blockeds))
                if g_is_scalar:
                    fd, g_path = tempfile.mkstemp(suffix=".dat")
                    os.close(fd)
                    d_out_ba = _write_scalar(g_path, dtype, g_lo)
                else:
                    # g's real data is already on disk (written when it was
                    # produced) - just reference it, no write needed.
                    g_page = derive_page_shape(policy, dtype, out_shape, num_inputs, "backward")
                    d_out_ba = BlockedArray(
                        self._g_ty.filename, out_shape, dtype, g_page
                    )
                algo = OOCAlgorithm(op_impl)
                algo.run_backward(blockeds, d_out_ba, grad_paths)
                return np.float32(0.0)

            io_args = [_as_lo(x) for x in xs] + [_as_lo(g)]
            marker = io_callback(run, jax.ShapeDtypeStruct((), dtype), *io_args)

            return tuple(
                DiskArray(path, ty.shape, ty.dtype, _lo_tracer=marker)
                for ty, path in zip(x_avals, grad_paths)
            )

        def vjp_fwd(self, nzs_in, *args):
            raise NotImplementedError(
                "second-order gradients are not supported for jask ops"
            )

    class HiOpGenerated(VJPHiPrimitive):
        def __init__(self, *input_avals, **op_kwargs):
            """Store inputs and derive the output shape/dtype.
            Extra kwargs (like axis=, axes=) get forwarded to op.from_inputs."""
            self._op_kwargs = op_kwargs
            self.in_avals = tuple(input_avals)
            input_shapes = tuple(a.shape for a in input_avals)
            dummy_shapes = [DummyBlocked(a.shape) for a in input_avals]
            temp_op = op_class.from_inputs(*dummy_shapes, **op_kwargs)
            out_shape = temp_op.output_shape(*input_shapes)
            dtype = input_avals[0].dtype
            self._dtype = dtype
            self._out_shape = out_shape
            if out_shape == ():
                self.out_aval = ShapedArray((), dtype)
                self._output_is_scalar = True
                self._out_path = None
            else:
                fd, out_path = tempfile.mkstemp(suffix=".dat")
                os.close(fd)
                self._out_path = out_path
                self.out_aval = DiskArrayType(out_shape, dtype, out_path)
                self._output_is_scalar = False
            self.params = {}
            super().__init__()

        def expand(self, *xs):
            los = [_as_lo(x) for x in xs]
            dtype = self._dtype
            op_kwargs = self._op_kwargs
            out_shape = self._out_shape
            output_is_scalar = self._output_is_scalar
            out_path = self._out_path
            num_inputs = len(xs)

            if not _is_tracing(*los):
                # Bare eager call - stay fully lazy, no io_callback needed.
                policy = get_default_policy()
                blockeds = [
                    _ensure_on_disk(
                        x, derive_page_shape(policy, dtype, x.shape, num_inputs, "forward")
                    )
                    for x in xs
                ]
                op_impl = op_class.from_inputs(*blockeds, **op_kwargs)
                out_page_shape = (
                    ()
                    if out_shape == ()
                    else derive_page_shape(policy, dtype, out_shape, num_inputs, "forward")
                )
                algo = OOCAlgorithm(op_impl)
                result_ba = algo.run_forward(
                    blockeds, out_page_shape, output_filename=out_path
                )
                if output_is_scalar:
                    return result_ba.to_jax()
                return _own_fresh_file(DiskArray(out_path, out_shape, dtype))

            filenames = [x.filename for x in xs]
            shapes = [x.shape for x in xs]

            def run(*los):
                policy = get_default_policy()
                blockeds = [
                    _input_blocked(
                        f, s, dtype, derive_page_shape(policy, dtype, s, num_inputs, "forward")
                    )
                    for f, s in zip(filenames, shapes)
                ]
                op_impl = op_class.from_inputs(*blockeds, **op_kwargs)
                out_page_shape = (
                    ()
                    if out_shape == ()
                    else derive_page_shape(policy, dtype, out_shape, num_inputs, "forward")
                )
                algo = OOCAlgorithm(op_impl)
                result_ba = algo.run_forward(
                    blockeds, out_page_shape, output_filename=out_path
                )
                if output_is_scalar:
                    return np.asarray(result_ba.to_jax())
                return np.float32(0.0)  # trivial marker - real data is on disk

            # Both branches declare a scalar result: the real (tiny) value
            # when output_is_scalar (out_shape is already () in that case),
            # or a trivial marker otherwise.
            result = io_callback(run, jax.ShapeDtypeStruct((), dtype), *los)

            if output_is_scalar:
                return result
            # No _own_fresh_file here - this instance is intermediate.
            # raise_val reconstructs another wrapper around out_path when
            # the value crosses back out of the jit boundary; finalizing
            # this one would delete out_path while that later wrapper is
            # still in use, same bug as a naive __del__ on the class.
            return DiskArray(out_path, out_shape, dtype, _lo_tracer=result)

        def vjp_fwd(self, nzs_in, *xs):
            return self(*xs), xs

        def vjp_bwd_retval(self, res, g):
            # Bind a SEPARATE hi-primitive instead of doing any concrete-data
            # work here directly - see the jit-compatibility note on
            # `make_op` above.
            if self._output_is_scalar:
                g_ty = ShapedArray((), self._dtype)
            else:
                g_ty = DiskArrayType(
                    self._out_shape, self._dtype, self._out_path
                ).to_tangent_aval()
            backward_op = HiOpBackward(
                self.in_avals, g_ty, self._op_kwargs, self._output_is_scalar
            )
            return backward_op(*res, g)

    def public_fn(*args, **kwargs):
        """User-facing entry point. kwargs (like axis=, axes=) get forwarded
        to the op's from_inputs classmethod."""
        input_types = tuple(DiskArrayType(a.shape, a.dtype, a.filename) for a in args)
        op = HiOpGenerated(*input_types, **kwargs)
        return op(*args)

    if doc is not None:
        public_fn.__doc__ = doc
    return public_fn


class DummyBlocked:
    """Placeholder shape holder for output_shape() calls at primitive-init time."""

    def __init__(self, shape):
        self.full_shape = shape
        self.shape = shape
