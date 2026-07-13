"""Disk-backed array types.

Public API:
- `DiskArray`: the user-facing type. Registered as a hijax `HiType`, so
  `jax.tree_util.tree_map` treats it as an atomic leaf and `optax` works
  transparently. `jax.grad` returns real `DiskArray` gradients (no
  placeholder handles).

Internal:
- `BlockedArray`: block-addressable version used by the existing
  `OOCAlgorithm` block loops (which power every op internally).
  `DiskArray._to_blocked()` bridges from the public type to this one.
- `SpillFile`: ephemeral `BlockedArray` for gradient buffers etc.
"""

import tempfile
import os
import mmap
import itertools
import weakref
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from jax.experimental.hijax import HiType, VJPHiPrimitive, register_hitype, ShapedArray

from .base_page import IOCost


def _safe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


# path -> its active weakref.finalize, so update_() can cancel one when a
# file's identity outlives the single object that first owned it (see
# _adopt_as_persistent below).
_active_finalizers: dict = {}


def _own_fresh_file(disk_array: "DiskArray") -> "DiskArray":
    """Call only at genuine new-file construction sites (from_numpy, an
    op's fresh output, vspace_zero) - never on a re-wrap of an existing
    file (raise_val, deterministic .grad paths, update_'s stable slot).
    Deletes the file once THIS instance is unreachable, not any later
    wrapper pointing at the same filename.
    """
    path = disk_array.filename
    _active_finalizers[path] = weakref.finalize(disk_array, _safe_remove, path)
    return disk_array


def _adopt_as_persistent(path: str) -> None:
    """Cancel path's finalizer, if any - its identity now persists across
    reassignment (e.g. `a = a.update_(...)`), so it must not be deleted
    just because the one Python object that first created it got GC'd.
    """
    finalizer = _active_finalizers.pop(path, None)
    if finalizer is not None:
        finalizer.detach()


#  public: DiskArray + HiType


@dataclass
class DiskArray:
    """A disk-backed array that behaves as a native JAX value.

    `DiskArray` represents an array whose data lives entirely on disk
    and is streamed through JAX one tile at a time - never fully loaded
    into memory - while still composing with `jax.grad`, `jax.jit`, and
    `optax` exactly like an ordinary `jax.Array` would.

    Parameters
    ----------
    filename : str
        Path to the memmap file backing this array's data.
    shape : tuple of int
        The array's shape.
    dtype : numpy.dtype
        The array's dtype.

    Notes
    -----
    `_lo_tracer` is an internal field, set only transiently by
    `DiskArrayType.raise_val` or a primitive's own `expand`, and is
    always a trivial marker (never real array data - see
    `DiskArrayType` for the zero-materialization mechanism). Real data
    always lives at `filename`. Not part of equality or `repr`.

    Examples
    --------
    >>> import numpy as np
    >>> import jask
    >>> jask.set_memory_budget("1GB")
    >>> a = jask.DiskArray.from_numpy(np.ones((4, 4), dtype=np.float32))
    >>> a.shape
    (4, 4)
    """

    filename: str
    shape: tuple
    dtype: np.dtype
    _lo_tracer: object = field(default=None, compare=False, repr=False)

    def to_memmap(self) -> np.memmap:
        """Return a read/write `numpy.memmap` view of this array's file.

        Returns
        -------
        numpy.memmap
            A memmap of shape `self.shape` and dtype `self.dtype`. Reads
            and writes go straight to the backing file; no data is
            copied into memory until a specific region is accessed.
        """
        return np.memmap(self.filename, dtype=self.dtype, mode="r+", shape=self.shape)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "DiskArray":
        """Write a numpy array to a temp file and wrap it as a DiskArray.

        Intended for quick experiments with arrays small enough to
        build in memory first. For genuinely large data, construct
        `DiskArray` directly with an existing file's path instead.

        Parameters
        ----------
        arr : numpy.ndarray
            The array to write to disk.

        Returns
        -------
        DiskArray
            A new disk-backed array holding a copy of `arr`'s data.

        Examples
        --------
        >>> import numpy as np
        >>> import jask
        >>> a = jask.DiskArray.from_numpy(np.arange(4, dtype=np.float32))
        >>> np.asarray(a.to_memmap())
        array([0., 1., 2., 3.], dtype=float32)
        """
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        return _own_fresh_file(cls(path, arr.shape, arr.dtype))

    def _to_blocked(self, page_shape: tuple) -> "BlockedArray":
        """Bridge to BlockedArray so ops can reuse the existing OOCAlgorithm
        block-loop machinery. page_shape must be supplied by the caller
        (derived once per op call, from that op's own num_inputs/phase) -
        not derived independently here, so every array involved in one op
        call is guaranteed to share the same page_shape."""
        return BlockedArray(self.filename, self.shape, self.dtype, page_shape)

    def __add__(self, other):
        from ..linalg import add as _add

        return _add(self, other)

    def __sub__(self, other):
        from ..linalg import sub as _sub

        return _sub(self, other)

    def __mul__(self, other):
        from ..linalg import mul as _mul

        return _mul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        from ..linalg import dot as _dot

        return _dot(self, other)

    def update_(self, new_value: "DiskArray") -> "DiskArray":
        """Overwrite this array's file in place with `new_value`'s data.

        Copies `new_value`'s data into this array's own backing file,
        tile by tile (never a full in-memory copy), and returns a
        `DiskArray` with the same filename/identity as `self`.

        This is the mechanism for loop-carried state (parameters,
        optimizer buffers) under `jax.jit`. Reassigning
        ``a = a + updates`` gives `a` a fresh filename every call, which
        makes `jax.jit` retrace on every single step (a new filename is
        a new type, so it's a cache miss). ``a = a.update_(a + updates)``
        keeps `a`'s filename identical across calls, so `jax.jit`
        compiles once and reuses that executable for the rest of the
        loop.

        Parameters
        ----------
        new_value : DiskArray
            The array whose data should replace this array's contents.
            Must have the same shape and dtype as `self`.

        Returns
        -------
        DiskArray
            A `DiskArray` with the same filename as `self`, now holding
            `new_value`'s data.

        Examples
        --------
        >>> import numpy as np
        >>> import jask
        >>> jask.set_memory_budget("1GB")
        >>> a = jask.DiskArray.from_numpy(np.zeros((4, 4), dtype=np.float32))
        >>> original_path = a.filename
        >>> new_value = jask.DiskArray.from_numpy(np.ones((4, 4), dtype=np.float32))
        >>> a = a.update_(new_value)
        >>> a.filename == original_path
        True
        >>> np.asarray(a.to_memmap())[0, 0]
        1.0
        """
        return _update_op(self, new_value)


@dataclass(frozen=True)
class DiskArrayType(HiType):
    """Tells JAX the shape/dtype/filename of a DiskArray and how to
    lower/raise it.

    Under jax.jit, the value flowing through XLA's traced graph is a
    trivial marker (`lo_ty` is always a scalar, regardless of the real
    shape), not the real array. All real computation happens inside
    jask's own tiled OOCAlgorithm block loops, run as an io_callback side
    effect that writes straight to disk; the callback's return value is
    just a dummy scalar for data-dependency ordering. `filename` has to be
    part of the type (not just a value attribute) because `raise_val` only
    ever receives that trivial marker, so it needs `self.filename` to
    reconstruct the right DiskArray.

    `raise_val` runs before JAX enters the trace context that owns the
    incoming lo-level tracer, so it must never touch that tracer's value -
    only bookkeeping is safe here.
    """

    shape: tuple
    dtype: np.dtype
    filename: str

    def lo_ty(self):
        return [ShapedArray((), self.dtype)]

    def lower_val(self, val: DiskArray):
        if val._lo_tracer is not None:
            return [val._lo_tracer]
        return [jnp.zeros((), dtype=self.dtype)]

    def raise_val(self, marker):
        return DiskArray(self.filename, self.shape, self.dtype, _lo_tracer=marker)

    def to_tangent_aval(self):
        # A DISTINCT but DETERMINISTIC (stable across repeat calls) location
        # from the primal - allocating a fresh path here would violate
        # jit's "same type => same compiled trace" requirement, since the
        # cotangent's type must exactly match this declared tangent aval
        # (filename included) on every call. Must also be IDEMPOTENT - JAX
        # expects to_tangent_aval(to_tangent_aval(t)) == to_tangent_aval(t).
        if self.filename.endswith(".grad"):
            return self
        return DiskArrayType(self.shape, self.dtype, self.filename + ".grad")

    def vspace_zero(self):
        # Used for an unused/zero-contribution cotangent branch within a
        # trace, not fed back as an external input across calls - a fresh
        # path is fine here (no retracing concern).
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        marker = jnp.zeros((), dtype=self.dtype)
        return _own_fresh_file(
            DiskArray(path, self.shape, self.dtype, _lo_tracer=marker)
        )

    def vspace_add(self, x, y):
        # Must compose through a real hi-primitive (jask's own `add`), not
        # raw python/file arithmetic - `x`/`y` may still be abstract
        # hi-tracers here (no concrete filename/data available yet).
        from ..linalg import add as _add

        return _add(x, y)


register_hitype(DiskArray, lambda v: DiskArrayType(v.shape, v.dtype, v.filename))


def _is_tracing(*vals):
    return any(isinstance(v, jax.core.Tracer) for v in vals)


def _as_lo(x):
    """Resolve a DiskArray (or an as-yet-abstract hi-tracer of one) to the
    TRIVIAL lo-level marker used purely for data-dependency ordering inside
    io_callback - never real array data (real data always lives at a
    statically-known `.filename`, read/written directly inside the
    callback). Already-lo-level (non-DiskArray) values pass through as-is.
    """
    if not isinstance(x, DiskArray):
        return x
    return x._lo_tracer if x._lo_tracer is not None else jnp.zeros((), x.dtype)


def _ensure_on_disk(x: DiskArray, page_shape: tuple) -> "BlockedArray":
    """Bridge a DiskArray to a BlockedArray for the bare-eager fast path
    (no active JAX trace anywhere), which skips io_callback entirely.

    `_lo_tracer` is always just a trivial marker, never real data - by the
    time any Python code outside a jit call holds a DiskArray, its file is
    already correct (jax.jit blocks until every io_callback, including the
    one that wrote it, has completed). So this is a thin alias for
    `_to_blocked()`. page_shape must be supplied by the caller (see
    `_to_blocked`'s docstring).
    """
    return x._to_blocked(page_shape)


#  internal: BlockedArray + SpillFile


@dataclass(frozen=True)
class BlockedArray:
    """Block-addressable memmap array used by OOCAlgorithm's block loops.

    Not public - users interact with `DiskArray` above. `_to_blocked()`
    on a `DiskArray` produces one of these on demand.
    """

    filename: str
    full_shape: tuple
    dtype: np.dtype
    page_shape: tuple
    # See disk_array.py history - _marker keeps custom_vjp autodiff happy
    # on the bridge ops. Not needed for the hijax public path.
    _marker: jax.Array = field(default_factory=lambda: jnp.zeros(()), compare=False)

    @classmethod
    def create(cls, filename, full_shape, dtype, page_shape) -> "BlockedArray":
        np.memmap(filename, dtype=dtype, mode="w+", shape=full_shape)
        return cls(filename, full_shape, dtype, page_shape)

    def _mmap(self, mode="r"):
        cached = self.__dict__.get("_mmap_obj")
        if cached is None:
            cached = np.memmap(
                self.filename, dtype=self.dtype, mode="r+", shape=self.full_shape
            )
            try:
                cached._mmap.madvise(mmap.MADV_HUGEPAGE)
            except (AttributeError, OSError, ValueError):
                pass
            object.__setattr__(self, "_mmap_obj", cached)
        return cached

    def to_jax(self) -> jax.Array:
        return jax.device_put(np.asarray(self._mmap(mode="r")))

    def block_grid(self):
        return itertools.product(
            *(range(-(-s // p)) for s, p in zip(self.full_shape, self.page_shape))
        )

    def _slice_for(self, block_idx: tuple) -> tuple[slice, ...]:
        return tuple(
            slice(i * p, min((i + 1) * p, s))
            for i, p, s in zip(block_idx, self.page_shape, self.full_shape)
        )

    def read_block(self, block_idx: tuple, io_cost: IOCost | None = None) -> np.ndarray:
        arr = self._mmap(mode="r")
        block = np.array(arr[self._slice_for(block_idx)])  # copy out before releasing
        self._release_pages(arr)
        if io_cost is not None:
            io_cost.total_pages += 1
        return block

    def write_block(
        self, block_idx: tuple, value: np.ndarray, io_cost: IOCost | None = None
    ):
        arr = self._mmap(mode="r+")
        arr[self._slice_for(block_idx)] = value
        arr.flush()  # must persist dirty pages before advising them away
        self._release_pages(arr)
        if io_cost is not None:
            io_cost.total_pages += 1

    def _release_pages(self, arr):
        """Proactively tell the OS this mmap's pages can be dropped now,
        instead of waiting on passive reclaim under memory pressure - the
        data is safely on disk (read_block already copied it out; write_block
        just flushed it), so this never discards anything, only releases
        pages the kernel would otherwise keep resident until it needs the
        RAM for something else. Whole-mmap rather than per-block-byte-range:
        page_shape often doesn't align to contiguous file bytes for N-D
        blocks, and only one block is ever "current" at a time anyway
        (_ReusingBlockReader caches at most one JAX-side copy, not a view
        into this mmap), so there is nothing else in this mapping that
        still needs to stay resident.
        """
        try:
            arr._mmap.madvise(mmap.MADV_DONTNEED)
        except (AttributeError, OSError, ValueError):
            pass


jax.tree_util.register_dataclass(
    BlockedArray,
    data_fields=["_marker"],
    meta_fields=["filename", "full_shape", "dtype", "page_shape"],
)


def _tiled_copy(src_path, dst_path, shape, dtype, page_shape):
    """Copy src -> dst one page at a time - never a full-array read/write."""
    src = BlockedArray(src_path, shape, dtype, page_shape)
    dst = BlockedArray.create(dst_path, shape, dtype, page_shape)
    if shape == ():
        dst.write_block((), np.asarray(src.read_block(())))
        return
    for idx in dst.block_grid():
        dst.write_block(idx, np.asarray(src.read_block(idx)))


class HiUpdate(VJPHiPrimitive):
    """DiskArray.update_(new_value): overwrite self's own file in place,
    tiled, returning a DiskArray with the SAME filename/identity - the
    mechanism that lets a jax.jit-compiled loop reuse one executable
    instead of retracing every step (see DiskArray.update_'s docstring)."""

    def __init__(self, self_ty: DiskArrayType, new_ty: DiskArrayType):
        self.in_avals = (self_ty, new_ty)
        self.out_aval = self_ty
        self.params = {}
        self._self_filename = self_ty.filename
        self._new_filename = new_ty.filename
        self._shape, self._dtype = self_ty.shape, self_ty.dtype
        # self_filename's identity now persists across reassignment
        # (`a = a.update_(...)`) - the object that first created it (e.g.
        # from_numpy) is about to become unreachable and get GC'd, but the
        # file must outlive it.
        _adopt_as_persistent(self_ty.filename)
        super().__init__()

    def expand(self, self_val, new_val):
        from .base_page import get_default_policy, derive_page_shape

        self_filename, new_filename = self._self_filename, self._new_filename
        shape, dtype = self._shape, self._dtype
        # A straight tiled copy (read one block, write one block) - no
        # accumulator/combine step, so this is even lighter than a normal
        # single-input forward pass; num_inputs=1 forward is a safe,
        # slightly conservative fit.
        page_shape = derive_page_shape(
            get_default_policy(), dtype, shape, num_inputs=1, phase="forward"
        )

        def run(m1, m2):
            _tiled_copy(new_filename, self_filename, shape, dtype, page_shape)
            return np.float32(0.0)

        marker = io_callback(
            run, jax.ShapeDtypeStruct((), dtype), _as_lo(self_val), _as_lo(new_val)
        )
        return DiskArray(self_filename, shape, dtype, _lo_tracer=marker)


def _update_op(self_arr: DiskArray, new_arr: DiskArray) -> DiskArray:
    op = HiUpdate(
        DiskArrayType(self_arr.shape, self_arr.dtype, self_arr.filename),
        DiskArrayType(new_arr.shape, new_arr.dtype, new_arr.filename),
    )
    return op(self_arr, new_arr)


@dataclass(frozen=True)
class SpillFile(BlockedArray):
    """BlockedArray backed by a fresh temp file - used for gradient buffers."""

    @classmethod
    def create(cls, full_shape, dtype, page_shape) -> "SpillFile":
        fd, path = tempfile.mkstemp(suffix=".spill")
        os.close(fd)
        np.memmap(path, dtype=dtype, mode="w+", shape=full_shape)
        return cls(path, full_shape, dtype, page_shape)


jax.tree_util.register_dataclass(
    SpillFile,
    data_fields=["_marker"],
    meta_fields=["filename", "full_shape", "dtype", "page_shape"],
)
