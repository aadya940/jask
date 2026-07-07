"""Metadata and capacity-planning primitives for disk-backed block computation.

An array too large for device (or host) memory is partitioned into a grid of
fixed-size blocks ("pages"), each backed by a slice of an `np.memmap`-mapped
file on disk. Disk-to-host transfer is handled implicitly by the OS's page
cache; host-to-device transfer is explicit and must stay within a fixed
memory budget, since no equivalent paging mechanism exists across that
boundary.

This module defines:
- `DiskArray`: a memmap-backed array partitioned into a grid of blocks.
  Owns block addressing (slicing, edge-block handling) and the actual
  disk reads/writes; callers address blocks purely by `block_idx`, never
  by raw byte offset.
- `IOCost`: a running tally of host-to-device transfers, used to reason
  about and benchmark the cost of a computation.
- `Policy`: derives how many blocks can be held in flight at once from a
  user-specified device memory budget and available host memory, so that
  computation can safely use more than the bare minimum working set
  without ever exceeding either budget.
- `SpillFile`: a `DiskArray` for ephemeral, computation-scoped data (e.g.
  gradients or reduced intermediates written mid-algorithm) rather than a
  user-supplied input/output array. Owns its own temp file and deletes it
  on cleanup instead of persisting past the run.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

import psutil
import itertools
import tempfile
import os
import mmap


@dataclass
class IOCost:
    """I/O Cost of a particular algorithm."""

    total_pages: int


OS_PAGE_SIZE = 4096  # standard OS memory page, bytes
DASK_DEFAULT_PAGE_SIZE = 128 * 1024 * 1024  # Dask's array.chunk-size default, bytes


def align_to_os_page(size: int) -> int:
    """Round a byte size down to the nearest multiple of OS_PAGE_SIZE.

    Block boundaries should land on OS page boundaries so disk reads don't
    span a partial extra page. Has no effect on sizes already aligned (e.g.
    DASK_DEFAULT_PAGE_SIZE, which is a multiple of OS_PAGE_SIZE already).
    """
    return size - (size % OS_PAGE_SIZE)


@dataclass
class Policy:
    max_memory: int
    pages_per_group: int
    page_size: int = align_to_os_page(DASK_DEFAULT_PAGE_SIZE)
    resident_pages: int = 0

    @property
    def device_capacity(self) -> int:
        return self.max_memory // (self.page_size * self.pages_per_group)

    @property
    def host_capacity(self) -> int:
        available = psutil.virtual_memory().available
        return available // (self.page_size * self.pages_per_group)

    @property
    def working_capacity(self) -> int:
        return min(self.device_capacity, self.host_capacity)

    @property
    def prefetch_depth(self) -> int:
        return max(1, self.working_capacity)

    @property
    def is_full(self) -> bool:
        return self.resident_pages >= self.device_capacity

    def mark_loaded(self):
        self.resident_pages += 1

    def mark_evicted(self):
        self.resident_pages = max(0, self.resident_pages - 1)


_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def _parse_memory(value: int | str) -> int:
    """Parse a memory size given as bytes (int) or a string like "4GB"."""
    if isinstance(value, int):
        return value
    value = value.strip().upper()
    for suffix, mult in sorted(_UNITS.items(), key=lambda kv: -len(kv[0])):
        if value.endswith(suffix):
            return int(float(value[: -len(suffix)]) * mult)
    return int(value)  # bare number string, assume bytes


_default_policy: Policy | None = None


def set_memory_budget(max_memory: int | str, pages_per_group: int = 3):
    """Set the process-wide default memory budget used by ops that don't
    receive an explicit Policy (e.g. jask.dot(a, b)). Call this once,
    not per-op-call.
    """
    global _default_policy
    _default_policy = Policy(
        max_memory=_parse_memory(max_memory), pages_per_group=pages_per_group
    )


def get_default_policy() -> Policy:
    if _default_policy is None:
        raise RuntimeError(
            'No memory budget set. Call jask.set_memory_budget("4GB") first.'
        )
    return _default_policy


def derive_page_shape(policy: Policy, dtype: np.dtype, full_shape: tuple) -> tuple:
    """Pick a block shape whose byte size fits policy.page_size, distributing
    it evenly across full_shape's dimensions (an N-th root split), clipped
    so no dimension's block exceeds the array's own extent in that dim.
    """
    itemsize = np.dtype(dtype).itemsize
    elements_per_page = max(1, policy.page_size // itemsize)
    ndim = len(full_shape)
    side = max(1, int(round(elements_per_page ** (1 / ndim))))
    return tuple(min(side, s) for s in full_shape)


@dataclass(frozen=True)
class DiskArray:
    filename: str
    full_shape: tuple
    dtype: np.dtype
    page_shape: tuple
    # One tiny, real pytree leaf (content unused). Without at least one real
    # leaf somewhere in a custom_vjp call's arguments, JAX's autodiff treats
    # the whole call as having nothing to differentiate and silently skips
    # invoking the backward rule entirely — confirmed empirically; every
    # DiskArray needs this so gradients are actually computed regardless of
    # which/how many arguments in a given op call are disk-backed.
    _marker: jax.Array = field(default_factory=lambda: jnp.zeros(()), compare=False)

    @classmethod
    def create(cls, filename, full_shape, dtype, page_shape) -> "DiskArray":
        np.memmap(filename, dtype=dtype, mode="w+", shape=full_shape)
        return cls(filename, full_shape, dtype, page_shape)

    def _mmap(self, mode="r"):
        # Cached per-instance, not globally — a permanent global dict keyed
        # by filename leaked memory badly (every file ever touched stayed
        # fully resident for the rest of the process, causing real OOMs
        # across multi-trial benchmarks). Within one algorithm call, the
        # same DiskArray Python objects (inputs/out/grads) are reused for
        # the whole loop, so this still gives the caching benefit — and it's
        # freed automatically via garbage collection once the call finishes
        # and local references drop, no manual eviction needed anywhere.
        # object.__setattr__ bypasses frozen=True for this internal cache.
        cached = self.__dict__.get("_mmap_obj")
        if cached is None:
            cached = np.memmap(
                self.filename, dtype=self.dtype, mode="r+", shape=self.full_shape
            )
            try:
                cached._mmap.madvise(mmap.MADV_HUGEPAGE)
            except (AttributeError, OSError, ValueError):
                pass  # not supported on this platform/kernel — non-fatal
            object.__setattr__(self, "_mmap_obj", cached)
        return cached

    def to_jax(self) -> jax.Array:
        """Explicitly materialize the whole array into device memory.

        Only call this when the array is known to be small enough to fit —
        it fully loads full_shape into RAM/device memory, defeating the
        out-of-core guarantee if used on something too big. No automatic
        or implicit path calls this; it exists as a deliberate escape hatch.
        """
        return jax.device_put(np.asarray(self._mmap(mode="r")))

    def block_grid(self):
        return itertools.product(
            *(
                range(-(-s // p)) for s, p in zip(self.full_shape, self.page_shape)
            )  # ceil div
        )

    def _slice_for(self, block_idx: tuple) -> tuple[slice, ...]:
        return tuple(
            slice(i * p, min((i + 1) * p, s))
            for i, p, s in zip(block_idx, self.page_shape, self.full_shape)
        )

    def read_block(self, block_idx: tuple, io_cost: IOCost | None = None) -> np.ndarray:
        arr = self._mmap(mode="r")
        block = arr[self._slice_for(block_idx)]
        if io_cost is not None:
            io_cost.total_pages += 1
        return block

    def write_block(
        self, block_idx: tuple, value: np.ndarray, io_cost: IOCost | None = None
    ):
        arr = self._mmap(mode="r+")
        arr[self._slice_for(block_idx)] = value
        if io_cost is not None:
            io_cost.total_pages += 1

    def is_edge_block(self, block_idx) -> bool:
        return any(
            s.stop - s.start != p
            for s, p in zip(self._slice_for(block_idx), self.page_shape)
        )


jax.tree_util.register_dataclass(
    DiskArray,
    data_fields=["_marker"],
    meta_fields=["filename", "full_shape", "dtype", "page_shape"],
)


@dataclass(frozen=True)
class SpillFile(DiskArray):
    """A DiskArray for ephemeral, computation-scoped data (grad buffers,
    reduced intermediates), same block I/O as DiskArray, but owns its own
    temp file and deletes it on cleanup instead of persisting past the run.
    """

    @classmethod
    def create(cls, full_shape, dtype, page_shape) -> "SpillFile":
        fd, path = tempfile.mkstemp(suffix=".spill")
        os.close(fd)
        np.memmap(path, dtype=dtype, mode="w+", shape=full_shape)
        return cls(path, full_shape, dtype, page_shape)

    def cleanup(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def __enter__(self) -> "SpillFile":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


jax.tree_util.register_dataclass(
    SpillFile,
    data_fields=["_marker"],
    meta_fields=["filename", "full_shape", "dtype", "page_shape"],
)
