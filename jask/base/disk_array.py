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
from dataclasses import dataclass, field

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.hijax import HiType, register_hitype, ShapedArray

from .base_page import IOCost

#  public: DiskArray + HiType


@dataclass
class DiskArray:
    """Disk-backed array. Data lives in a memmap file at `filename`."""

    filename: str
    shape: tuple
    dtype: np.dtype

    def to_memmap(self):
        return np.memmap(self.filename, dtype=self.dtype, mode="r+", shape=self.shape)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "DiskArray":
        fd, path = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr
        mm.flush()
        return cls(path, arr.shape, arr.dtype)

    def _to_blocked(self) -> "BlockedArray":
        """Bridge to BlockedArray so ops can reuse the existing OOCAlgorithm
        block-loop machinery. Derives page_shape from the current policy."""
        from .base_page import get_default_policy, derive_page_shape

        policy = get_default_policy()
        page_shape = derive_page_shape(policy, self.dtype, self.shape)
        return BlockedArray(self.filename, self.shape, self.dtype, page_shape)

    @classmethod
    def _from_blocked(cls, ba: "BlockedArray") -> "DiskArray":
        return cls(ba.filename, ba.full_shape, ba.dtype)

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


@dataclass(frozen=True)
class DiskArrayType(HiType):
    """Tells JAX the shape/dtype of a DiskArray and how to lower/raise it."""

    shape: tuple
    dtype: np.dtype

    def lo_ty(self):
        return [ShapedArray(self.shape, self.dtype)]

    def lower_val(self, val: DiskArray):
        # Return the memmap directly - np.memmap is a subclass of ndarray,
        # data stays on disk, only touched pages fault into RAM.
        return [np.memmap(val.filename, dtype=val.dtype, mode="r", shape=val.shape)]

    def raise_val(self, arr):
        # In practice, our HiPrimitives return DiskArray directly from
        # expand() so raise_val is never called; kept for API completeness.
        return DiskArray.from_numpy(np.asarray(arr))

    def to_tangent_aval(self):
        return self

    def vspace_zero(self):
        return DiskArray.from_numpy(np.zeros(self.shape, dtype=self.dtype))

    def vspace_add(self, x, y):
        return x + y


register_hitype(DiskArray, lambda v: DiskArrayType(v.shape, v.dtype))


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

    @property
    def grad(self) -> "BlockedArray":
        return BlockedArray(
            self.filename + ".grad", self.full_shape, self.dtype, self.page_shape
        )

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
    BlockedArray,
    data_fields=["_marker"],
    meta_fields=["filename", "full_shape", "dtype", "page_shape"],
)


@dataclass(frozen=True)
class SpillFile(BlockedArray):
    """Ephemeral BlockedArray - owns its own temp file, auto-deletes on cleanup."""

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
