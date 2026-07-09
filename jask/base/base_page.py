"""Metadata and capacity-planning primitives for disk-backed block computation.

An array too large for device (or host) memory is partitioned into a grid of
fixed-size blocks ("pages"), each backed by a slice of an `np.memmap`-mapped
file on disk. Disk-to-host transfer is handled implicitly by the OS's page
cache; host-to-device transfer is explicit and must stay within a fixed
memory budget, since no equivalent paging mechanism exists across that
boundary.

This module defines:
- `IOCost`: a running tally of host-to-device transfers, used to reason
  about and benchmark the cost of a computation.
- `Policy`: derives how many blocks can be held in flight at once from a
  user-specified device memory budget and available host memory, so that
  computation can safely use more than the bare minimum working set
  without ever exceeding either budget.
- Memory-budget helpers: `set_memory_budget`, `get_default_policy`,
  `derive_page_shape`, `align_to_os_page`.
"""

from dataclasses import dataclass

import numpy as np

import psutil


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
