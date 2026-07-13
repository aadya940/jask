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
- `Policy`: the process-wide memory budget, set once via `set_memory_budget`.
- Memory-budget helpers: `set_memory_budget`, `get_default_policy`,
  `derive_page_shape`, `align_to_os_page`.

Block size is derived per op call from `num_inputs`/`phase`/`pipelined`
(see `_peak_blocks`) - the number of blocks actually resident at once
differs for a single-input op's forward pass vs. a two-input op's
backward pass, so a single fixed divisor doesn't correctly bound every
case. This is still a *static* bound per op-type (known ahead of time,
not something that changes moment to moment during execution) - not to
be confused with re-deriving block size based on live conditions, which
`set_memory_budget`'s one-time clamp below explicitly avoids doing.
"""

from dataclasses import dataclass

import numpy as np

import psutil


@dataclass
class IOCost:
    """I/O Cost of a particular algorithm."""

    total_pages: int


OS_PAGE_SIZE = 4096  # standard OS memory page, bytes


def align_to_os_page(size: int) -> int:
    """Round a byte size down to the nearest multiple of OS_PAGE_SIZE.

    Block boundaries should land on OS page boundaries so disk reads don't
    span a partial extra page.
    """
    return size - (size % OS_PAGE_SIZE)


@dataclass
class Policy:
    """The process-wide memory budget - see `set_memory_budget`."""

    max_memory: int


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


def set_memory_budget(max_memory: int | str):
    """Set the process-wide memory budget used by every jask op.

    Call this once at the start of your program, not per-op-call. Every
    op (``jask.dot``, ``jask.add``, etc.) that doesn't receive an
    explicit `Policy` uses this budget to decide how large a tile ("page")
    of an array it reads into memory at a time.

    The effective budget is ``min(max_memory, 0.8 * currently available
    RAM)`` - a one-time safety clamp taken at the moment this function is
    called, so a budget that doesn't match actual conditions on this
    machine (e.g. copied from a different machine, or simply optimistic)
    doesn't go completely unchecked. This is checked once here, not
    continuously during execution - block size stays fixed for the rest
    of the session once set, the same way it would if you'd passed a
    correct value yourself. It does not protect against available memory
    dropping significantly *after* this call, only against a mismatch
    that already exists when you call it.

    Parameters
    ----------
    max_memory : int or str
        The memory budget, either in bytes (int) or as a string with a
        unit suffix, e.g. ``"4GB"``, ``"512MB"``.

    Examples
    --------
    >>> import jask
    >>> jask.set_memory_budget("4GB")
    """
    global _default_policy
    user_bytes = _parse_memory(max_memory)
    available = psutil.virtual_memory().available
    effective_bytes = min(user_bytes, int(0.8 * available))
    _default_policy = Policy(max_memory=effective_bytes)


def get_default_policy() -> Policy:
    if _default_policy is None:
        raise RuntimeError(
            'No memory budget set. Call jask.set_memory_budget("4GB") first.'
        )
    return _default_policy


def _peak_blocks(num_inputs: int, phase: str, pipelined: bool = False) -> int:
    """How many blocks are actually resident at once for a given op call -
    traced precisely from OOCAlgorithm's forward/backward loops:
    - forward: num_inputs (the reader's cached block per input) + 1
      (the freshly computed partial) + 1 (the running accumulator).
    - backward: num_inputs (reader cache) + num_inputs (this k's gradient
      blocks) + 1 (d_out_block, held for the whole inner loop) + 1
      (the transient "existing" value during the scatter-write).
    - pipelined (prefetching the next block while processing the
      current one) roughly doubles whichever of the above applies, since
      two blocks' worth of work are in flight instead of one.
    """
    if phase == "forward":
        base = num_inputs + 2
    elif phase == "backward":
        base = 2 * num_inputs + 2
    else:
        raise ValueError(f"phase must be 'forward' or 'backward', got {phase!r}")
    return base * 2 if pipelined else base


def derive_page_shape(
    policy: Policy,
    dtype: np.dtype,
    full_shape: tuple,
    num_inputs: int,
    phase: str,
    pipelined: bool = False,
) -> tuple:
    """Pick a block shape sized so that `_peak_blocks` blocks of it fit in
    `policy.max_memory`, distributing it evenly across full_shape's
    dimensions (an N-th root split), clipped so no dimension's block
    exceeds the array's own extent in that dim.
    """
    peak = _peak_blocks(num_inputs, phase, pipelined)
    itemsize = np.dtype(dtype).itemsize
    page_size = align_to_os_page(max(OS_PAGE_SIZE, policy.max_memory // peak))
    elements_per_page = max(1, page_size // itemsize)
    ndim = len(full_shape)
    side = max(1, int(round(elements_per_page ** (1 / ndim))))
    return tuple(min(side, s) for s in full_shape)
