"""Process-wide configuration and capacity-planning primitives for
disk-backed block computation.

An array too large for device (or host) memory is partitioned into a grid of
fixed-size blocks ("pages"), each backed by a slice of an `np.memmap`-mapped
file on disk. Disk-to-host transfer is handled implicitly by the OS's page
cache; host-to-device transfer is explicit and must stay within a fixed
memory budget, since no equivalent paging mechanism exists across that
boundary.

This module defines:
- `IOCost`: a running tally of host-to-device transfers, used to reason
  about and benchmark the cost of a computation.
- `Config`: the process-wide configuration - memory budget AND scratch
  directory, set once via `set_memory_budget`. Every jask op writes its
  internal files (op outputs, gradient buffers, spill files) under
  `Config.scratch_dir` - never relies on Python's `tempfile` default
  location, which can silently be a RAM-backed filesystem (tmpfs) on
  many modern Linux systems and most containers. Writing "disk-backed"
  array data there would silently consume real memory instead of disk,
  defeating jask's entire out-of-core guarantee - so the resolved
  scratch_dir is checked against this and rejected by default if it's
  RAM-backed (see `_is_ram_backed`).
- `derive_page_shape`: block size, derived per op call from
  `num_inputs`/`phase`/`pipelined` (see `_peak_blocks`) - the number of
  blocks actually resident at once differs for a single-input op's
  forward pass vs. a two-input op's backward pass, so a single fixed
  divisor doesn't correctly bound every case. This is still a *static*
  bound per op-type (known ahead of time, not something that changes
  moment to moment during execution).
"""

import os
import tempfile
from dataclasses import dataclass

import numpy as np

import psutil


@dataclass
class IOCost:
    """I/O Cost of a particular algorithm."""

    total_pages: int


OS_PAGE_SIZE = 4096  # standard OS memory page, bytes
_RAM_BACKED_FSTYPES = {"tmpfs", "ramfs"}


def align_to_os_page(size: int) -> int:
    """Round a byte size down to the nearest multiple of OS_PAGE_SIZE.

    Block boundaries should land on OS page boundaries so disk reads don't
    span a partial extra page.
    """
    return size - (size % OS_PAGE_SIZE)


@dataclass(frozen=True)
class Config:
    """Process-wide jask configuration - see `set_memory_budget`.

    Attributes
    ----------
    max_memory : int
        The effective memory budget in bytes, after the one-time
        available-memory safety clamp.
    scratch_dir : str
        Directory every jask op writes its internal files under (op
        outputs, gradient buffers, spill files) - never a RAM-backed
        filesystem, checked at `set_memory_budget` time.
    """

    max_memory: int
    scratch_dir: str


def _is_ram_backed(path: str) -> bool:
    """True if `path` resolves to a RAM-backed filesystem (tmpfs/ramfs).

    Writing "disk-backed" DiskArray data there would silently consume
    real system memory instead of disk - not a performance quirk, a
    correctness-adjacent violation of jask's entire out-of-core premise.
    Matches by the longest mountpoint prefix of `path` (the most
    specific mount covering it), same approach `df`/`mount` use.
    """
    path = os.path.abspath(path)
    best_match = None
    for part in psutil.disk_partitions(all=True):
        mountpoint = part.mountpoint
        if path == mountpoint or path.startswith(mountpoint.rstrip(os.sep) + os.sep):
            if best_match is None or len(mountpoint) > len(best_match.mountpoint):
                best_match = part
    return best_match is not None and best_match.fstype in _RAM_BACKED_FSTYPES


def _default_scratch_dir() -> str:
    """A starting guess more likely to be real disk than
    `tempfile.gettempdir()` (`/tmp` on most systems, which is RAM-backed
    tmpfs on many modern Linux setups and nearly all containers) - still
    validated by `_is_ram_backed` regardless of where this points, so a
    bad guess here is caught, not trusted blindly.
    """
    return os.path.join(os.getcwd(), ".jask_scratch")


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


_config: Config | None = None


def set_memory_budget(
    max_memory: int | str,
    scratch_dir: str | None = None,
    allow_tmpfs: bool = False,
):
    """Set the process-wide configuration used by every jask op.

    Call this once at the start of your program, not per-op-call. Every
    op (``jask.dot``, ``jask.add``, etc.) that doesn't receive an
    explicit `Config` uses this to decide how large a tile ("page") of
    an array it reads into memory at a time, and where it writes its own
    internal files (op outputs, gradient buffers, spill files).

    The effective memory budget is ``min(max_memory, 0.8 * currently
    available RAM)`` - a one-time safety clamp taken at the moment this
    function is called, so a budget that doesn't match actual conditions
    on this machine doesn't go completely unchecked. This is checked
    once here, not continuously during execution - block size stays
    fixed for the rest of the session once set. It does not protect
    against available memory dropping significantly *after* this call,
    only against a mismatch that already exists when you call it.

    `scratch_dir` is checked against RAM-backed filesystems (tmpfs,
    ramfs - e.g. `/tmp` on many modern Linux systems and most
    containers) and rejected by default: writing "disk-backed" array
    data there would silently consume real memory instead of disk,
    defeating jask's out-of-core guarantee entirely and invisibly.

    Parameters
    ----------
    max_memory : int or str
        The memory budget, either in bytes (int) or as a string with a
        unit suffix, e.g. ``"4GB"``, ``"512MB"``.
    scratch_dir : str, optional
        Directory for jask's own internal files. Defaults to
        ``.jask_scratch`` under the current working directory - still
        subject to the same RAM-backed-filesystem check, so a bad
        default is caught, not trusted blindly. Created if it doesn't
        exist.
    allow_tmpfs : bool, optional
        Set True to allow `scratch_dir` to be RAM-backed anyway (e.g.
        deliberately testing with small data). Default False.

    Raises
    ------
    RuntimeError
        If the resolved `scratch_dir` is on a RAM-backed filesystem and
        `allow_tmpfs` is False.

    Examples
    --------
    >>> import jask
    >>> jask.set_memory_budget("4GB", scratch_dir="/data/jask_scratch")
    """
    global _config
    user_bytes = _parse_memory(max_memory)
    available = psutil.virtual_memory().available
    effective_bytes = min(user_bytes, int(0.8 * available))

    resolved_scratch_dir = os.path.abspath(scratch_dir or _default_scratch_dir())
    os.makedirs(resolved_scratch_dir, exist_ok=True)

    if _is_ram_backed(resolved_scratch_dir) and not allow_tmpfs:
        raise RuntimeError(
            f"scratch_dir {resolved_scratch_dir!r} is on a RAM-backed filesystem "
            "(tmpfs/ramfs) - writing DiskArray data there would silently "
            "consume real memory instead of disk, defeating jask's "
            "out-of-core guarantee. Pass an explicit scratch_dir on real "
            "disk, or allow_tmpfs=True if you genuinely want this (e.g. "
            "small test data)."
        )

    _config = Config(max_memory=effective_bytes, scratch_dir=resolved_scratch_dir)


def get_config() -> Config:
    if _config is None:
        raise RuntimeError(
            'No memory budget set. Call jask.set_memory_budget("4GB") first.'
        )
    return _config


def scratch_mkstemp(suffix: str = ".dat") -> tuple:
    """tempfile.mkstemp(), but always under the configured scratch_dir -
    never Python's tempfile default location (see `set_memory_budget`'s
    scratch_dir/allow_tmpfs docs for why this matters). Every jask op
    that needs a fresh internal file should use this, not
    tempfile.mkstemp directly.
    """
    return tempfile.mkstemp(suffix=suffix, dir=get_config().scratch_dir)


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
    config: Config,
    dtype: np.dtype,
    full_shape: tuple,
    num_inputs: int,
    phase: str,
    pipelined: bool = False,
) -> tuple:
    """Pick a block shape sized so that `_peak_blocks` blocks of it fit in
    `config.max_memory`, distributing it evenly across full_shape's
    dimensions (an N-th root split), clipped so no dimension's block
    exceeds the array's own extent in that dim.
    """
    peak = _peak_blocks(num_inputs, phase, pipelined)
    itemsize = np.dtype(dtype).itemsize
    page_size = align_to_os_page(max(OS_PAGE_SIZE, config.max_memory // peak))
    elements_per_page = max(1, page_size // itemsize)
    ndim = len(full_shape)
    side = max(1, int(round(elements_per_page ** (1 / ndim))))
    return tuple(min(side, s) for s in full_shape)
