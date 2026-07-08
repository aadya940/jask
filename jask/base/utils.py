"""Small internal helpers used by OOCAlgorithm - not part of the public API."""

import jax

from .base_page import IOCost


class _ReusingBlockReader:
    """Reads and device_puts blocks per input position, reusing the previous
    result when the requested block-index sequence is identical to last
    time - e.g. matmul's A row-blocks repeat identically across every
    consecutive j for the same i. Not a general cache: only remembers the
    single most recent read per position, so it's freed automatically once
    the reader itself goes out of scope at the end of one run_forward/
    run_backward call.
    """

    def __init__(self, num_inputs: int, io_tracker: IOCost):
        """Set up empty per-input-position reuse state."""
        self._io_tracker = io_tracker
        self._last_idx_seq = [None] * num_inputs
        self._last_blocks = [None] * num_inputs

    def read(self, pos: int, arr, idx_seq: tuple) -> list:
        """Return device blocks for idx_seq, reusing the last read if unchanged."""
        if idx_seq == self._last_idx_seq[pos]:
            return self._last_blocks[pos]
        blocks = [jax.device_put(arr.read_block(i, self._io_tracker)) for i in idx_seq]
        self._last_idx_seq[pos] = idx_seq
        self._last_blocks[pos] = blocks
        return blocks
