"""Small internal helpers used by OOCAlgorithm - not part of the public API."""

import jax

from .base_page import IOCost


class _ReusingBlockReader:
    """Reads and device_puts ONE block at a time per input position, reusing
    the previous result when the requested index is identical to last time -
    e.g. matmul's A row-block repeats identically across every consecutive j
    for the same i.

    Must read one block at a time, not a whole K-sequence up front: an op
    like Sum has a single output tile whose index_map lists every block in
    the entire input, so pre-loading the whole sequence at once would pull
    the entire array into RAM regardless of the memory budget - exactly the
    thing tiling exists to prevent. Only remembers the single most recent
    read per position, so it's freed automatically once the reader itself
    goes out of scope at the end of one run_forward/run_backward call.
    """

    def __init__(self, num_inputs: int, io_tracker: IOCost):
        """Set up empty per-input-position reuse state."""
        self._io_tracker = io_tracker
        self._last_idx = [None] * num_inputs
        self._last_block = [None] * num_inputs

    def read_one(self, pos: int, arr, idx: tuple):
        """Return the device block at idx, reusing the last read if unchanged."""
        if idx == self._last_idx[pos]:
            return self._last_block[pos]
        block = jax.device_put(arr.read_block(idx, self._io_tracker))
        self._last_idx[pos] = idx
        self._last_block[pos] = block
        return block
