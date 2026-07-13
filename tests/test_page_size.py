"""Tests for jask's Config: per-op-type page-size derivation, the
one-time available-memory safety clamp, and the RAM-backed-filesystem
(tmpfs) scratch_dir check - added after a real, serious finding: every
one of jask's internal tempfile.mkstemp() calls used to have no `dir=`,
so every op's output/gradient files silently landed wherever Python's
tempfile default resolved to - /tmp on this dev machine, which is
tmpfs (RAM), not disk. This defeated jask's entire out-of-core
guarantee invisibly, for the whole session, until caught by accident.
See log.md's Roadmap for the full story.

Deliberately NOT using pytest's tmp_path fixture for scratch_dir in most
of these: tmp_path is commonly placed under /tmp, which is exactly the
RAM-backed filesystem this whole feature exists to catch (confirmed on
this dev machine) - using it here would make tests fail (or silently
pass for the wrong reason) depending on the machine they run on. Tests
that don't care about the scratch_dir's filesystem type explicitly opt
out via allow_tmpfs=True instead; tests that need a real disk directory
use one under the repo's own working directory.
"""

import os
import shutil
from unittest.mock import patch

import numpy as np
import pytest

import jask.base.base_page as base_page
from jask.base.base_page import (
    Config,
    _is_ram_backed,
    _peak_blocks,
    derive_page_shape,
    get_config,
    scratch_mkstemp,
    set_memory_budget,
)

REAL_DISK_DIR = os.path.join(os.getcwd(), ".test_scratch")


@pytest.fixture(autouse=True)
def _restore_config():
    """These tests call set_memory_budget, which mutates process-wide
    state - restore it afterward so other tests (which assume the
    session fixture's "1GB" budget is still active) aren't affected."""
    original = base_page._config
    yield
    base_page._config = original
    shutil.rmtree(REAL_DISK_DIR, ignore_errors=True)


@pytest.mark.parametrize(
    "num_inputs,phase,pipelined,expected",
    [
        (1, "forward", False, 3),
        (1, "backward", False, 4),
        (2, "forward", False, 4),
        (2, "backward", False, 6),
        (1, "forward", True, 6),
        (2, "backward", True, 12),
    ],
)
def test_peak_blocks_formula(num_inputs, phase, pipelined, expected):
    """num_inputs + 2 (forward), 2*num_inputs + 2 (backward), doubled if
    pipelined - traced precisely from OOCAlgorithm's actual loops."""
    assert _peak_blocks(num_inputs, phase, pipelined) == expected


def test_peak_blocks_rejects_invalid_phase():
    """Only 'forward'/'backward' are valid - a typo shouldn't silently
    produce a wrong (possibly unsafe) block size."""
    with pytest.raises(ValueError):
        _peak_blocks(1, "sideways")


def test_backward_gets_smaller_blocks_than_forward():
    """Backward has more peak-resident blocks than forward for the same
    op, so it must get a smaller page_size - not just different, smaller
    in the correct direction."""
    config = Config(max_memory=3 * 1024**3, scratch_dir=REAL_DISK_DIR)
    shape = (50000, 50000)
    fwd = derive_page_shape(config, np.float32, shape, num_inputs=2, phase="forward")
    bwd = derive_page_shape(config, np.float32, shape, num_inputs=2, phase="backward")
    fwd_bytes = fwd[0] * fwd[1] * 4
    bwd_bytes = bwd[0] * bwd[1] * 4
    assert bwd_bytes < fwd_bytes


def test_derive_page_shape_is_deterministic():
    """Same inputs must always produce the same page_shape - no hidden
    state, since callers rely on every array in one op call using
    consistent (even if not identical, for differently-shaped arrays)
    derivation."""
    config = Config(max_memory=3 * 1024**3, scratch_dir=REAL_DISK_DIR)
    shape = (50000, 50000)
    r1 = derive_page_shape(config, np.float32, shape, num_inputs=2, phase="forward")
    r2 = derive_page_shape(config, np.float32, shape, num_inputs=2, phase="forward")
    assert r1 == r2


def test_set_memory_budget_does_not_clamp_a_conservative_request():
    """If the user's budget is already well below available memory, the
    clamp must not touch it - it's a safety net, not an override."""
    with patch("jask.base.base_page.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 10 * 1024**3  # 10GB "available"
        user_bytes = 1 * 1024**3  # 1GB requested - well under 0.8*10GB
        set_memory_budget(user_bytes, scratch_dir=REAL_DISK_DIR)
        assert get_config().max_memory == user_bytes


def test_set_memory_budget_clamps_an_optimistic_request():
    """If the user's budget exceeds 80% of actual available memory, it
    must be clamped - this is the exact gap that caused a real OOM crash
    (a benchmark using a fixed '4GB' budget that didn't match the
    machine's actual available memory at the time)."""
    with patch("jask.base.base_page.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 2 * 1024**3  # 2GB "available"
        user_bytes = 10 * 1024**3  # 10GB requested - way more than available
        set_memory_budget(user_bytes, scratch_dir=REAL_DISK_DIR)
        assert get_config().max_memory == int(0.8 * 2 * 1024**3)


def test_set_memory_budget_accepts_string_units():
    """Regression: the clamp shouldn't break the existing string-budget
    API (e.g. "512MB")."""
    with patch("jask.base.base_page.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 10 * 1024**3
        set_memory_budget("512MB", scratch_dir=REAL_DISK_DIR)
        assert get_config().max_memory == 512 * 1024**2


# --- scratch_dir / tmpfs safety ---


def test_set_memory_budget_uses_given_scratch_dir():
    """The resolved scratch_dir must be exactly what was passed in."""
    set_memory_budget("1GB", scratch_dir=REAL_DISK_DIR)
    assert get_config().scratch_dir == REAL_DISK_DIR


def test_scratch_mkstemp_creates_files_under_scratch_dir():
    """The whole point: files jask creates must actually land in
    scratch_dir, not wherever Python's tempfile default resolves to."""
    set_memory_budget("1GB", scratch_dir=REAL_DISK_DIR)
    fd, path = scratch_mkstemp(suffix=".dat")
    os.close(fd)
    assert os.path.dirname(path) == REAL_DISK_DIR
    assert os.path.exists(path)


def test_is_ram_backed_detects_tmpfs():
    """Sanity check against something we know is RAM-backed on Linux -
    /dev/shm is universally tmpfs."""
    if not os.path.isdir("/dev/shm"):
        pytest.skip("/dev/shm not present on this system")
    assert _is_ram_backed("/dev/shm") is True


def test_is_ram_backed_false_for_real_disk():
    """A directory under the repo's own cwd must not be flagged as
    RAM-backed - this is the actual disk this test suite runs from."""
    os.makedirs(REAL_DISK_DIR, exist_ok=True)
    assert _is_ram_backed(REAL_DISK_DIR) is False


def test_set_memory_budget_rejects_tmpfs_scratch_dir_by_default():
    """The actual regression test for the real bug: writing DiskArray
    data to a RAM-backed scratch_dir must be refused, not silently
    allowed - this is what should have caught the original problem
    immediately instead of an entire session of confusion."""
    if not os.path.isdir("/dev/shm"):
        pytest.skip("/dev/shm not present on this system")
    with pytest.raises(RuntimeError, match="RAM-backed"):
        set_memory_budget("1GB", scratch_dir="/dev/shm/jask_test_scratch")


def test_set_memory_budget_allows_tmpfs_with_explicit_opt_in():
    """allow_tmpfs=True must bypass the check for users who genuinely
    want it (e.g. deliberately testing with small data)."""
    if not os.path.isdir("/dev/shm"):
        pytest.skip("/dev/shm not present on this system")
    try:
        set_memory_budget(
            "1GB", scratch_dir="/dev/shm/jask_test_scratch", allow_tmpfs=True
        )
        assert get_config().scratch_dir == "/dev/shm/jask_test_scratch"
    finally:
        shutil.rmtree("/dev/shm/jask_test_scratch", ignore_errors=True)


def test_set_memory_budget_default_scratch_dir_is_not_rejected():
    """The default (no scratch_dir given) must itself pass the tmpfs
    check when run from a real-disk working directory - a bad default
    should be caught, not silently trusted, but a good default
    shouldn't force every caller to pass scratch_dir explicitly.

    Deliberately NOT deleting the resulting directory afterward - this
    is the SAME .jask_scratch the session-wide conftest fixture already
    created and every other test in the suite relies on for the rest of
    the session; removing it here would break everything that runs after.
    """
    set_memory_budget("1GB")  # no scratch_dir - uses the default
    expected = os.path.join(os.getcwd(), ".jask_scratch")
    assert get_config().scratch_dir == expected
