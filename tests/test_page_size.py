"""Tests for the per-op-type page-size derivation and the one-time
available-memory safety clamp added to set_memory_budget - see log.md's
Roadmap for why: a fixed, unjustified divisor could pick a block size
that doesn't actually bound the real peak-resident-block count for a
given op's forward/backward pass, and a stale/optimistic user-supplied
budget could go completely unchecked against actual conditions.
"""

from unittest.mock import patch

import numpy as np
import pytest

import jask.base.base_page as base_page
from jask.base.base_page import (
    Policy,
    _peak_blocks,
    derive_page_shape,
    set_memory_budget,
    get_default_policy,
)


@pytest.fixture(autouse=True)
def _restore_default_policy():
    """These tests call set_memory_budget, which mutates process-wide
    state - restore it afterward so other tests (which assume the
    session fixture's "1GB" budget is still active) aren't affected."""
    original = base_page._default_policy
    yield
    base_page._default_policy = original


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
    policy = Policy(max_memory=3 * 1024**3)
    shape = (50000, 50000)
    fwd = derive_page_shape(policy, np.float32, shape, num_inputs=2, phase="forward")
    bwd = derive_page_shape(policy, np.float32, shape, num_inputs=2, phase="backward")
    fwd_bytes = fwd[0] * fwd[1] * 4
    bwd_bytes = bwd[0] * bwd[1] * 4
    assert bwd_bytes < fwd_bytes


def test_derive_page_shape_is_deterministic():
    """Same inputs must always produce the same page_shape - no hidden
    state, since callers rely on every array in one op call using
    consistent (even if not identical, for differently-shaped arrays)
    derivation."""
    policy = Policy(max_memory=3 * 1024**3)
    shape = (50000, 50000)
    r1 = derive_page_shape(policy, np.float32, shape, num_inputs=2, phase="forward")
    r2 = derive_page_shape(policy, np.float32, shape, num_inputs=2, phase="forward")
    assert r1 == r2


def test_set_memory_budget_does_not_clamp_a_conservative_request():
    """If the user's budget is already well below available memory, the
    clamp must not touch it - it's a safety net, not an override."""
    with patch("jask.base.base_page.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 10 * 1024**3  # 10GB "available"
        user_bytes = 1 * 1024**3  # 1GB requested - well under 0.8*10GB
        set_memory_budget(user_bytes)
        assert get_default_policy().max_memory == user_bytes


def test_set_memory_budget_clamps_an_optimistic_request():
    """If the user's budget exceeds 80% of actual available memory, it
    must be clamped - this is the exact gap that caused a real OOM crash
    (a benchmark using a fixed '4GB' budget that didn't match the
    machine's actual available memory at the time)."""
    with patch("jask.base.base_page.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 2 * 1024**3  # 2GB "available"
        user_bytes = 10 * 1024**3  # 10GB requested - way more than available
        set_memory_budget(user_bytes)
        assert get_default_policy().max_memory == int(0.8 * 2 * 1024**3)


def test_set_memory_budget_accepts_string_units():
    """Regression: the clamp shouldn't break the existing string-budget
    API (e.g. "512MB")."""
    with patch("jask.base.base_page.psutil.virtual_memory") as mock_vm:
        mock_vm.return_value.available = 10 * 1024**3
        set_memory_budget("512MB")
        assert get_default_policy().max_memory == 512 * 1024**2
