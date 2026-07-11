"""Shared pytest fixtures and setup for the jask test suite."""

import numpy as np
import pytest

import jask


@pytest.fixture(scope="session", autouse=True)
def _set_memory_budget():
    """Set a small memory budget once per test session."""
    jask.set_memory_budget("1GB")


@pytest.fixture
def rng():
    """Seeded numpy Generator for reproducible test arrays."""
    return np.random.default_rng(0)


@pytest.fixture
def small_2d(rng):
    """A small 2D numpy array as a DiskArray."""
    A = rng.random((4, 6)).astype(np.float32)
    return A, jask.DiskArray.from_numpy(A)


@pytest.fixture
def small_2d_matched(rng):
    """Two shape-matched 2D DiskArrays (for add/sub/mul tests)."""
    A = rng.random((4, 6)).astype(np.float32)
    B = rng.random((4, 6)).astype(np.float32)
    return A, B, jask.DiskArray.from_numpy(A), jask.DiskArray.from_numpy(B)


def read(result):
    """Read a jask op's result back as a plain numpy array, whether it's a
    DiskArray (elementwise/dot/etc.) or a real jax.Array (sum's output)."""
    if isinstance(result, jask.DiskArray):
        return np.asarray(result.to_memmap())
    return np.asarray(result)
