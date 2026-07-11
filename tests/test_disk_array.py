"""Tests for the DiskArray public class - constructors and dunders.

Op correctness (forward/grad) lives in test_ops.py; these just check the
class's own surface (construction, dunder routing) works.
"""

import operator

import numpy as np
import pytest
import jask


def test_from_numpy_roundtrip(rng):
    """from_numpy writes to disk and to_memmap reads back exact values."""
    A = rng.random((3, 5)).astype(np.float32)
    a = jask.DiskArray.from_numpy(A)
    assert a.shape == (3, 5)
    assert a.dtype == np.float32
    assert np.allclose(np.asarray(a.to_memmap()), A, atol=0)


def test_constructor_from_existing_file(tmp_path, rng):
    """DiskArray(filename, shape, dtype) points at an existing file without copying."""
    A = rng.random((2, 3)).astype(np.float32)
    path = str(tmp_path / "data.dat")
    mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(2, 3))
    mm[:] = A
    mm.flush()

    a = jask.DiskArray(path, shape=(2, 3), dtype=np.float32)
    assert np.allclose(np.asarray(a.to_memmap()), A)


@pytest.mark.parametrize(
    "op,expected",
    [
        pytest.param(operator.add, lambda A, B: A + B, id="add"),
        pytest.param(operator.sub, lambda A, B: A - B, id="sub"),
        pytest.param(operator.mul, lambda A, B: A * B, id="mul"),
    ],
)
def test_dunder_matches_numpy(small_2d_matched, op, expected):
    """`a + b`, `a - b`, `a * b` route through jask's ops and match numpy."""
    A, B, a, b = small_2d_matched
    c = op(a, b)
    assert isinstance(c, jask.DiskArray)
    assert np.allclose(np.asarray(c.to_memmap()), expected(A, B), atol=1e-5)


@pytest.mark.parametrize(
    "make_result",
    [
        pytest.param(lambda a: a * 2.5, id="mul_scalar_right"),
        pytest.param(lambda a: 2.5 * a, id="mul_scalar_left_via_rmul"),
    ],
)
def test_scalar_mul_dunder(small_2d, make_result):
    """`a * scalar` and `scalar * a` (via __rmul__) both work - required by optax."""
    A, a = small_2d
    c = make_result(a)
    assert np.allclose(np.asarray(c.to_memmap()), 2.5 * A, atol=1e-5)
