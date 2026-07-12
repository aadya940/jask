"""Tests for the weakref-based file cleanup added to fix the temp-file
leak (see log.md's Roadmap) - and the update_() regression found while
building the README example: reassigning `a = a.update_(...)` used to
delete the file the new object still needed, because the old object's
finalizer fired the moment it lost its last reference.
"""

import gc
import os

import numpy as np
import jax
import jask


def _gc():
    gc.collect()
    gc.collect()  # a second pass catches anything freed by the first


def test_from_numpy_file_deleted_on_gc(rng):
    """A fresh DiskArray's file is cleaned up once nothing references it."""
    A = rng.random((4, 6)).astype(np.float32)
    a = jask.DiskArray.from_numpy(A)
    path = a.filename
    assert os.path.exists(path)

    del a
    _gc()
    assert not os.path.exists(path)


def test_eager_op_output_file_deleted_on_gc(small_2d_matched):
    """A fresh eager op result's file is cleaned up once unreferenced -
    the actual fix for the leak: every intermediate in a chain used to
    live forever."""
    _, _, a, b = small_2d_matched
    c = jask.add(a, b)
    path = c.filename
    assert os.path.exists(path)

    del c
    _gc()
    assert not os.path.exists(path)


def test_update_reassignment_does_not_delete_the_file(small_2d):
    """Regression test: `a = a.update_(new)` must not delete the file -
    the old object (which owns the cleanup finalizer) loses its last
    reference the instant `a` gets rebound, but the new object points at
    the SAME file and must keep working."""
    A, a = small_2d
    original_path = a.filename

    new_val = jask.DiskArray.from_numpy(A * 2)
    a = a.update_(new_val)
    _gc()

    assert a.filename == original_path
    assert os.path.exists(original_path)
    assert np.allclose(np.asarray(a.to_memmap()), A * 2, atol=1e-5)


def test_update_survives_many_reassignments(rng):
    """update_() called repeatedly (a training loop's actual pattern)
    must not break or accumulate deleted-file errors across iterations."""
    A = rng.random((4, 6)).astype(np.float32)
    a = jask.DiskArray.from_numpy(A)
    original_path = a.filename

    for step in range(10):
        new_val = jask.DiskArray.from_numpy(A + step)
        a = a.update_(new_val)
        _gc()
        assert a.filename == original_path
        assert np.allclose(np.asarray(a.to_memmap()), A + step, atol=1e-5)


def test_update_then_use_in_jit_grad_does_not_crash(small_2d_matched):
    """The exact scenario that broke before the fix: update_() inside a
    loop, then feeding the result into jax.jit(jax.grad(...)) again."""
    A, B, a, b = small_2d_matched

    def loss(a, b):
        return jask.sum(jask.add(a, b))

    grad_fn = jax.jit(jax.grad(loss))
    for _ in range(3):
        grad_a = grad_fn(a, b)
        new_val = a + (-0.01) * grad_a
        a = a.update_(new_val)

    assert os.path.exists(a.filename)
    # no exception means the fix holds - also check we can still read it
    np.asarray(a.to_memmap())


def test_grad_output_file_survives_gc_of_intermediate_wrappers(small_2d):
    """jax.jit(jax.grad(...))'s output must survive even though JAX
    reconstructs transient DiskArray wrappers around the same file
    internally (raise_val) - those must never carry their own finalizer."""
    A, a = small_2d
    grad_a = jax.jit(jax.grad(jask.sum))(a)
    _gc()

    assert os.path.exists(grad_a.filename)
    assert np.allclose(np.asarray(grad_a.to_memmap()), 1.0, atol=1e-4)
