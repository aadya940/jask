"""Memory-bound regression tests, using pytest-memray.

These encode the actual bugs found and fixed via manual bigger-than-RAM
stress testing (see log.md): the block reader used to batch-load an
entire K-sequence at once (fatal for reductions like Sum, where K spans
every block in the input), and set_memory_budget used to silently do
nothing. Both would show up here as a peak memory usage close to the
test array's own size instead of a small, bounded multiple of one block.

Arrays here are sized to require many blocks under a small budget - large
enough that a regression to "load everything" would clearly blow the
limit, small enough to keep the test suite fast.
"""

import numpy as np
import jax
import pytest

import jask


@pytest.fixture
def many_block_array(rng):
    """~380MB array, forced into ~36 blocks by a small memory budget -
    large enough that a full-materialization regression is unmissable,
    small enough to keep this test fast."""
    original_policy = jask.base.base_page._default_policy
    jask.set_memory_budget("16MB")
    A = rng.random((10000, 10000)).astype(np.float32)
    a = jask.DiskArray.from_numpy(A)
    yield A, a
    jask.base.base_page._default_policy = original_policy


## NOTE on the limits below: memray's default accounting counts each
## np.memmap() call's FULL requested virtual mapping size as "allocated" -
## not actual resident physical memory (which is what the fixed bugs were
## about - block-count-scaling RSS, not virtual address space). A single
## ~380MB array legitimately gets re-mapped a small, fixed number of times
## per call (traced precisely earlier: ~2x for a grad call, ~2x more for
## the jit(grad) path) - that's correct, expected behavior, not the bug.
## The regression this test actually catches: the OLD block reader
## batch-loading a whole K-sequence via jax.device_put would show up as
## many MORE additional megabytes in `batched_device_put`'s own allocation
## site specifically (roughly the array's full size again, on top of the
## expected remappings) - that's what would blow well past these limits.


@pytest.mark.limit_memory("900 MB")
def test_sum_forward_stays_bounded(many_block_array):
    """jask.sum(a)'s forward pass must not pull the whole array into RAM -
    regression test for the block reader's old K-sequence batch-load bug."""
    A, a = many_block_array
    result = jask.sum(a)
    assert np.isclose(float(result), A.sum(), atol=1e-1)


@pytest.mark.limit_memory("1200 MB")
def test_grad_stays_bounded(many_block_array):
    """jax.grad(jask.sum)(a)'s backward pass must also stay bounded, not
    just the forward pass - regression test covering _accumulate_grad_block."""
    A, a = many_block_array
    grad_a = jax.grad(jask.sum)(a)
    assert np.allclose(np.asarray(grad_a.to_memmap()[:10, :10]), 1.0, atol=1e-4)


@pytest.mark.limit_memory("1600 MB")
def test_jit_of_grad_stays_bounded(many_block_array):
    """jax.jit(jax.grad(jask.sum))(a) must stay bounded too - the
    zero-materialization path, most directly exercised by today's fixes."""
    A, a = many_block_array
    grad_a = jax.jit(jax.grad(jask.sum))(a)
    assert grad_a._lo_tracer.shape == ()
    assert np.allclose(np.asarray(grad_a.to_memmap()[:10, :10]), 1.0, atol=1e-4)
