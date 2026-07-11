"""Optax integration - the main non-JAX ecosystem component we compose with."""

import numpy as np
import jax
import optax
import jask


def _sum_loss(a, b):
    return jask.sum(jask.add(a, b))


def test_sgd_update_matches_manual(small_2d_matched):
    """optax.sgd.update(grads) uses tree_map internally; verifies our hijax type is atomic."""
    A, B, a, b = small_2d_matched
    lr = 0.01

    grad_a = jax.grad(_sum_loss)(a, b)

    opt = optax.sgd(lr)
    opt_state = opt.init(a)
    updates, opt_state = opt.update(grad_a, opt_state)
    new_a = a + updates

    expected = A - lr * np.ones_like(A)
    assert np.allclose(np.asarray(new_a.to_memmap()), expected, atol=1e-4)


def test_adam_init(small_2d):
    """optax.adam.init walks the params pytree; verifies DiskArray flattens as a leaf."""
    A, a = small_2d
    opt = optax.adam(1e-3)
    opt_state = opt.init(a)
    assert opt_state is not None


def test_sgd_multi_step_decreases_loss(small_2d_matched):
    """A short SGD loop actually decreases the loss - end-to-end sanity."""
    A, B, a, b = small_2d_matched
    lr = 0.01

    opt = optax.sgd(lr)
    opt_state = opt.init(a)

    loss_before = float(_sum_loss(a, b))
    for _ in range(3):
        grad_a = jax.grad(_sum_loss)(a, b)
        updates, opt_state = opt.update(grad_a, opt_state)
        a = a + updates
    loss_after = float(_sum_loss(a, b))
    assert loss_after < loss_before
