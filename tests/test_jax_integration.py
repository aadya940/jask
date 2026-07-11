"""Integration with core JAX transformations: vjp, tree_util, jit, and grad
composed together, plus the zero-materialization and update_() guarantees.

Op-level forward/grad correctness lives in test_ops.py - these tests are
about jask composing correctly with the rest of JAX, not re-checking each
op's math.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import jask

from conftest import read

#  jax.vjp


def test_vjp_manual(small_2d):
    """jax.vjp gives forward output and a pullback; jask ops support it natively."""
    A, a = small_2d
    y, pullback = jax.vjp(jask.sum, a)
    (grad_a,) = pullback(jnp.ones_like(y))
    assert np.allclose(read(grad_a), np.ones_like(A), atol=1e-4)


#  jax.tree_util


def test_tree_leaves_treats_diskarray_as_leaf(small_2d):
    """tree_leaves must see DiskArray as one leaf (else optax's tree_map corrupts data)."""
    _, a = small_2d
    leaves = jax.tree_util.tree_leaves(a)
    assert leaves == [a]


def test_tree_map_calls_fn_on_diskarray(small_2d):
    """tree_map should call fn(disk_array), not dive into internal fields."""
    _, a = small_2d
    result = jax.tree_util.tree_map(lambda x: type(x).__name__, a)
    assert result == "DiskArray"


def test_tree_map_scalar_multiplication(small_2d):
    """tree_map(-lr * x, params) - the exact pattern optax runs internally."""
    A, a = small_2d
    lr = 0.1
    scaled = jax.tree_util.tree_map(lambda x: -lr * x, a)
    assert np.allclose(read(scaled), -lr * A, atol=1e-5)


def test_pytree_with_dict_of_diskarrays(rng):
    """Params dict {name: DiskArray} works with tree_map (multi-array training)."""
    A = rng.random((2, 3)).astype(np.float32)
    B = rng.random((2, 3)).astype(np.float32)
    params = {"w1": jask.DiskArray.from_numpy(A), "w2": jask.DiskArray.from_numpy(B)}

    doubled = jax.tree_util.tree_map(lambda x: x * 2.0, params)
    assert np.allclose(read(doubled["w1"]), 2 * A, atol=1e-5)
    assert np.allclose(read(doubled["w2"]), 2 * B, atol=1e-5)


#  jax.jit alone


def _sum_loss(a):
    return jask.sum(a)


def _add_sum_loss(a, b):
    return jask.sum(jask.add(a, b))


def test_jit_forward_matches_eager_single_input(small_2d):
    """jax.jit alone (no grad) on a single-input op matches the eager result."""
    A, a = small_2d
    assert np.isclose(float(jax.jit(_sum_loss)(a)), float(_sum_loss(a)), atol=1e-3)


def test_jit_forward_matches_eager_two_input(small_2d_matched):
    """jax.jit alone (no grad) on a two-input op matches the eager result."""
    A, B, a, b = small_2d_matched
    assert np.isclose(
        float(jax.jit(_add_sum_loss)(a, b)), float(_add_sum_loss(a, b)), atol=1e-3
    )


#  jax.jit(jax.grad(...)) and jax.grad(jax.jit(...))
#
# Requires DiskArrayType's vspace_zero/vspace_add to behave correctly under
# abstract tracing, and backward to run as its own deferred hi-primitive -
# see base_algo.py's make_op docstring for the mechanism.


def _mse_chain_loss(a, d, c, t):
    z = jask.dot(jask.dot(a, d), c)
    diff = jask.sub(z, t)
    return jask.sum(jask.square(diff))


@pytest.mark.parametrize(
    "grad_fn",
    [
        pytest.param(jax.jit(jax.grad(_sum_loss)), id="jit_of_grad"),
        pytest.param(jax.grad(jax.jit(_sum_loss)), id="grad_of_jit"),
    ],
)
def test_jit_and_grad_compose_single_input(small_2d, grad_fn):
    """jit(grad(sum)) and grad(jit(sum)) both give the correct gradient -
    composition order doesn't matter."""
    A, a = small_2d
    grad_a = grad_fn(a)
    assert np.allclose(read(grad_a), np.ones_like(A), atol=1e-4)


def test_jit_of_grad_two_input(small_2d_matched):
    """jit(grad(...)) over a two-input op exercises vspace_add's real code
    path (cotangent combination), not just single-input passthrough."""
    A, B, a, b = small_2d_matched
    grad_a, grad_b = jax.jit(jax.grad(_add_sum_loss, argnums=(0, 1)))(a, b)
    assert np.allclose(read(grad_a), np.ones_like(A), atol=1e-4)
    assert np.allclose(read(grad_b), np.ones_like(B), atol=1e-4)


def test_jit_of_grad_full_chain(rng):
    """jit(grad(...)) over the full MSE chain (dot.dot.sub.square.sum),
    matching analytic gradients - the actual production training-step pattern."""
    A = rng.random((4, 6)).astype(np.float32)
    D = rng.random((6, 4)).astype(np.float32)
    C = rng.random((4, 3)).astype(np.float32)
    T = rng.random((4, 3)).astype(np.float32)
    a, d, c, t = (jask.DiskArray.from_numpy(x) for x in (A, D, C, T))

    grad_a, grad_d, grad_c = jax.jit(jax.grad(_mse_chain_loss, argnums=(0, 1, 2)))(
        a, d, c, t
    )
    diff = A @ D @ C - T
    assert np.allclose(read(grad_a), 2 * diff @ C.T @ D.T, atol=1e-2)
    assert np.allclose(read(grad_d), A.T @ (2 * diff) @ C.T, atol=1e-2)
    assert np.allclose(read(grad_c), (A @ D).T @ (2 * diff), atol=1e-2)


def test_jit_of_grad_with_optax_sgd_step(small_2d_matched):
    """Full production pattern: jax.jit(jax.grad(loss)) feeding optax.sgd,
    run for several steps, loss must actually decrease."""
    import optax

    A, B, a, b = small_2d_matched

    opt = optax.sgd(0.01)
    opt_state = opt.init(a)
    grad_fn = jax.jit(jax.grad(_add_sum_loss))

    loss_before = float(_add_sum_loss(a, b))
    for _ in range(3):
        grad_a = grad_fn(a, b)
        updates, opt_state = opt.update(grad_a, opt_state)
        a = a + updates
    loss_after = float(_add_sum_loss(a, b))

    assert loss_after < loss_before


#  zero-materialization: jit never brings a full array into RAM


def test_jit_forward_never_materializes_full_array(small_2d_matched):
    """Under jit, an array-typed result (add) must stay a trivial marker
    (shape ()) - never the real array."""
    A, B, a, b = small_2d_matched
    result = jax.jit(jask.add)(a, b)
    assert result._lo_tracer.shape == ()


def test_jit_of_grad_never_materializes_full_array(small_2d_matched):
    """Gradients flowing out of jit(grad(...)) must also stay trivial
    markers, not full materialized arrays."""
    A, B, a, b = small_2d_matched
    grad_a = jax.jit(jax.grad(_add_sum_loss))(a, b)
    assert grad_a._lo_tracer.shape == ()


def test_bare_eager_call_stays_lazy(small_2d_matched):
    """A plain eager op call (no jit, no grad) must never populate
    _lo_tracer at all - the out-of-core guarantee for ordinary use."""
    A, B, a, b = small_2d_matched
    c = jask.add(a, b)
    assert c._lo_tracer is None


#  DiskArray.update_(): stable-slot mechanism for jit-compiled loops


def test_update_preserves_filename_identity(small_2d):
    """update_ must return a DiskArray with the SAME filename as self -
    this identity is what lets a jit-compiled loop reuse one executable."""
    A, a = small_2d
    original_filename = a.filename
    new_value = jask.DiskArray.from_numpy(A * 2)
    updated = a.update_(new_value)
    assert updated.filename == original_filename
    assert np.allclose(read(updated), A * 2, atol=1e-5)


def test_update_loop_no_retracing(small_2d_matched):
    """A jit-compiled grad function called repeatedly across an update_-based
    loop must compile exactly once (filename identity stays stable), not
    retrace every step."""
    A, B, a, b = small_2d_matched
    trace_count = {"n": 0}

    def loss(a, b):
        trace_count["n"] += 1  # only runs at Python-trace time, under jit
        return jask.sum(jask.add(a, b))

    grad_fn = jax.jit(jax.grad(loss))
    first_filename = a.filename
    for _ in range(4):
        grad_a = grad_fn(a, b)
        a = a.update_(jask.add(a, jask.mul(grad_a, -0.01)))

    assert trace_count["n"] == 1
    assert a.filename == first_filename
