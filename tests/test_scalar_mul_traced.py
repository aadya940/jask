"""ScalarMul with the scalar as a genuine traced input, not just a Python
literal closed over in the function body - the roadmap item this was
fixing. Covers gradients w.r.t. both the array and the scalar itself,
since making the scalar a real input means JAX expects both.
"""

import numpy as np
import jax
import jask


def _loss(a, s):
    return jask.sum(jask.mul(a, s))


def test_scalar_as_jit_argument_grad_wrt_array(small_2d):
    """The case that used to fail: scalar passed as a real jit argument,
    not a closed-over literal - grad w.r.t. the array."""
    A, a = small_2d
    grad_a = jax.jit(jax.grad(_loss, argnums=0))(a, 2.5)
    assert np.allclose(np.asarray(grad_a.to_memmap()), 2.5, atol=1e-4)


def test_scalar_as_jit_argument_grad_wrt_scalar(small_2d):
    """The scalar is now a real input, so its own gradient must also be
    correct: d(sum(s * a))/ds = sum(a)."""
    A, a = small_2d
    grad_s = jax.jit(jax.grad(_loss, argnums=1))(a, 2.5)
    assert np.isclose(float(grad_s), A.sum(), atol=1e-2)


def test_scalar_as_jit_argument_both_grads_at_once(small_2d):
    """Both gradients together, in one call."""
    A, a = small_2d
    grad_a, grad_s = jax.jit(jax.grad(_loss, argnums=(0, 1)))(a, 2.5)
    assert np.allclose(np.asarray(grad_a.to_memmap()), 2.5, atol=1e-4)
    assert np.isclose(float(grad_s), A.sum(), atol=1e-2)


def test_scalar_as_plain_grad_argument_no_jit(small_2d):
    """Same, without jit - plain jax.grad alone."""
    A, a = small_2d
    grad_a = jax.grad(_loss, argnums=0)(a, 2.5)
    grad_s = jax.grad(_loss, argnums=1)(a, 2.5)
    assert np.allclose(np.asarray(grad_a.to_memmap()), 2.5, atol=1e-4)
    assert np.isclose(float(grad_s), A.sum(), atol=1e-2)


def test_scalar_closed_over_literal_still_works(small_2d):
    """The old, still-supported pattern: scalar as a Python literal in the
    function body, not a jit argument - must not regress."""
    A, a = small_2d
    grad_a = jax.jit(jax.grad(lambda a: jask.sum(jask.mul(a, 2.5))))(a)
    assert np.allclose(np.asarray(grad_a.to_memmap()), 2.5, atol=1e-4)


def test_scalar_mul_reversed_order_still_works(small_2d):
    """mul(scalar, a) - order-independence must not regress."""
    A, a = small_2d
    result = jask.mul(3.5, a)
    assert np.allclose(np.asarray(result.to_memmap()), 3.5 * A, atol=1e-5)
