"""Multi-op chains through jax.grad, matching in-memory analytic gradients."""

import numpy as np
import jax
import jask


def test_two_dot_chain_forward(rng):
    """dot . dot forward matches numpy's A @ B @ C - hits the k-block loop twice."""
    A = rng.random((4, 6)).astype(np.float32)
    B = rng.random((6, 4)).astype(np.float32)
    C = rng.random((4, 3)).astype(np.float32)
    a, b, c = (jask.DiskArray.from_numpy(x) for x in (A, B, C))

    z = jask.dot(jask.dot(a, b), c)
    assert np.allclose(np.asarray(z.to_memmap()), A @ B @ C, atol=1e-3)


def test_mse_loss_grads(rng):
    """Full MSE loss end-to-end: dot . dot . sub . square . sum, 3 grads."""
    A = rng.random((4, 6)).astype(np.float32)
    B = rng.random((6, 4)).astype(np.float32)
    C = rng.random((4, 3)).astype(np.float32)
    T = rng.random((4, 3)).astype(np.float32)
    a, b, c, target = (jask.DiskArray.from_numpy(x) for x in (A, B, C, T))

    def loss(a, b, c, target):
        z = jask.dot(jask.dot(a, b), c)
        diff = jask.sub(z, target)
        sq = jask.square(diff)
        return jask.sum(sq)

    grad_A, grad_B, grad_C = jax.grad(loss, argnums=(0, 1, 2))(a, b, c, target)
    dA = np.asarray(grad_A.to_memmap())
    dB = np.asarray(grad_B.to_memmap())
    dC = np.asarray(grad_C.to_memmap())

    diff = A @ B @ C - T
    expected_dA = 2 * diff @ C.T @ B.T
    expected_dB = A.T @ (2 * diff) @ C.T
    expected_dC = (A @ B).T @ (2 * diff)

    assert np.allclose(dA, expected_dA, atol=1e-2)
    assert np.allclose(dB, expected_dB, atol=1e-2)
    assert np.allclose(dC, expected_dC, atol=1e-2)
