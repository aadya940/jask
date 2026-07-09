"""End-to-end demo of jask: disk-backed arrays, differentiable and
JIT-compatible, without ever materializing the full arrays in memory.

Run with:
    conda activate scipy-dev && python example.py
"""

import numpy as np
import jax
import optax

import jask


def main():
    jask.set_memory_budget("1GB")

    np.random.seed(0)
    A = np.random.rand(4, 6).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)
    C = np.random.rand(4, 3).astype(np.float32)
    T = np.random.rand(4, 3).astype(np.float32)

    a = jask.DiskArray.from_numpy(A)
    b = jask.DiskArray.from_numpy(B)
    c = jask.DiskArray.from_numpy(C)
    target = jask.DiskArray.from_numpy(T)

    # Full MSE loss - hijax primitives compose naturally with jax.grad.
    def mse_loss(a, b, c, target):
        z = jask.dot(jask.dot(a, b), c)
        diff = jask.sub(z, target)
        sq = jask.square(diff)
        return jask.sum(sq)

    grad_A, grad_B, grad_C = jax.grad(mse_loss, argnums=(0, 1, 2))(a, b, c, target)
    dA = np.asarray(grad_A.to_memmap())
    dB = np.asarray(grad_B.to_memmap())
    dC = np.asarray(grad_C.to_memmap())

    diff = A @ B @ C - T
    expected_dA = 2 * diff @ C.T @ B.T
    expected_dB = A.T @ (2 * diff) @ C.T
    expected_dC = (A @ B).T @ (2 * diff)

    print("mse dA matches:", np.allclose(dA, expected_dA, atol=1e-3))
    print("mse dB matches:", np.allclose(dB, expected_dB, atol=1e-3))
    print("mse dC matches:", np.allclose(dC, expected_dC, atol=1e-3))

    # optax.sgd training step - grads are real DiskArrays.
    lr = 0.01
    opt = optax.sgd(lr)
    opt_state = opt.init(a)

    grad_a = jax.grad(lambda a: jask.sum(a))(a)
    updates, opt_state = opt.update(grad_a, opt_state)
    # optax.apply_updates uses jnp.asarray internally; use `+` directly.
    new_a = a + updates

    result = np.asarray(new_a.to_memmap())
    expected = A - lr * np.ones_like(A)
    print("optax sgd step matches:", np.allclose(result, expected, atol=1e-4))


if __name__ == "__main__":
    main()
