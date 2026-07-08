"""Minimal end-to-end demo of jask: disk-backed arrays, differentiable and
JIT-compatible, without ever materializing the full arrays in memory.

Run with the environment that has jax installed:
    conda activate scipy-dev && python example.py
"""

import tempfile
import os

import numpy as np
import jax

import jask
from jask.base import DiskArray


def make_array(data: np.ndarray, page_shape: tuple) -> DiskArray:
    """Write a NumPy array to a temp file and wrap it as a DiskArray."""
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    arr = DiskArray.create(path, data.shape, data.dtype, page_shape)
    mm = arr._mmap(mode="r+")
    mm[:] = data
    mm.flush()
    return arr


def main():
    jask.set_memory_budget("1GB")

    np.random.seed(0)
    A = np.random.rand(4, 6).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)
    C = np.random.rand(4, 3).astype(np.float32)
    T = np.random.rand(4, 3).astype(np.float32)

    a = make_array(A, page_shape=(2, 2))
    b = make_array(B, page_shape=(2, 2))
    c = make_array(C, page_shape=(2, 2))
    target = make_array(T, page_shape=(2, 2))

    # Forward: disk-backed matmul, tiled under the hood.
    y = jask.dot(a, b)
    print("forward matches:", np.allclose(y.to_jax(), A @ B, atol=1e-4))

    # Nested under jax.jit - jask ops compose with normal JAX transformations.
    @jax.jit
    def outer(a, b):
        return jask.dot(a, b)

    print("jit-nested matches:", np.allclose(outer(a, b).to_jax(), A @ B, atol=1e-4))

    # MSE loss end-to-end. Five disk-backed ops chained; jax.grad flows
    # gradients disk-to-disk without materializing any intermediate array.
    def mse_loss(a, b, c, target):
        z = jask.dot(jask.dot(a, b), c)
        diff = jask.sub(z, target)
        sq = jask.square(diff)
        return jask.materialize(jask.sum(sq))

    grad_A, grad_B, grad_C = jax.grad(mse_loss, argnums=(0, 1, 2))(a, b, c, target)
    dA = np.asarray(grad_A.grad.to_jax())
    dB = np.asarray(grad_B.grad.to_jax())
    dC = np.asarray(grad_C.grad.to_jax())

    diff = A @ B @ C - T
    expected_dA = 2 * diff @ C.T @ B.T
    expected_dB = A.T @ (2 * diff) @ C.T
    expected_dC = (A @ B).T @ (2 * diff)

    print("mse dA matches:", np.allclose(dA, expected_dA, atol=1e-3))
    print("mse dB matches:", np.allclose(dB, expected_dB, atol=1e-3))
    print("mse dC matches:", np.allclose(dC, expected_dC, atol=1e-3))


if __name__ == "__main__":
    main()
