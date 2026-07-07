"""Minimal end-to-end example of jask: a disk-backed matmul, differentiable
and JIT-compatible, without ever materializing the full arrays in memory.

Run with the environment that has jax installed, e.g.:
    conda activate scipy-dev && python example.py
"""

import tempfile
import os

import numpy as np
import jax

import jask
from jask.base import DiskArray, gradient_of


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
    # Set once, process-wide , every jask op reads this, none of them take
    # a policy/page_shape argument at the call site.
    jask.set_memory_budget("1GB")

    np.random.seed(0)
    A = np.random.rand(4, 6).astype(np.float32)
    B = np.random.rand(6, 4).astype(np.float32)

    a = make_array(A, page_shape=(2, 2))
    b = make_array(B, page_shape=(2, 2))

    #  forward: disk-backed matmul, tiled under the hood 
    y = jask.dot(a, b)
    result = np.asarray(y.to_jax())  # explicit, deliberate materialization
    print("forward result matches A @ B:", np.allclose(result, A @ B, atol=1e-4))

    #  same call, nested inside an ordinary jax.jit'd function 
    @jax.jit
    def outer(a, b):
        return jask.dot(a, b)

    y_jit = outer(a, b)
    print(
        "jit-nested result matches:",
        np.allclose(np.asarray(y_jit.to_jax()), A @ B, atol=1e-4),
    )

    #  backward: explicit cotangent via jax.vjp 
    # (jax.grad through a jnp.sum(y.to_jax())-style reduction isn't
    # supported yet , to_jax() has no registered gradient rule, so
    # autodiff can't connect a loss back through it. Supplying the
    # cotangent directly is the currently-supported path.)
    dC = np.random.rand(4, 4).astype(np.float32)
    y2, pullback = jax.vjp(jask.dot, a, b)

    mm = np.memmap(y2.filename, dtype=y2.dtype, mode="r+", shape=y2.full_shape)
    mm[:] = dC
    mm.flush()

    grad_a, grad_b = pullback(y2)
    # grad_a/grad_b are placeholders (same identity as a/b) , the real
    # gradient data lives at "<filename>.grad", retrieved via gradient_of.
    dA = np.asarray(gradient_of(grad_a).to_jax())
    dB = np.asarray(gradient_of(grad_b).to_jax())

    expected_dA = dC @ B.T
    expected_dB = A.T @ dC
    print("backward dA matches:", np.allclose(dA, expected_dA, atol=1e-3))
    print("backward dB matches:", np.allclose(dB, expected_dB, atol=1e-3))


if __name__ == "__main__":
    main()
