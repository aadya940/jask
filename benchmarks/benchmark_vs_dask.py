"""Forward-pass throughput: jask.dot vs dask.array matmul.

Forward-only - Dask has no autodiff, so gradients aren't a comparison
point; that's jask's differentiator, not a speed claim.

Measures three things per array size, over multiple trials:
  - jask "as-used" (jask.dot(a,b) rebuilds and re-JITs the op every call -
    this is the honest cost of today's public API)
  - jask "steady-state" (op built + JIT'd once, reused across calls - what
    it'd cost if make_jax_op's result were cached/reused, which it isn't yet)
  - dask.array, chunk size matched to jask's page_shape
  - dask.array, chunk size at Dask's own tuned default (128MB target)

Run with: conda activate scipy-dev && python benchmark_vs_dask.py
"""

import os
import tempfile
import time

import numpy as np
import dask
import dask.array as da

import jask
from jask.base import DiskArray, Policy
from jask.base import make_jax_op
from jask.linalg.matmul import Dot


def make_array(data: np.ndarray, page_shape: tuple) -> DiskArray:
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    arr = DiskArray.create(path, data.shape, data.dtype, page_shape)
    mm = arr._mmap(mode="r+")
    mm[:] = data
    mm.flush()
    return arr


def time_calls(fn, n_trials):
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def mean_std(times):
    arr = np.array(times)
    return arr.mean(), arr.std()


def bench_jask_as_used(A, B, page, n_trials):
    times = []
    for _ in range(n_trials):
        a = make_array(A, (page, page))
        b = make_array(B, (page, page))
        t0 = time.perf_counter()
        y = jask.dot(a, b)
        result = np.asarray(y.to_jax())
        times.append(time.perf_counter() - t0)
        for f in (a.filename, b.filename, y.filename):
            os.remove(f)
    return times, result


def bench_jask_steady_state(A, B, page, n_trials):
    policy = Policy(max_memory=2 * 1024**3, pages_per_group=3)
    k_blocks = -(-A.shape[1] // page)
    op = Dot(k_blocks=k_blocks)
    jax_op = make_jax_op(op, policy, (page, page))  # built + JIT'd once

    warmup_a = make_array(A, (page, page))
    warmup_b = make_array(B, (page, page))
    _ = np.asarray(jax_op(warmup_a, warmup_b).to_jax())  # trigger compilation
    os.remove(warmup_a.filename)
    os.remove(warmup_b.filename)

    times = []
    for _ in range(n_trials):
        a = make_array(A, (page, page))
        b = make_array(B, (page, page))
        t0 = time.perf_counter()
        y = jax_op(a, b)
        result = np.asarray(y.to_jax())
        times.append(time.perf_counter() - t0)
        for f in (a.filename, b.filename, y.filename):
            os.remove(f)
    return times, result


def bench_dask(A, B, chunks, n_trials):
    times = []
    for _ in range(n_trials):
        da_a = da.from_array(A, chunks=chunks)
        da_b = da.from_array(B, chunks=chunks)
        t0 = time.perf_counter()
        result = (da_a @ da_b).compute()
        times.append(time.perf_counter() - t0)
    return times, result


def main():
    jask.set_memory_budget("2GB")
    dask.config.set({"array.slicing.split_large_chunks": True})

    sizes = [1024, 2048, 4096]
    page = 256
    n_trials = 5

    print(
        f"page_shape=({page},{page}), {n_trials} trials per config, first jask "
        f"call includes JIT compile time (as-used); steady-state excludes it.\n"
    )

    header = (
        f"{'N':>6} {'jask as-used':>16} {'jask steady-state':>20} "
        f"{'dask (matched chunk)':>22} {'dask (auto chunk)':>20}"
    )
    print(header)

    for N in sizes:
        np.random.seed(0)
        A = np.random.rand(N, N).astype(np.float32)
        B = np.random.rand(N, N).astype(np.float32)
        expected = A @ B

        t_as_used, r1 = bench_jask_as_used(A, B, page, n_trials)
        t_steady, r2 = bench_jask_steady_state(A, B, page, n_trials)
        t_dask_matched, r3 = bench_dask(A, B, (page, page), n_trials)
        t_dask_auto, r4 = bench_dask(A, B, "auto", n_trials)

        for name, r in [
            ("as-used", r1),
            ("steady", r2),
            ("dask-matched", r3),
            ("dask-auto", r4),
        ]:
            ok = np.allclose(r, expected, atol=1e-1)
            if not ok:
                print(f"  WARNING: {name} result incorrect at N={N}")

        m1, s1 = mean_std(t_as_used)
        m2, s2 = mean_std(t_steady)
        m3, s3 = mean_std(t_dask_matched)
        m4, s4 = mean_std(t_dask_auto)

        print(
            f"{N:>6} {m1:>7.3f}s±{s1:<6.3f} {m2:>11.3f}s±{s2:<6.3f} "
            f"{m3:>13.3f}s±{s3:<6.3f} {m4:>11.3f}s±{s4:<6.3f}"
        )

    print(
        "\nRatios (>1 = jask slower) computed from the table above by hand if needed."
    )


if __name__ == "__main__":
    main()
