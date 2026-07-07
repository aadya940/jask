"""Forward-pass throughput at genuinely large scale: 1GiB arrays, where
Dask's own auto-chunker (128MB target) is forced to actually tile, unlike
the earlier benchmark where arrays were too small to trigger real chunking.

Chunk size (page_shape) is derived by jask itself from the memory budget,
not hand-picked — jask.set_memory_budget(...) + derive_page_shape decide it,
and jask.dot is called directly, no manual Policy/Dot/make_jax_op wiring.

Run with: conda activate scipy-dev && python benchmark_1gb.py
"""

import os
import tempfile
import time

import numpy as np
import dask
import dask.array as da

import jask
from jask.base import DiskArray, get_default_policy, derive_page_shape

# Real ext4-on-NVMe disk, not /tmp (which is tmpfs — RAM-backed, no real
# disk latency at all). Confirmed via `mount` before this benchmark.
DISK_DIR = "/home/aadya-chinubhai/Desktop/projects/personal-projects/.benchdata"


def make_array(data: np.ndarray, page_shape: tuple) -> DiskArray:
    fd, path = tempfile.mkstemp(suffix=".dat", dir=DISK_DIR)
    os.close(fd)
    arr = DiskArray.create(path, data.shape, data.dtype, page_shape)
    mm = arr._mmap(mode="r+")
    mm[:] = data
    mm.flush()
    return arr


def mean_std(times):
    arr = np.array(times)
    return arr.mean(), arr.std()


def main():
    N = 16384  # 16384^2 * 4 bytes = exactly 1 GiB per array
    n_trials = 2

    jask.set_memory_budget("4GB")
    policy = get_default_policy()
    page_shape = derive_page_shape(policy, np.float32, (N, N))

    print(f"N={N} ({(N*N*4)/1024**3:.2f} GiB per array), jask-derived "
          f"page_shape={page_shape} (from 4GB budget), {n_trials} trials\n")

    print("Generating input arrays (this itself takes a moment at 1GiB each)...")
    np.random.seed(0)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # Reference for correctness — use a small submatrix only, since a full
    # 16384x16384 numpy matmul is itself a large, slow computation and not
    # the point of this benchmark.
    check_n = 512
    expected_corner = A[:check_n] @ B[:, :check_n]

    # --- jask, fully self-configured: no Policy/Dot/make_jax_op by hand ---
    jask_times = []
    for i in range(n_trials):
        a = make_array(A, page_shape)
        b = make_array(B, page_shape)
        t0 = time.perf_counter()
        y = jask.dot(a, b)
        arr = np.memmap(y.filename, dtype=y.dtype, mode="r", shape=y.full_shape)
        corner = np.array(arr[:check_n, :check_n])
        jask_times.append(time.perf_counter() - t0)
        ok = np.allclose(corner, expected_corner, atol=1e-1)
        print(f"  jask trial {i+1}: {jask_times[-1]:.2f}s, correct={ok}")
        for f in (a.filename, b.filename, y.filename):
            os.remove(f)

    # --- dask, chunk size matched to jask's derived page_shape ---
    dask_matched_times = []
    for i in range(n_trials):
        da_a = da.from_array(A, chunks=page_shape)
        da_b = da.from_array(B, chunks=page_shape)
        t0 = time.perf_counter()
        result = (da_a @ da_b)[:check_n, :check_n].compute()
        dask_matched_times.append(time.perf_counter() - t0)
        ok = np.allclose(result, expected_corner, atol=1e-1)
        print(f"  dask (matched chunk) trial {i+1}: {dask_matched_times[-1]:.2f}s, correct={ok}")

    # --- dask, auto chunk (its own 128MB-target heuristic) ---
    dask_auto_times = []
    da_a_auto = da.from_array(A, chunks="auto")
    print(f"  dask auto chunk size chosen: {da_a_auto.chunksize}")
    for i in range(n_trials):
        da_a = da.from_array(A, chunks="auto")
        da_b = da.from_array(B, chunks="auto")
        t0 = time.perf_counter()
        result = (da_a @ da_b)[:check_n, :check_n].compute()
        dask_auto_times.append(time.perf_counter() - t0)
        ok = np.allclose(result, expected_corner, atol=1e-1)
        print(f"  dask (auto chunk) trial {i+1}: {dask_auto_times[-1]:.2f}s, correct={ok}")

    print()
    m1, s1 = mean_std(jask_times)
    m2, s2 = mean_std(dask_matched_times)
    m3, s3 = mean_std(dask_auto_times)
    print(f"jask (self-configured):        {m1:.2f}s ± {s1:.2f}")
    print(f"dask (matched to jask's chunk): {m2:.2f}s ± {s2:.2f}")
    print(f"dask (auto chunk):              {m3:.2f}s ± {s3:.2f}")


if __name__ == "__main__":
    main()
