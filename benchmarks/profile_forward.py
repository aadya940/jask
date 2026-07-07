"""cProfile both jask.dot() and dask.array's matmul on the same workload,
side by side, to compare where time actually goes."""

import cProfile
import io
import os
import pstats
import tempfile

import numpy as np
import dask.array as da

import jask
from jask.base import DiskArray, get_default_policy, derive_page_shape

DISK_DIR = "/home/aadya-chinubhai/Desktop/projects/personal-projects/.benchdata"


def make_array(data, page_shape):
    fd, path = tempfile.mkstemp(suffix=".dat", dir=DISK_DIR)
    os.close(fd)
    arr = DiskArray.create(path, data.shape, data.dtype, page_shape)
    mm = arr._mmap(mode="r+")
    mm[:] = data
    mm.flush()
    return arr


def print_profile(label, profiler):
    print()
    print("=" * 78)
    print(f"{label}")
    print("=" * 78)
    print("\nTOP 20 by internal (self) time:")
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("tottime").print_stats(20)
    print(s.getvalue())

    print("TOP 20 by cumulative time:")
    s = io.StringIO()
    pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(20)
    print(s.getvalue())


def main():
    N = 16384

    jask.set_memory_budget("4GB")
    policy = get_default_policy()
    page_shape = derive_page_shape(policy, np.float32, (N, N))
    print(f"N={N}, page_shape={page_shape}")

    np.random.seed(0)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    # --- jask ---
    a = make_array(A, page_shape)
    b = make_array(B, page_shape)

    # warmup
    y_warm = jask.dot(a, b)
    os.remove(y_warm.filename)

    prof_jask = cProfile.Profile()
    prof_jask.enable()
    y = jask.dot(a, b)
    _ = np.memmap(y.filename, dtype=y.dtype, mode="r", shape=y.full_shape)[0, 0]
    prof_jask.disable()

    os.remove(y.filename)
    os.remove(a.filename)
    os.remove(b.filename)

    # --- dask, same chunk size as jask ---
    da_a_warm = da.from_array(A, chunks=page_shape)
    da_b_warm = da.from_array(B, chunks=page_shape)
    _ = (da_a_warm @ da_b_warm)[0, 0].compute()

    prof_dask = cProfile.Profile()
    prof_dask.enable()
    da_a = da.from_array(A, chunks=page_shape)
    da_b = da.from_array(B, chunks=page_shape)
    result = (da_a @ da_b).compute()
    prof_dask.disable()

    # --- dask, its own default chunk ---
    _ = (da.from_array(A) @ da.from_array(B))[0, 0].compute()

    prof_dask_auto = cProfile.Profile()
    prof_dask_auto.enable()
    da_a = da.from_array(A)
    da_b = da.from_array(B)
    result = (da_a @ da_b).compute()
    prof_dask_auto.disable()

    print_profile("JASK", prof_jask)
    print_profile("DASK (matched chunk)", prof_dask)
    print_profile("DASK (auto chunk)", prof_dask_auto)


if __name__ == "__main__":
    main()
