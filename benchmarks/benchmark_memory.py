"""Measure whether jask.dot's memory footprint stays flat as array size
grows - the core "flat memory line" claim from the project spec.

Each size N is run in a fresh subprocess so resource.getrusage's peak-RSS
(a process-lifetime maximum, not a point-in-time snapshot) reflects only
that one call. We also snapshot RSS immediately before and after the
jask.dot call itself, reporting the delta - isolating its footprint from
test-setup cost (np.random.rand allocating full A/B) and from the
verification step (.to_jax() + naive `A @ B`), both of which are O(N^2)
by design and would otherwise dominate the measurement.

Run with: conda activate scipy-dev && python benchmark_memory.py
"""

import subprocess
import sys

_WORKER = r"""
import os, sys, resource, tempfile
import numpy as np
import jask
from jask.base import DiskArray

def make_array(data, page_shape):
    fd, path = tempfile.mkstemp(suffix=".dat")
    os.close(fd)
    arr = DiskArray.create(path, data.shape, data.dtype, page_shape)
    mm = arr._mmap(mode="r+")
    mm[:] = data
    mm.flush()
    return arr

n = int(sys.argv[1])
page_shape = (64, 64)

jask.set_memory_budget("1GB")
A = np.random.rand(n, n).astype(np.float32)
B = np.random.rand(n, n).astype(np.float32)
a = make_array(A, page_shape)
b = make_array(B, page_shape)

# Baseline: after test setup (A, B, and their disk copies already exist),
# but before jask.dot itself runs.
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

y = jask.dot(a, b)

# Peak right after jask.dot, before the O(N^2) verification step below.
rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

result = np.asarray(y.to_jax())
ok = np.allclose(result, A @ B, atol=1e-2)

for f in (a.filename, b.filename, y.filename):
    os.remove(f)

delta_mb = rss_after - rss_before
print(f"{n} {rss_after:.1f} {delta_mb:.1f} {ok}")
"""


def main():
    print(
        f"{'N':>6} {'peak RSS (MB)':>15} {'delta over dot() (MB)':>22} {'array bytes (MB)':>18} {'correct':>8}"
    )
    for n in (128, 256, 512, 1024, 2048, 4096):
        result = subprocess.run(
            [sys.executable, "-c", _WORKER, str(n)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"{n:>6}  FAILED: {result.stderr.strip()[-300:]}")
            continue
        n_out, rss, delta, ok = result.stdout.strip().split()
        array_mb = (int(n_out) ** 2 * 4) / (1024 * 1024)
        print(
            f"{n_out:>6} {float(rss):>15.1f} {float(delta):>22.1f} {array_mb:>18.1f} {ok:>8}"
        )


if __name__ == "__main__":
    main()
