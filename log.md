# jask status log

## Working

- All 8 ops (add, sub, mul, square, transpose, sum, dot, materialize) - correct, disk-only, eager.
- `jax.grad` alone - real gradients, zero materialization.
- `jax.jit` alone, any op - zero materialization, marker stays `shape=()` even at 500x500.
- `jax.jit(jax.grad(...))` and `jax.grad(jax.jit(...))`, either order - correct gradients, zero materialization.
- Multi-op chains under `jit(grad(...))` (e.g. `dot.dot.sub.square.sum`) - matches analytic gradients.
- `optax.sgd` training loop under `jit(grad(...))` - loss actually decreases.
- Mixed `jask` + plain `jnp` ops/params in one loss, under `jit(grad(...))` - both gradients correct.
- `DiskArray.update_()` - preserves filename identity, so a jit-compiled loop compiles once, no retracing.
- `mul(a, b)` scalar-vs-array dispatch - type-based (`jax.typeof`), works for Python floats and jax scalars, either order, eager and under jit.
- `jask.materialize()` - correct under eager, jit, grad, and jit(grad(...)).

## Fixed this session (real bugs, found via actual bigger-than-RAM stress testing)

- `_ReusingBlockReader` used to load an entire K-sequence at once in a list comprehension. Harmless for ops like `dot` (K bounded by the contraction dim), but `Sum`'s `index_map` returns *every block in the input* as one K-group - so summing a large array pulled the whole thing into RAM at once, defeating tiling entirely. Fixed: reads one block per k-iteration (`read_one`), reusing only the single most recent block per input position.
- `set_memory_budget(...)` never actually set `Policy.page_size` - it always used Dask's hardcoded 128MB default regardless of the budget string passed in, so `derive_page_shape` silently ignored the user's budget entirely. Fixed: `page_size` is now derived from `max_memory // pages_per_group`.
- Gradient output for BlockParallelOp ops went through a redundant detour: written to a scratch `SpillFile`, then fully copied to the deterministic `<filename>.grad` path via `_tiled_copy`. Fixed: `run_backward` now writes directly to the target path when given one - cuts full-array-sized buffer constructions from 5 to 3 per `jax.grad` call on a single-input op (confirmed via instrumented tracing). Effect on peak RSS was real but modest/noisy in testing, not dramatic - this wasn't the dominant memory cost.
- jask was not installed in editable mode - any script run from outside the repo directory (e.g. `/tmp`) silently imported a stale, several-fixes-behind copy from an earlier `pip install .`. This is why an earlier fix appeared not to work on retest. Reinstalled with `pip install -e .`; verified it now resolves to the live repo from any cwd.

## Key finding: peak memory is a roughly-fixed baseline, not array-size-scaling

Traced with `memray` and a moderate-scale sweep (0.064GB to 1.024GB, fixed block size): final RSS stays roughly constant (~0.3-0.4GB) regardless of array size, meaning the RSS/array-size ratio *decreases* as arrays get bigger (4.6x at 64MB down to ~0.35-0.37x at 1GB). This baseline is dominated by JAX/XLA/Python overhead and mmap page-cache accounting (touched-but-reclaimable file pages counted toward this process's RSS), not by jask duplicating array data in memory. Confirmed the top allocators by size were `np.memmap.__new__` (redundant re-mapping of the same file, a virtual-address-space cost, not physical RAM) and `jax.device_put`'s dispatch machinery (real but small, spread across many block reads, not one big simultaneous allocation).

## Confirmed: real bigger-than-RAM scale now works end-to-end

Added `read_block`'s explicit copy-out (a numpy slice is a view, not a copy - correctness-critical once pages get advised away) plus `_release_pages()` calling `madvise(MADV_DONTNEED)` after every `read_block`/`write_block` (flushing dirty pages first, so nothing is ever discarded before it's persisted). Combined with the three other fixes above, ran the full sequence - `jask.sum`, `jax.grad(jask.sum)`, `jax.jit(jax.grad(jask.sum))` - on an 11.8GB array on a 14GB-total-RAM machine. All three completed successfully, gradient verified correct (all-ones, as expected), peak RSS never exceeded ~0.92GB (under 8% of the array's size) across the entire run. No crash, no reboot. This is the first clean end-to-end pass at this scale after three prior crashes (two of which required a full reboot) during earlier attempts without these fixes.

## Memory-bound regressions now covered by the automated test suite

Added `tests/test_memory_bounds.py` using `pytest-memray` (`@pytest.mark.limit_memory`) so the block-reader and budget fixes above are enforced automatically, not just verified once by hand. Note: memray's default accounting counts each `np.memmap()` call's full requested virtual mapping size as "allocated" - not physical RSS - so the limits are set generously enough to allow the correct, expected number of same-file re-mappings (traced precisely: ~2x per `jax.grad` call, ~2x more for `jit(grad(...))`), while still catching a real regression (the old K-sequence batch-load would add roughly another full array's worth of allocation in `batched_device_put` specifically, well past these limits). Run via `pytest --memray tests/test_memory_bounds.py`. `pytest-memray` and `optax` added to a `test` extra in `pyproject.toml`.

## Open question: system crashed once on a ~400MB test, unrelated to array size

One of the reboots during this session happened while a ~400MB memray test was running - far too small to plausibly exhaust this machine's RAM on its own. This doesn't match the risk profile of anything jask was doing at that moment, suggesting at least one of the several crashes this session may reflect broader system instability (or memray's own instrumentation overhead) rather than a jask memory bug. Not confirmed either way - flagged for awareness, not treated as evidence against the fixes above.

## Roadmap

- ~~Temp file cleanup~~ - **fixed.** Added `_own_fresh_file()` (`disk_array.py`), registering `weakref.finalize` only at genuine new-file construction sites: `from_numpy`, `vspace_zero`, and an op's fresh eager-mode output. Deliberately NOT applied to `raise_val`, deterministic `.grad` paths, `update_`'s stable slot, or a jit-branch's intermediate output object - those are re-wraps of a file something else still owns, and finalizing them would delete it out from under the real owner. Confirmed via a naive `__del__`-on-the-class experiment first: it crashed `jax.jit(jax.grad(...))` mid-run with `FileNotFoundError`, which is exactly why the distinction matters. Verified: file is deleted on GC, `jit(grad(...))` still works.
- **`CustomOp`'s contract in `run_backward` is inconsistent.** The `target_paths` fast-path added for `BlockParallelOp` doesn't apply to `CustomOp` - it just returns whatever `CustomOp.backward` produces, with no guarantee it lands at the deterministic `.grad` path. Needs reconciling before the first real `CustomOp` (FFT, softmax, sort, etc.) is added.
- **`io_callback` ordering relies entirely on data dependencies, not `ordered=True`** (confirmed: no call site passes it, default is `False`). Correct today because every real dependency is always threaded through the marker plumbing - but this is an invariant the design upholds, not something `io_callback` enforces on its own. If two ops ever touched the same underlying file without that dependency reflected in the markers, nothing would catch it. Deferred - not a live bug, just an assumption worth revisiting if the file-aliasing story ever changes.
- ~~`ScalarMul`'s multiplier can't be a jit-traced argument~~ - **fixed.** Rewrote `ScalarMul` as a hand-written primitive (`HiScalarMul`/`HiScalarMulBackward` in `linalg/mul.py`) instead of routing through `make_op`: the scalar is now a genuine traced input (`ShapedArray(())`), not a Python float baked in at construction time. Since the scalar is a real differentiable input, its own gradient is now computed too (`d(sum(s*a))/ds = sum(a)`) - a reduction across every block, structurally different from a normal per-block array gradient, computed in the same tiled backward pass as `a`'s gradient. Verified: works as a jit argument, as a closed-over literal (old pattern, unregressed), grad w.r.t. either operand or both, eager and under jit, either multiplication order. 6 new tests in `test_scalar_mul_traced.py`.

## `jask.dot` now supports batched N-D matmul (`@` semantics)

Generalized the existing `Dot` op from hardcoded 2D indexing to `[:-2]`/`[-2:]`-based slicing and `jnp.swapaxes(-1,-2)` instead of `.T` - no new op/architecture needed, since `@`/`jnp.matmul` already broadcasts correctly over leading batch dims on its own, and the tiled block loop just needed to stop assuming exactly rank 2. Batch dims must match exactly (no broadcasting yet). Wired `DiskArray.__matmul__` to it. Verified against `jnp.matmul` on a (5,3,4)@(5,4,6) batched case: forward, `@` operator, and both gradients all correct. Note: `jnp.dot`'s own literal N-D semantics (a different, rarer contraction rule - sums over a's last axis and b's second-to-last, not a batched op) is a separate, deferred task, not implemented here.

## Not working / known gaps

- Considered but deliberately deferred (real risk, low expected payoff relative to the fixes above): caching the underlying `np.memmap` object per file path to avoid re-mapping the same file for forward vs. backward. Real correctness traps (mode mismatches, staleness when a path gets reused for new data) not yet designed through - no longer needed given the confirmed pass above, but still a legitimate future optimization if profiling shows it matters.
- Second-order gradients - explicitly raises `NotImplementedError`, not silently wrong.
- `CustomOp` path (non-block-parallel ops) - no op uses it, never exercised under the current architecture.
- `vmap` / `pmap` / `shard_map` on `DiskArray` - unsupported by design, not a bug. jask is single-node only; scale to multiple devices via native JAX on a real `jax.Array` from `jask.materialize()`. vmap batching over `forward_block` was tried and found slower for jask's block sizes anyway.
- Training loops only tested to ~4 steps, not hundreds.
- No concurrency / multi-process testing of the file-based scheme.
