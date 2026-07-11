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

## Not working / known gaps

- Second-order gradients - explicitly raises `NotImplementedError`, not silently wrong.
- Scalar multiplier passed as a jit-traced argument (`jax.jit(loss)(a, lr)`) fails - `ScalarMul` bakes its multiplier as a compile-time Python float. Works fine as a Python literal closed over in the function body (the pattern the whole test suite uses).
- `CustomOp` path (non-block-parallel ops) - no op uses it, never exercised under the current architecture.
- `vmap` / `pmap` / `shard_map` on `DiskArray` - unsupported by design, not a bug. jask is single-node only; scale to multiple devices via native JAX on a real `jax.Array` from `jask.materialize()`. vmap batching over `forward_block` was tried and found slower for jask's block sizes anyway.
- Never stress-tested at genuinely bigger-than-RAM scale - zero materialization is verified mechanically, not under real memory pressure.
- Training loops only tested to ~4 steps, not hundreds.
- No concurrency / multi-process testing of the file-based scheme.
