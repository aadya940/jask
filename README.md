<h1 align="center">
Jask
</h1>

<p align="center">
<img src="logo.png" width="400px" height="400px" border="10px" />
</p>

Jask is a JAX-compatible library for operations on arrays too large to fit in RAM.
It provides out-of-core algorithms with a JAX-native API - `jax.jit`, `jax.grad`,
and `optax` work directly on disk-backed arrays, no wrappers or placeholder gradients.

This lets you run machine learning training or inference pipelines with weights or
data that don't fit in memory, using disk as extra space, while staying inside
ordinary JAX code.

## Features

- **Disk-backed arrays** (`DiskArray`), partitioned into blocks and streamed through
  JAX's tiled block loop one page at a time.
- **Never materializes a full array**, even under `jax.jit`. XLA's traced graph only
  ever carries a trivial marker; every real read/write happens as a side effect of
  jask's own tiled loop.
- **`jax.grad` support**: chain multiple disk-backed operations in a single loss
  function. Gradients come back as real `DiskArray`s, not placeholder handles.
- **`jax.jit` and `jax.grad` compose**, in either order, on the same op. A jitted
  training step that differentiates through disk-backed ops just works.
- **`optax` integration** - `optax.sgd`, `optax.adam`, and any other optimizer work
  transparently, since `DiskArray` registers as an atomic pytree leaf.
- **`DiskArray.update_()`** lets a training loop reuse one compiled `jax.jit`
  executable across every step, instead of retracing every call.
- Easy to extend: new ops are a handful of block-level methods, see below.

## Quick start

```python
import numpy as np
import jax
import optax
import jask

jask.set_memory_budget("4GB")

# Point at existing memmap files on disk - no data is loaded yet.
a = jask.DiskArray("weights_a.dat", shape=(50000, 10000), dtype=np.float32)
b = jask.DiskArray("weights_b.dat", shape=(10000, 5000), dtype=np.float32)
c = jask.DiskArray("weights_c.dat", shape=(5000, 100), dtype=np.float32)
target = jask.DiskArray("target.dat", shape=(50000, 100), dtype=np.float32)

# Chain disk-backed ops into a scalar loss and differentiate with jax.grad.
# jax.jit works here too, in either order: jax.jit(jax.grad(mse_loss)) or
# jax.grad(jax.jit(mse_loss)) both produce correct, disk-backed gradients.
def mse_loss(a, b, c, target):
    z = jask.dot(jask.dot(a, b), c)
    diff = jask.sub(z, target)
    sq = jask.square(diff)
    return jask.sum(sq)  # returns a real scalar jax.Array

grad_fn = jax.jit(jax.grad(mse_loss, argnums=(0, 1, 2)))
grad_a, grad_b, grad_c = grad_fn(a, b, c, target)

# optax works out of the box - gradients are DiskArrays.
opt = optax.sgd(0.01)
opt_state = opt.init(a)
updates, opt_state = opt.update(grad_a, opt_state)

# Use update_() (not `a = a + updates`) inside a jitted training loop: it
# overwrites a's own file in place, so the compiled step reuses one
# executable across every iteration instead of retracing each time.
a = a.update_(a + updates)
```

For quick experiments with small arrays that fit in memory, `jask.DiskArray.from_numpy(arr)`
writes a numpy array to a temp file and wraps it. See `example.py` for a full training loop.

## Scope: single machine, one process

Jask solves "my array doesn't fit in this machine's RAM." It does not solve "my
computation needs multiple machines or devices" - that's what JAX's own `pmap` and
`shard_map` are for, and they already work well on ordinary `jax.Array`s.

If you need both: use jask for the out-of-core part, then `jask.materialize(x)` to
bridge a `DiskArray` into a real `jax.Array` before handing it to your own
`pmap`/`shard_map` code. `DiskArray` itself never needs to cross that boundary.

## Why not Dask?

Dask is an excellent library for distributed and out-of-core array computation.
Jask has a narrower, different goal: bringing out-of-core execution directly into
the JAX programming model, on a single machine.

Dask arrays don't participate in JAX transformations like `jax.grad` or `jax.jit`;
building a differentiable out-of-core pipeline in Dask requires custom gradient rules.

Jask registers `DiskArray` as a hijax type, so it behaves as a native JAX value: you
compose ops, differentiate with `jax.grad`, jit the whole thing, and use optax
without writing any custom autodiff plumbing yourself.

## Currently supported ops

- `dot` - matrix multiplication
- `add` - elementwise addition
- `sub` - elementwise subtraction
- `mul` - elementwise or scalar multiplication
- `sum` - full reduction to a scalar
- `square` - elementwise square
- `transpose` - permute axes (2D transpose by default)
- `materialize` - bridge a `DiskArray` into an in-memory `jax.Array`

This is a minimal set of the ops needed to differentiate an MSE-loss pipeline, not
a complete linear algebra library yet - things like division, broadcasting,
reshape, axis-wise reductions, and activation functions aren't implemented. Open a
PR if you want an op supported; adding one only requires writing block-level math
(`forward_block`, `backward_block`, `index_map`, `combine`, `output_shape`) - see
any file in `jask/linalg/` for the pattern.

## Known limitations

See [log.md](log.md) for what's tested and working versus what isn't yet -
including a couple of real, specific gaps (a scalar multiplier passed as a jitted
argument, second-order gradients) rather than a vague disclaimer.
