<h1 align="center">
Jask
</h1>

<p align="center">
<img src="logo.png" width="400px" height="400px" border="10px" />
</p>

Jask is a JAX-compatible library for operations on arrays too large to fit in RAM.
It provides out-of-core algorithms with a JAX-native API - `jax.jit`, `jax.grad`,
and `optax` work directly on disk-backed arrays with no wrappers.

This lets you run machine learning training/inference pipelines with weights or data
that don't fit in memory, using disk as extra space.

## Features

- **Disk-backed arrays** partitioned into blocks, streamed through JAX one tile at a time.
- **`jax.grad` support**: chain multiple disk-backed operations in a single loss function.
  Gradients are returned as disk-backed arrays, no placeholder handles.
- **Composable with `jax.jit`** - an op like `jask.dot(a, b)` behaves like `jnp.dot` from the outside.
- **`optax` integrations** - `optax.sgd`, `optax.adam`, and any other optimizer works transparently
  via hijax pytree registration.
- Easy to extend with new operations.

## Quick start

```python
import numpy as np
import jax
import optax
import jask

jask.set_memory_budget("4GB")

# Point at existing memmap files on disk - no data is loaded.
a = jask.DiskArray("weights_a.dat", shape=(50000, 10000), dtype=np.float32)
b = jask.DiskArray("weights_b.dat", shape=(10000, 5000), dtype=np.float32)
c = jask.DiskArray("weights_c.dat", shape=(5000, 100), dtype=np.float32)
target = jask.DiskArray("target.dat", shape=(50000, 100), dtype=np.float32)

# Chain disk-backed ops into a scalar loss and differentiate with jax.grad.
def mse_loss(a, b, c, target):
    z = jask.dot(jask.dot(a, b), c)
    diff = jask.sub(z, target)
    sq = jask.square(diff)
    return jask.sum(sq)  # returns a real scalar jax.Array

grad_a, grad_b, grad_c = jax.grad(mse_loss, argnums=(0, 1, 2))(a, b, c, target)

# Optax works out of the box - grads are DiskArrays.
opt = optax.sgd(0.01)
opt_state = opt.init(a)
updates, opt_state = opt.update(grad_a, opt_state)
new_a = a + updates   # use dunders instead of optax.apply_updates
```

For quick experiments with small arrays that fit in memory, `jask.DiskArray.from_numpy(arr)`
writes a numpy array to a temp file and wraps it. See `example.py`.

## Why not Dask?

Dask is an excellent library for distributed and out-of-core array computation.
Jask has a different goal: bringing out-of-core execution directly into the JAX
programming model.

Dask arrays don't participate in JAX transformations like `jax.grad` or `jax.jit`;
building a differentiable out-of-core pipeline in Dask requires custom gradient rules.

Jask registers `DiskArray` as a hijax type, so it behaves as a native JAX value: you can
compose ops, differentiate with `jax.grad`, and use optax without writing any custom
autodiff plumbing yourself.

## Currently supported ops

- `dot` - matrix multiplication
- `add` - elementwise addition
- `sub` - elementwise subtraction
- `mul` - elementwise or scalar multiplication
- `sum` - full reduction to a scalar
- `square` - elementwise square
- `transpose` - 2D transpose
- `materialize` - bridge a small `DiskArray` into an in-memory `jax.Array`

Open a PR if you want an op supported.
