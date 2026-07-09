
<h1 align="center">
Jask
</h1>

<p align="center">
<img src="logo.png" width="400px" height="400px" />
</p>

Jask is a JAX compatible library for operations that are too large to fit in the RAM. It 
provides algorithms for Out of Core computation with a JAX like API with support for JIT,
`jax.grad` and `optax`, designed for deep integrations with a JAX machine learning pipeline.

## Features

- Support for **Disk-backed arrays** partitioned into blocks, streamed through JAX one tile at a time.
- **`jax.grad` support**, including chaining multiple disk-backed operations together in one loss function
    and integrations to work alongside other JAX functions.
- **Composable with `jax.jit`**, an op like `jask.dot(a, b)` behaves like `jnp.dot` from the outside.
- Integrations with `optax` for easy gradient based weight updates
- Easy to extend and add more Operations

## Quick start

```python
import numpy as np
import jax
import jask
from jask.base import DiskArray

jask.set_memory_budget("4GB")

# Wrap memmap'd arrays as DiskArrays (see example.py for the helper).
a: DiskArray = ...
b: DiskArray = ...
c: DiskArray = ...
target: DiskArray = ...

# Chain disk-backed ops into a normal scalar loss, then use jax.grad directly.
def mse_loss(a, b, c, target):
    z = jask.dot(jask.dot(a, b), c)
    diff = jask.sub(z, target)
    sq = jask.square(diff)
    return jask.materialize(jask.sum(sq))

grad_a, grad_b, grad_c = jax.grad(mse_loss, argnums=(0, 1, 2))(a, b, c, target)

# Gradients are DiskArrays too - materialize once you know they fit in memory.
dA = np.asarray(grad_a.grad.to_jax())
```

See `example.py` for a full runnable demo.

## Why not Dask?

Dask is an excellent library for distributed and out-of-core array computation. Jask has a different goal: bringing out-of-core execution directly into the JAX programming model.

While Dask can process arrays larger than memory, it is not a drop-in replacement for JAX arrays. Operations on Dask arrays do not automatically participate in JAX transformations such as jax.grad or jax.jit. Building differentiable out-of-core pipelines therefore requires additional integration work.

Jask is designed so that disk-backed arrays behave as JAX values for the operations it supports. You can compose multiple disk-backed operations, differentiate them with jax.grad, and integrate them into Optax optimization loops without writing custom gradient rules in your own code.



## Currently supported ops

- `dot` - matmul
- `sum` - full reduction to a scalar
- `sub` - elementwise subtraction
- `square` - elementwise square
- `materialize` - bridge from disk-backed to in-memory JAX computation

Open a PR if you would like an op supported.
