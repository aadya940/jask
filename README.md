# jask

Out-of-core, differentiable array computation for JAX.
Your data is bigger than your GPU (or RAM); train on it anyway, using disk as extra memory, with correct gradients.

## What it is

- **Disk-backed arrays** partitioned into blocks, streamed through JAX one tile at a time.
- **`jax.grad` support**, including chaining multiple disk-backed ops together in one loss function.
- **Composable with `jax.jit`** - an op like `jask.dot(a, b)` behaves like `jnp.dot` from the outside.

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

## Currently supported ops

- `dot` - matmul
- `sum` - full reduction to a scalar
- `sub` - elementwise subtraction
- `square` - elementwise square
- `materialize` - bridge from disk-backed to in-memory JAX computation

Open a PR if you would like an op supported.
