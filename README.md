# jask

Out-of-core, differentiable array computation for JAX. 
Your data is bigger than your GPU (or RAM); train on it anyway, using disk as extra memory, with correct gradients.

## What it is

- **Disk-backed arrays** partitioned into blocks, streamed through JAX one tile at a time.
- **`jax.grad` support**.
- **Composable with `jax.jit`** an op like `jask.dot(a, b)` behaves like `jnp.dot` from the outside.

## Quick start

```python
import numpy as np
import jask
from jask.base import DiskArray

jask.set_memory_budget("4GB")

# Wrap two memmap'd arrays as DiskArrays (see example.py for the helper).
a: DiskArray = ...
b: DiskArray = ...

# Disk-backed matmul , never materializes the full inputs.
y = jask.dot(a, b)          # y is a DiskArray

# Materialize once you know the result fits in memory.
result = np.asarray(y.to_jax())
```

See `example.py` for a full runnable demo, including nesting under `@jax.jit` and computing gradients via `jax.vjp`.

## Currently Supported Ops

- dot/matmul
- Open a PR if you would like an Op supported.
