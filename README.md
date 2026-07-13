<h1 align="center">
Jask
</h1>

<p align="center">
<img src="logo.png" width="400px" height="400px" border="10px" />
</p>

<p align="center">
Extremely lightweight, disk-backed arrays that behave like native JAX values!
</p>



- Full JAX integration: `jax.jit`, `jax.grad`, and both composed, in either order.<br />
- Zero-materialization: XLA's traced graph never carries a full array, even under `jit`. <br />
- `optax` works transparently - `DiskArray` is a registered pytree leaf. <br />

Train on data or weights that don't fit in RAM, without leaving ordinary JAX code!

* ✅ Real `jax.grad` gradients, returned as actual `DiskArray`s - never placeholder handles
* ✅ `jax.jit` and `jax.grad` compose in either order on the same op
* ✅ Never materializes a full array in RAM, even under `jit` - verified with `memray`
* ✅ `optax.sgd`, `optax.adam`, and any other optimizer work out of the box
* ✅ `DiskArray.update_()` lets a jitted training loop compile once, not retrace every step
* ✅ Adding an op is a handful of block-level methods - no hijax/JAX-internals boilerplate
* ✅ Single machine, one process, by design - pair with native `pmap`/`shard_map` for multi-device

## Get started!

* Read [log.md](log.md) for what's tested and working versus known gaps
* Open a PR if you want an op supported - see [Currently supported ops](#currently-supported-ops)

## Table of contents

* [Installation](#installation)
* [Usage](#usage)
* [Example](#example)
* [Scope: single machine, one process](#scope-single-machine-one-process)
* [Why not Dask?](#why-not-dask)
* [Currently supported ops](#currently-supported-ops)
* [Known limitations](#known-limitations)

## Installation

Not yet published to PyPI - install from a local clone:

```
git clone <this-repo>
cd jask
pip install -e .
```

## Usage

There are three steps to a jask training/inference pipeline:

1. Wrap your data with `jask.DiskArray` - either pointing at an existing memmap file on disk, or via `jask.DiskArray.from_numpy(arr)` for quick experiments.
2. Compose ops (`jask.dot`, `jask.add`, `jask.sub`, `jask.mul`, `jask.square`, `jask.sum`, `jask.transpose`) into a loss function, exactly like you would with `jnp`.
3. Use `jax.grad`, `jax.jit`, and `optax` directly - no wrappers, no placeholder gradients, no custom autodiff plumbing.

## Example

```python
import numpy as np
import jax
import optax
import jask

jask.set_memory_budget("1GB")
np.random.seed(0)

A = np.random.rand(4, 6).astype(np.float32)
B = np.random.rand(6, 4).astype(np.float32)
C = np.random.rand(4, 3).astype(np.float32)
T = np.random.rand(4, 3).astype(np.float32)
a, b, c, target = (jask.DiskArray.from_numpy(x) for x in (A, B, C, T))

def mse_loss(a, b, c, target):
    z = jask.dot(jask.dot(a, b), c)
    diff = jask.sub(z, target)
    return jask.sum(jask.square(diff))

# jax.jit and jax.grad compose directly on disk-backed ops.
grad_fn = jax.jit(jax.grad(mse_loss, argnums=(0, 1, 2)))
opt = optax.sgd(0.001)
opt_state = opt.init(a)

for step in range(5):
    loss = float(mse_loss(a, b, c, target))
    print(f"step {step}: loss = {loss:.4f}")
    grad_a, grad_b, grad_c = grad_fn(a, b, c, target)
    updates, opt_state = opt.update(grad_a, opt_state)
    # update_() overwrites a's own file in place, so grad_fn stays compiled
    # once instead of retracing every step.
    a = a.update_(a + updates)
```

```
step 0: loss = 80.9999
step 1: loss = 77.2756
step 2: loss = 73.7257
step 3: loss = 70.3418
step 4: loss = 67.1164
```

See [example.py](example.py) for the full version, including a manual gradient-correctness check against numpy.

`set_memory_budget` also picks where jask's own scratch files (op outputs,
gradients) are written. By default that's `.jask_scratch` in the current
working directory. It refuses to use a RAM-backed filesystem (like `/tmp` on
many Linux systems, which is often tmpfs) as scratch space, since that would
silently defeat the out-of-core guarantee - pass `scratch_dir=` to pick a
real-disk location explicitly, or `allow_tmpfs=True` to opt in anyway:

```python
jask.set_memory_budget("4GB", scratch_dir="/data/jask_scratch")
```

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
