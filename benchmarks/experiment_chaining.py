"""Two toy tests exploring whether a small placeholder leaf inside a DiskArray-
like pytree can carry cross-op gradient signal through JAX's autodiff.

Test A: The current jask design. DiskArray has ONE real leaf (a scalar marker).
        Chain two custom_vjp ops. See whether the cotangent for the intermediate
        DiskArray reaches the first op's f_bwd, and if so, what it looks like.

Test B: The current jask design with a chain, checking whether the second op's
        computed input-gradient value survives back to the first op's f_bwd
        as usable data (or whether it collapses to zero/garbage).
"""

import jax
import jax.numpy as jnp
from dataclasses import dataclass, field

# A DiskArray-shaped pytree: metadata + one tiny "marker" leaf


@dataclass(frozen=True)
class Handle:
    filename: str
    marker: jax.Array = field(default_factory=lambda: jnp.zeros(()))


jax.tree_util.register_dataclass(
    Handle,
    data_fields=["marker"],  # the ONE real leaf JAX will differentiate through
    meta_fields=["filename"],  # everything else is static
)


# Simulate a disk-backed op via custom_vjp.
# In real jask, forward writes to a file; here we just print, and store the
# "gradient" in a global dict (side-effect equivalent of writing to disk).

grad_disk = {}  # filename -> the "gradient" that got written to disk


def _op_impl(name: str, x: Handle) -> Handle:
    return Handle(filename=f"out_{name}", marker=x.marker * 2.0)


_op = jax.custom_vjp(_op_impl, nondiff_argnums=(0,))


def _op_fwd(name: str, x: Handle):
    return _op(name, x), (x,)


def _op_bwd(name: str, residuals, g: Handle):
    (x,) = residuals
    print(f"[bwd of {name}] received cotangent for output - marker={g.marker}")
    computed_input_grad = g.marker * 3.14
    grad_disk[x.filename] = float(computed_input_grad)
    print(f"[bwd of {name}] wrote grad for {x.filename}: {computed_input_grad}")
    return (Handle(filename=x.filename, marker=computed_input_grad),)


_op.defvjp(_op_fwd, _op_bwd)


def op(x: Handle, name: str) -> Handle:
    return _op(name, x)


# The actual test: chain op1 -> op2, then jax.grad


def loss_fn(a: Handle) -> jax.Array:
    y = op(a, "op1")  # first disk-backed op
    z = op(y, "op2")  # second - feeds output of op1 as its input
    return z.marker  # something scalar for grad to work


a = Handle(filename="a", marker=jnp.array(1.0))

print("=" * 60)
print("Running jax.grad(loss_fn)(a) on a two-op chain:")
print("=" * 60)
grad = jax.grad(loss_fn)(a)
print()
print(f"Final gradient handle returned by jax.grad: {grad}")
print(f"Its marker value: {grad.marker}")
print()
print("Contents of grad_disk (what f_bwd side-effect wrote per op):")
for k, v in grad_disk.items():
    print(f"  {k}: {v}")
