"""Forward and gradient correctness for every jask op.

One table, one forward test, one gradient test: each op is compared
against the equivalent plain-jnp computation (forward value, and
jax.grad's own result on that plain computation), instead of a hand-derived
formula per op. Adding an op means adding one row here, not a new test.
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest
import jask

from conftest import read


def _mat(shape):
    return lambda rng: rng.random(shape).astype(np.float32)


def _scalar(value):
    return lambda rng: value


# Each row: (make_args, jask_op, jnp_op). make_args entries are either an
# array-shape factory (-> DiskArray) or a plain scalar (left as-is) - mixed
# argument ops (e.g. scalar mul) are expressed the same way as array-only
# ops (e.g. add), just with a scalar factory in the tuple.
OPS = [
    pytest.param(
        (_mat((4, 6)), _mat((4, 6))),
        jask.add,
        lambda a, b: a + b,
        id="add",
    ),
    pytest.param(
        (_mat((4, 6)), _mat((4, 6))),
        jask.sub,
        lambda a, b: a - b,
        id="sub",
    ),
    pytest.param(
        (_mat((4, 6)), _mat((4, 6))),
        jask.mul,
        lambda a, b: a * b,
        id="mul_elementwise",
    ),
    pytest.param(
        (_mat((4, 6)), _scalar(3.5)),
        jask.mul,
        lambda a, s: a * s,
        id="mul_scalar_right",
    ),
    pytest.param(
        (_scalar(3.5), _mat((4, 6))),
        jask.mul,
        lambda s, a: s * a,
        id="mul_scalar_left",
    ),
    pytest.param(
        (_mat((4, 6)),),
        jask.square,
        lambda a: a**2,
        id="square",
    ),
    pytest.param(
        (_mat((4, 6)), _mat((6, 3))),
        jask.dot,
        jnp.dot,
        id="dot",
    ),
    pytest.param(
        (_mat((4, 6)),),
        jask.transpose,
        jnp.transpose,
        id="transpose_default",
    ),
    pytest.param(
        (_mat((3, 4, 5)),),
        lambda a: jask.transpose(a, axes=(2, 0, 1)),
        lambda a: jnp.transpose(a, (2, 0, 1)),
        id="transpose_axes",
    ),
    pytest.param(
        (_mat((4, 6)),),
        jask.sum,
        jnp.sum,
        id="sum",
    ),
    pytest.param(
        (_mat((4, 6)),),
        jask.materialize,
        lambda a: a,
        id="materialize",
    ),
]


def _make_args(rng, make_args):
    """Build (raw_args, jask_args) - arrays become DiskArrays, scalars pass through."""
    raw = [factory(rng) for factory in make_args]
    jask_args = [
        jask.DiskArray.from_numpy(x) if isinstance(x, np.ndarray) else x for x in raw
    ]
    array_positions = [i for i, x in enumerate(raw) if isinstance(x, np.ndarray)]
    return raw, jask_args, array_positions


@pytest.mark.parametrize("make_args,jask_op,jnp_op", OPS)
def test_forward_matches_jnp(rng, make_args, jask_op, jnp_op):
    """Every op's forward output matches the equivalent jnp computation."""
    raw, jask_args, _ = _make_args(rng, make_args)
    result = jask_op(*jask_args)
    expected = jnp_op(*raw)
    assert np.allclose(read(result), np.asarray(expected), atol=1e-3)


def _to_scalar(result):
    """Reduce an op's result to a scalar loss - jask.sum for a DiskArray
    result, jnp.sum for anything already a plain jax value (sum, materialize).
    Checked via jax.typeof (not isinstance): under jax.grad's tracing,
    `result` may still be a bare tracer, not a concrete DiskArray object."""
    from jask.base.disk_array import DiskArrayType

    is_disk_array = isinstance(jax.typeof(result), DiskArrayType)
    return jask.sum(result) if is_disk_array else jnp.sum(result)


@pytest.mark.parametrize("make_args,jask_op,jnp_op", OPS)
def test_grad_matches_jnp_autodiff(rng, make_args, jask_op, jnp_op):
    """Every op's gradient (via a scalar loss) matches jax.grad on the
    equivalent plain-jnp computation - no hand-derived formulas."""
    raw, jask_args, array_positions = _make_args(rng, make_args)
    if not array_positions:
        pytest.skip("no array arguments to differentiate")

    jask_grad_fn = jax.grad(
        lambda *args: _to_scalar(jask_op(*args)), argnums=tuple(array_positions)
    )
    jnp_grad_fn = jax.grad(
        lambda *args: jnp.sum(jnp_op(*args)), argnums=tuple(array_positions)
    )

    # argnums is always a tuple here, so jax.grad always returns a tuple,
    # even for a single array argument - no extra wrapping needed.
    jask_grads = jask_grad_fn(*jask_args)
    jnp_grads = jnp_grad_fn(*raw)

    for jg, ng in zip(jask_grads, jnp_grads):
        assert np.allclose(read(jg), np.asarray(ng), atol=1e-2)


def test_sum_output_is_real_jax_array_not_diskarray(small_2d):
    """sum(a) must return a real scalar jax.Array, not a DiskArray - it's
    small enough to materialize directly, and downstream code (a scalar
    loss) expects an ordinary jax value."""
    _, a = small_2d
    result = jask.sum(a)
    assert isinstance(result, jax.Array)
    assert result.shape == ()


def test_grad_returns_real_diskarray_not_placeholder(small_2d):
    """jax.grad must return a real DiskArray with correct shape/data, not a
    placeholder handle - the core hijax integration guarantee."""
    A, a = small_2d
    grad_a = jax.grad(jask.sum)(a)
    assert isinstance(grad_a, jask.DiskArray)
    assert grad_a.shape == A.shape
    assert np.allclose(read(grad_a), np.ones_like(A), atol=1e-4)
