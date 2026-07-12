# Adding a new op

Most ops in jask are added by subclassing `BlockParallelOp` and writing
five small methods, each operating on one block ("tile") at a time. You
never touch `jax.jit`, `io_callback`, or hijax primitives directly - the
`make_op` factory wires all of that for you, including zero-materialization
under `jax.jit` and correct gradients under `jax.grad`.

## The five methods

```python
from jask.base import BlockParallelOp
from jask.base.base_algo import make_op


class Add(BlockParallelOp):
    """a + b elementwise."""

    def forward_block(self, a_block, b_block):
        return a_block + b_block

    def index_map(self, out_idx):
        return [(out_idx, out_idx)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        return (d_out_block, d_out_block)

    def output_shape(self, a_shape, b_shape):
        return a_shape


add = make_op(Add, doc="Elementwise sum of two disk-backed arrays.")
```

That's the actual, complete implementation of `jask.add`. Each method:

- **`forward_block(self, *input_blocks)`** - the real math, given one
  block from each input, entirely in memory. This is the only place you
  write the operation's forward computation.
- **`index_map(self, out_idx)`** - given one output tile's coordinates,
  which input block coordinates does it need? Returns a list of tuples,
  one per input, describing every contributing set of blocks. For a
  simple elementwise op, the output tile at `out_idx` needs exactly the
  input tiles at the *same* coordinates - one entry, `(out_idx, out_idx)`.
  For a reduction (see `jask.sum`), this can return every block in the
  input. For matrix multiplication (see `jask.dot`), this returns one
  entry per contraction-dimension block.
- **`combine(self, acc, partial)`** - folds one new partial result into
  the running accumulator for an output tile. Called once per entry
  `index_map` returned, not on a collected list - only the accumulator
  and one partial are ever resident at once.
- **`backward_block(self, d_out_block, *input_blocks)`** - the VJP:
  given the cotangent for one output block and the same input blocks
  `forward_block` saw, return a gradient for each input. This is what
  makes the op work under `jax.grad`.
- **`output_shape(self, *input_shapes)`** - the op's output shape, given
  the inputs' full shapes (not block shapes).

Adding an op means writing these five methods and calling `make_op`.
Nothing else - no jit compatibility code, no hijax primitives, no
`io_callback`. `make_op` handles the zero-materialization mechanism
(see [architecture notes in `base_algo.py`](../jask/base/base_algo.py))
uniformly for every op built this way.

## Ops with extra parameters

Some ops need more than array inputs - `jask.transpose` takes an
`axes=` keyword. Override `from_inputs` to build the op instance from
those extra keyword arguments:

```python
class Transpose(BlockParallelOp):
    def __init__(self, axes=None):
        self.axes = axes

    def forward_block(self, a_block):
        return jnp.transpose(a_block, self.axes)

    # ... index_map, combine, backward_block, output_shape ...

    @classmethod
    def from_inputs(cls, a, axes=None):
        return cls(axes=axes)


transpose = make_op(Transpose)
```

Any keyword arguments passed to the public function (e.g.
`jask.transpose(a, axes=(2, 0, 1))`) get forwarded to `from_inputs`.

## When `BlockParallelOp` isn't enough

Two situations need something other than a plain `BlockParallelOp`:

**A gradient that isn't shaped like its input.** `ScalarMul` (`a * scalar`,
where `scalar` is a real traced value, not a Python literal) needs a
gradient for `scalar` that's a *reduction* across every block
(`d(sum(scalar * a))/d(scalar) = sum(a)`), not a per-block array
gradient like every other op's backward pass. `make_op`'s contract
assumes every gradient output is spatial and tiled like its input, so
this case is hand-written as its own hijax primitive instead - see
`jask/linalg/mul.py`'s `HiScalarMul`/`HiScalarMulBackward` for the
pattern, if you need it.

**An op that can't be block-parallelized at all** (FFT, sort, and
similar). `CustomOp` exists for this - you implement `forward(self,
algo, *inputs)` and `backward(self, algo, inputs, d_out)` directly,
using `algo` (an `OOCAlgorithm`) as a toolkit for block I/O. As of this
writing no op in jask actually uses `CustomOp`, and its `run_backward`
contract doesn't yet guarantee gradients land at the deterministic
`.grad` path the way the `BlockParallelOp` path does (see the Roadmap
in `log.md`) - treat it as unfinished if you're the first to need it.

## Testing a new op

Correctness (forward value and both eager/jit gradients) is tested by
comparing against the equivalent plain `jnp` computation, in one shared,
parametrized table - not a hand-written test per op. Add one row to
`OPS` in `tests/test_ops.py`:

```python
pytest.param(
    (_mat((4, 6)), _mat((4, 6))),
    jask.your_new_op,
    lambda a, b: <the equivalent jnp expression>,
    id="your_new_op",
),
```

This gets you both a forward-correctness test and a gradient-correctness
test (checked against `jax.grad` on the plain-jnp version) for free -
see the top of `tests/test_ops.py` for the full pattern, including how
mixed scalar/array arguments are expressed.
