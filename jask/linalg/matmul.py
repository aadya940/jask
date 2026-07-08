from ..base import BlockParallelOp, make_jax_op, get_default_policy, derive_page_shape

# Cache built (op, jax_op) pairs by the config that actually determines the
# compiled function's shape, without this, dot() rebuilds Dot and re-JITs
# via make_jax_op on every single call, forcing full retracing even for
# repeated calls with identical shapes (e.g. every step of a training loop).
# Keyed on policy's *stable* fields, not the whole Policy object - its
# resident_pages field mutates at runtime and isn't part of the compiled
# function's identity.
_DOT_OP_CACHE: dict[tuple, tuple] = {}


class Dot(BlockParallelOp):
    """C = A @ B, tiled over the shared (contraction) dimension."""

    def __init__(self, k_blocks: int):
        self.k_blocks = k_blocks

    def forward_block(self, a_block, b_block):
        return a_block @ b_block

    def index_map(self, out_idx):
        i, j = out_idx
        return [((i, k), (k, j)) for k in range(self.k_blocks)]

    def combine(self, acc, partial):
        return acc + partial

    def backward_block(self, d_out_block, a_block, b_block):
        d_a = d_out_block @ b_block.T
        d_b = a_block.T @ d_out_block
        return (d_a, d_b)

    def output_shape(self, a_shape, b_shape):
        return (a_shape[0], b_shape[1])


def dot(a, b):
    """Disk-backed matmul: C = A @ B.

    a, b: DiskArray handles. Uses the process-wide default memory budget
    (set via jask.set_memory_budget) and derives its own tiling, no
    engine configuration needed at the call site.
    """
    policy = get_default_policy()
    # Output block (i,j) must line up with A's row-blocks and B's col-blocks,
    # since index_map addresses inputs by those same (i,j) coordinates -
    # it can't be derived independently of a/b's own page_shape.
    page_shape = (a.page_shape[0], b.page_shape[1])

    k_blocks = -(
        -a.full_shape[1] // a.page_shape[1]
    )  # ceil div, matches DiskArray.block_grid

    cache_key = (
        k_blocks,
        page_shape,
        policy.max_memory,
        policy.page_size,
        policy.pages_per_group,
    )
    cached = _DOT_OP_CACHE.get(cache_key)
    if cached is None:
        op = Dot(k_blocks=k_blocks)
        jax_op = make_jax_op(op, policy, page_shape)
        _DOT_OP_CACHE[cache_key] = (op, jax_op)
    else:
        op, jax_op = cached

    return jax_op(a, b)
