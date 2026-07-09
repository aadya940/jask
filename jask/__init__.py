from .base import set_memory_budget, Policy
from .base.disk_array import DiskArray as DiskArray

# Public API - hijax versions (jax.grad returns real gradients, no .grad hack).
from .linalg.matmul import hi_dot as dot
from .linalg.sum import hi_sum as sum
from .linalg.sub import hi_sub as sub
from .linalg.add import hi_add as add
from .linalg.mul import hi_mul as mul
from .linalg.square import hi_square as square
from .linalg.transpose import hi_transpose as transpose
from .linalg.materialize import hi_materialize as materialize
