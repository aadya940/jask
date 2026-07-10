from .base import set_memory_budget, Policy
from .base.disk_array import DiskArray

# Public API - each op is auto-registered as a hijax primitive via make_op.
from .linalg.matmul import dot
from .linalg.sum import sum
from .linalg.sub import sub
from .linalg.add import add
from .linalg.mul import mul
from .linalg.square import square
from .linalg.transpose import transpose
from .linalg.materialize import hi_materialize as materialize
