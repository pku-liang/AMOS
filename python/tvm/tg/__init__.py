
"""Namespace for Tensor Graph
"""

from .autodiff import gradient
from .autodiff import expr_equal, grad_op
from .graph import get_batch_like_dim, find_axis_in, count_operation, count_input_occur, \
                      subgraph_partition
