from tensor_graph.core2.graph.concrete import Tensor, FloatTensor, Compute
from tensor_graph.core2.nn import functional as F

from .base import StateTensor, Module


class ReLU(Module):
    """ReLU of float32 precision
    """
    def __init__(self):
        super(ReLU, self).__init__()
  
    def forward(self, inputs):
        return F.ReLU(inputs, output_dtype="float32", requires_grad=inputs.requires_grad)