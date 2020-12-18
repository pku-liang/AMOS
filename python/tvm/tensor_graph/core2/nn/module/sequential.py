from tensor_graph.core2.graph.concrete import Tensor, FloatTensor, Compute
from tensor_graph.core2.nn import functional as F

from .base import StateTensor, Module


class Sequential(Module):
  def __init__(self, *args):
    super(Sequential, self).__init__()
    for i, arg in enumerate(args):
      setattr(self, "layer_" + str(i), arg)
    self.num_args = len(args)
  
  def forward(self, x):
    for i in range(self.num_args):
      x = getattr(self, "layer_" + str(i))(x)
    return x