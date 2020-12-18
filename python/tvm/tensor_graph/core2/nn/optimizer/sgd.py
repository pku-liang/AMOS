import tvm
import numpy as np

from tensor_graph.core2.graph.abstract import TensorType
from tensor_graph.core2.graph.concrete import Tensor, FloatTensor, StateTensor, Compute
from .base import Optimizer



class NaiveSGD(Optimizer):
  def __init__(self, weights, lr=0.01, decay=0.99):
    super(NaiveSGD, self).__init__()
    self.lr_tensor = FloatTensor([1])
    self.lr_data = lr
    self.decay = decay

  @property
  def inputs(self):
    return [self.lr_tensor]
  
  @property
  def inputs_data(self):
    self.lr_data = self.decay * self.lr_data
    return [self.lr_data]

  @property
  def states(self):
    return []

  def step(self, weights, gradients):
    weights = list(weights)
    gradients = list(gradients)

    updates = []

    def _update(w, g):
      def _for_spatial(*args):
        def _for_reduce():
          return w(*args) - self.lr_tensor[0] * g(*args)
        return _for_reduce, [], "none"
      return _for_spatial

    for w, g in zip(weights, gradients):
      updates.append(Compute(w.shape, w.dtype, w, g, fhint=_update, name="naive_sgd_update"))

    return updates

