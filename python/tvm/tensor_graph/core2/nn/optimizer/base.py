import tvm
import numpy as np

from tvm.tensor_graph.core2.graph.concrete import FloatTensor, StateTensor


class Optimizer(object):
  def __init__(self):
    pass

  def step(self, weights, updates):
    raise NotImplementedError()

  @property
  def inputs(self):
    return []

  @property
  def inputs_data(self):
    return []

  @property
  def states(self):
    return []

  