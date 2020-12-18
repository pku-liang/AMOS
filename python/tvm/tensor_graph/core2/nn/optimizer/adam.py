import tvm
import numpy as np

from tensor_graph.core2.graph.abstract import TensorType
from tensor_graph.core2.graph.concrete import Tensor, FloatTensor, StateTensor, Compute
from .base import Optimizer



class Adam(Optimizer):
  def __init__(self, weights, alpha=0.01, momentum=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
    super(Adam, self).__init__()
    dtype = "float32"
    self.tvm_alpha = tvm.tir.const(alpha, dtype)
    self.tvm_momentum = tvm.tir.const(momentum, dtype)
    self.tvm_beta1 = tvm.tir.const(beta1, dtype)
    self.tvm_beta2 = tvm.tir.const(beta2, dtype)
    self.tvm_epsilon = tvm.tir.const(epsilon, dtype)
    self.m = []
    self.v = []
    target = "llvm"
    device = 0
    for w in weights:
      target = w.target
      device = w.device
      m = StateTensor(w.tensor_type, w.target, w.device, name="adam_m_"+w.name)
      v = StateTensor(w.tensor_type, w.target, w.device, name="adam_v_"+w.name)
      self.m.append(m)
      self.v.append(v)
    self.counter = Tensor(TensorType([1], "int32"), target, device, name="counter")
    self.counter_data = 0

  @property
  def inputs(self):
    return [self.counter]
  
  @property
  def inputs_data(self):
    self.counter_data += 1
    return [self.counter_data]

  @property
  def states(self):
    return self.m + self.v

  def step(self, weights, gradients):
    weights = list(weights)
    gradients = list(gradients)

    def _update_m(m, grad):
      def _for_spatial(*args):
        def _for_reduce():
          return (self.tvm_beta1 * m(*args) + (1 - self.tvm_beta1) * grad(*args))
        return _for_reduce, [], "none"
      return _for_spatial
    
    update_m = []
    for m, g in zip(self.m, gradients):
      update = Compute(m.shape, m.dtype, m, g, fhint=_update_m, name="adam_update_m_"+m.name)
      update_m.append(update)
    
    def _update_v(v, grad):
      def _for_spatial(*args):
        def _for_reduce():
          return (self.tvm_beta2 * v(*args) + (1 - self.tvm_beta2) * tvm.tir.power(grad(*args), 2))
        return _for_reduce, [], "none"
      return _for_spatial

    update_v = []
    for v, g in zip(self.v, gradients):
      update = Compute(v.shape, v.dtype, v, g, fhint=_update_v, name="adam_update_v_"+v.name)
      update_v.append(update)


    def _update(mt, vt, grad):
      def _for_spatial(*args):
        def _for_reduce():
          mtt = mt(*args) / (1 - tvm.tir.power(self.tvm_beta1, self.counter[0] + 1))
          vtt = vt(*args) / (1 - tvm.tir.power(self.tvm_beta2, self.counter[0] + 1))
          vtt_sqrt = tvm.tir.sqrt(vtt)
          return grad(*args) - self.tvm_alpha * mtt / (vtt_sqrt + self.tvm_epsilon)
        return _for_reduce, [], "none"
      return _for_spatial

    updates = []
    for mt, vt, gt in zip(update_m, update_v, gradients):
      update = Compute(mt.shape, mt.dtype, mt, vt, gt, fhint=_update, name="adam_update_grad_"+gt.name)
      updates.append(update)
    
    # remember to set states
    for m, mt in zip(self.m, update_m):
      m.set_update(mt)
    for v, vt in zip(self.v, update_v):
      v.set_update(vt)
    
    return updates

