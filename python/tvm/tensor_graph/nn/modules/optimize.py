import tvm
import numpy as np
from tensor_graph.core import compute, GraphOp, GraphTensor


#################################################3
# Optimizers
class Optimizer(object):
  def __init__(self, lr, dtype="float32"):
    self.lr = lr
    self.dtype = dtype
    
    self.lr_tensor = tvm.te.placeholder([1], name="lr", requires_grad=False)

  def get_lr(self):
    """
    The newly updated lr
    """
    return np.array([self.lr]).astype(self.dtype)


class SGD(Optimizer):
  def __init__(self, lr, dtype="float32"):
    super(SGD, self).__init__(lr, dtype)

  def __call__(self, weight_tensors, gradient_tensors):
    """
    weight_tensors: (list of) NamedDimTensor or tvm Tensor
    gradient_tensors: (list of) NamedDimTensor or tvm Tensor

    returns:
    update_tensors: list of NamedDimTensor
    """
    if not isinstance(weight_tensors, (list, tuple)):
      weight_tensors = [weight_tensors]
    if not isinstance(gradient_tensors, (list, tuple)):
      gradient_tensors = [gradient_tensors]

    def func(*args):
      assert len(args) > 2
      return compute(
        args[:-2],
        lambda *indices: args[-2](*indices) - self.lr_tensor[0] * args[-1](*indices),
        # tag="update_dim" + str(len(args) - 2),
        requires_grad=True)

    update_tensors = []
    for (weight, gradient) in zip(weight_tensors, gradient_tensors):
      update_tensors.append(func(*weight.shape, weight, gradient))
    return update_tensors

class Adam(Optimizer):
  def __init__(self, lr, dtype="float32"):
    super(Adam, self).__init__(lr, dtype)
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.epsilon = 1e-8
    self.new_m_list = [] # in next_iter(), new will replace the old
    self.new_v_list = [] # In order to avoid use GraphOp inplace
    self.old_m_list = [] # list of GraphTensor Adam_m, initialized as 0
    self.old_v_list = [] # list of GraphTensor Adam_v, initialized as 0
    self.t = 1

  def __call__(self, weight_tensors, gradient_tensors):
    """
    weight_tensors: (list of) NamedDimTensor or tvm Tensor
    gradient_tensors: (list of) NamedDimTensor or tvm Tensor

    returns:
    update_tensors: list of NamedDimTensor
    """
    if not isinstance(weight_tensors, (list, tuple)):
      weight_tensors = [weight_tensors]
    if not isinstance(gradient_tensors, (list, tuple)):
      gradient_tensors = [gradient_tensors]
    for gradient in range(len(gradient_tensors)):
      self.old_m_list.append(GraphTensor(gradient.shape, requires_grad=False)) # All zero init
      self.old_v_list.append(GraphTensor(gradient.shape, requires_grad=False)) # All zero init
    
    def _inner_m(*args):
      shape_ = args[:-2]
      old_m = args[-2]
      gradient_ = args[-1]
      return compute(shape_, 
        lambda *indices: self.beta1 * old_m(*indices) + (1 - self.beta1) * gradient_(*indices),
        name="Adam_new_m",
        # tag="Adam_new_m",
        requires_grad=False)
    
    def _inner_v(*args):
      shape_ = args[:-2]
      old_v = args[-2]
      gradient_ = args[-1]
      return compute(shape_,
        lambda *indices: self.beta2 * old_v(*indices) + (1 - self.beta2) * gradient_(*indices) * gradient_(*indices),
        name="Adam_new_v",
        # tag="Adam_new_v",
        requires_grad=False)    
    
    def adam_func(idx, *args):
      assert len(args) > 2
      shape = args[:-2]
      weight_tensor = args[-2]
      gradient_Tensor = args[-1]

      new_lr = self.lr * (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t)
      assert len(self.new_m_list) == idx
      assert len(self.new_v_list) == idx
      
      new_m = GraphOp(shape, [], [self.old_m_list[idx], gradient_Tensor], _inner_m, name="Adam_new_m", requires_grad=False)
      new_v = GraphOp(shape, [], [self.old_v_list[idx], gradient_Tensor], _inner_v, name="Adam_new_v", requires_grad=False)

      self.new_m_list.append(new_m)
      self.new_v_list.append(new_v)

      return compute(
        shape,
        lambda *indices: weight_tensor(*indices) - new_lr * new_m(*indices) / (tvm.te.sqrt(new_v(*indices)) + self.epsilon),
        name="Adam_weight_update",
        # tag="Adam_weight_update",
        requires_grad=False)

    update_tensors = []
    idx = 0
    # This idx is used to index gradient_tensors
    for (weight, gradient) in zip(weight_tensors, gradient_tensors):
      update_tensors.append(adam_func(idx, *weight.shape, weight, gradient))
      idx += 1
    return update_tensors
  
  def next_iter(self):
    self.t += 1
    assert len(self.new_m_list) == len(self.new_v_list)
    assert len(self.old_m_list) == len(self.old_v_list)
    assert len(self.new_m_list) == len(self.old_m_list)
    # Save the newly computed m and v for next iteration
    for i in range(len(self.new_m_list)):
      self.old_m_list[i] = self.new_m_list[i]
      self.old_v_list[i] = self.new_v_list[i]
    self.new_m_list = []
    self.new_v_list = []

