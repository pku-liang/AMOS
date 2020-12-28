import tvm
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp


###################################################
# Loss
class Loss(object):
  def __init__(self, labels):
    if not isinstance(labels, (list, tuple)):
      labels = [labels]
    self.label_tensors = []
    for label in labels:
      assert isinstance(label, GraphTensor)
      label_tensor, _ = label({})
      self.label_tensors.append(label_tensor)

  def __call__(self, output_tensors):
    raise NotImplementedError()


class MSELoss(Loss):
  def __init__(self, labels):
    """
    labels: GraphTensor
    """
    super(MSELoss, self).__init__(labels)
    assert len(self.label_tensors) == 1
    assert len(self.label_tensors[0].shape) == 2, "MSE loss expects shape [batch, feature]"

  def __call__(self, output_tensors):
    """
    output_tensors: (list of) NamedDimTensor or tvm Tensor
    """
    if not isinstance(output_tensors, (list, tuple)):
      output_tensors = [output_tensors]
    assert len(self.label_tensors) == len(output_tensors)
    
    def _mse(batch, feature, A, B):
      rn = tvm.te.reduce_axis([0, batch], name="rn")
      rf = tvm.te.reduce_axis([0, feature], name="rf")
      return compute([1],
        lambda i: tvm.te.sum(tvm.tir.power(A[i+rn, rf] - B[i+rn, rf], 2) / feature, axis=[rn, rf]), name="mse")
    
    return _mse(*self.label_tensors[0].shape, self.label_tensors[0], output_tensors[0]), self.label_tensors
    
class CELoss(Loss):
  def __init__(self, labels):
    """
    labels: GraphTensor
    """
    super(CELoss, self).__init__(labels)
    assert len(self.label_tensors) == 1
    assert len(self.label_tensors[0].shape) == 2, "CE loss expects shape [batch, feature]"
  
  def __call__(self, output_tensors):
    """
    output_tensors: (list of) NamedDimTensor or tvm Tensor
    """
    if not isinstance(output_tensors, (list, tuple)):
      output_tensors = [output_tensors]
    assert len(self.label_tensors) == len(output_tensors)

    def _ce(batch, feature, label, out):
      N, C = out.shape
      N1, C1 = label.shape

      assert N.value == N1.value and C.value == C1.value
      c = tvm.te.reduce_axis([0, C], "c")
      k1 = tvm.te.reduce_axis([0, C], name="k1")
      # First compute the maximum for each batch
      max_val = compute([N], lambda n: tvm.te.max(out[n, k1], axis=[k1]), name="max_val", requires_grad=False)
      # Use the log_softmax trick to avoid overflow
      sum_val = compute([N], lambda i: tvm.te.sum(tvm.tir.exp(out[i, c] - max_val[i]), axis=[c]), "sum_val")
      rrn = tvm.te.reduce_axis([0, N], "rrn")
      rrc = tvm.te.reduce_axis([0, C], "rrc")
      return compute([1], lambda i: tvm.te.sum(
          label[i + rrn, rrc] * ((tvm.tir.log(sum_val[i + rrn]) + max_val[i + rrn]) - 
            out[i + rrn, rrc] * label[i + rrn, rrc]) / N,
          axis=[rrn, rrc]), name="cross_entropy")
    
    return _ce(*self.label_tensors[0].shape, self.label_tensors[0], output_tensors[0]), self.label_tensors


class MarginLoss(Loss):
  def __init__(self, labels):
    """
    labels: GraphTensor
    """
    super(MarginLoss, self).__init__(labels)
    assert len(self.label_tensors) == 1
    assert len(self.label_tensors[0].shape) == 2, "CE loss expects shape [batch, feature]"
  
  def __call__(self, output_tensors):
    """
    output_tensors: In Capsule Network, GraphNode [batch, num_classes, feature] 
    """
    # output_tensor [20, 10, 16]
    # label_tensors [20, 10]
    # print("output_tensors[0].shape", output_tensors[0].shape)
    # print("len:output_tensor", len(output_tensors))
    batch, num_classes, features = output_tensors[0].shape
    # v_c = torch.sqrt((x**2).sum(dim=2))
    # v_c torch.Size([20, 10])
    def _inner_vc(batch, num_classes, features, output_tensor):
      r = tvm.te.reduce_axis([0, features], "r")
      return compute([batch, num_classes],
              lambda i, j: tvm.te.sum(output_tensor[i, j, r], axis=[r]),
              name="v_c",
              tag="v_c")
    v_c = _inner_vc(batch, num_classes, features, output_tensors[0])
    # v_c = GraphOp([batch, num_classes], [features], [output_tensors[0]], _inner_vc, name="v_c")
    
    # left = F.relu(0.9 - v_c).view(batch_size, -1)
    # left torch.Size([20, 10])
    def _inner_left(batch, num_classes, v_c):
      return compute([batch, num_classes],
              lambda i, j: tvm.te.max(0.9 - v_c[i, j], tvm.tir.const(0, "float32")),
              name="left",
              tag="left")
    left = _inner_left(batch, num_classes, v_c)
    # left = GraphOp([batch, num_classes], [], [v_c], _inner_left, name="left")

    # right = F.relu(v_c - 0.1).view(batch_size, -1)
    # right torch.Size([20, 10])
    def _inner_right(batch, num_classes, v_c):
      return compute([batch, num_classes],
              lambda i, j: tvm.te.max(v_c[i, j] - 0.1, tvm.tir.const(0, "float32")),
              name="right",
              tag="right")
    right = _inner_right(batch, num_classes, v_c)
    # right = GraphOp([batch, num_classes], [], [v_c], _inner_right, name="right")

    # lables -> self.label_tensors[0]
    # sum the losses, with a lambda = 0.5
    # margin_loss = labels * left + 0.5 * (1. - labels) * right
    # margin_loss1 torch.Size([20, 10])
    def _inner_margin(batch, num_classes, labels, left, right):
      return compute([batch, num_classes],
              lambda i, j: labels[i, j] * left[i, j] + 0.5 * (1 - labels[i, j]) * right[i, j],
              name="margin",
              tag="margin")
    margin = _inner_margin(batch, num_classes, self.label_tensors[0], left, right)
    # margin = GraphOp([batch, num_classes], [], [self.label_tensors[0], left, right],
    #           _inner_margin, name="margin")
    
    def _inner_margin_loss(one, batch, num_classes, margin):
      t1 = tvm.te.reduce_axis([0, batch], name="t1")
      t2 = tvm.te.reduce_axis([0, num_classes], name="t2")
      return compute([1],
              lambda i: tvm.te.sum(margin[i+t1, t2], axis=[t1, t2]),
              name="margin_loss",
              tag="margin_loss")
    margin_loss = _inner_margin_loss(1, batch, num_classes, margin)
    # margin_loss = GraphOp([1], [batch, num_classes], [margin], _inner_margin_loss, 
    #         name="margin_loss")
    return margin_loss, self.label_tensors
    

class LSTMCELoss(Loss):
  def __init__(self, labels):
    """
    labels: GraphTensor
    """
    super(LSTMCELoss, self).__init__(labels)
    assert len(self.label_tensors) == 1
    assert len(self.label_tensors[0].shape) == 2, "CE loss expects shape [batch, feature]"
  
  def __call__(self, output_tensors):
    """
    output_tensors: (list of) NamedDimTensor or tvm Tensor
    """
    # result, new_h, new_c
    assert len(output_tensors) == 3
    assert isinstance(output_tensors, (list, tuple))

    def _ce(batch, feature, label, out):
      N, C = out.shape
      N1, C1 = label.shape

      assert N.value == N1.value and C.value == C1.value
      c = tvm.te.reduce_axis([0, C], "c")
      k1 = tvm.te.reduce_axis([0, C], name="k1")
      # First compute the maximum for each batch
      max_val = compute([N], lambda n: tvm.te.max(out[n, k1], axis=[k1]), name="max_val", requires_grad=False)
      # Use the log_softmax trick to avoid overflow
      sum_val = compute([N], lambda i: tvm.te.sum(tvm.tir.exp(out[i, c] - max_val[i]), axis=[c]), "sum_val")
      rrn = tvm.te.reduce_axis([0, N], "rrn")
      rrc = tvm.te.reduce_axis([0, C], "rrc")
      return compute([1], lambda i: tvm.te.sum(
          label[i + rrn, rrc] * ((tvm.tir.log(sum_val[i + rrn]) + max_val[i + rrn]) - 
            out[i + rrn, rrc] * label[i + rrn, rrc]) / N,
          axis=[rrn, rrc]), name="cross_entropy")
    
    return _ce(*self.label_tensors[0].shape, self.label_tensors[0], output_tensors[0]), self.label_tensors