import tvm
import time
import numpy as np
from tvm.tensor_graph.core import ForwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tvm.tensor_graph.nn import MSELoss, SGD
from tvm.tensor_graph.core.schedule_generator import LayoutTransform
                                  
from tvm.tensor_graph.core.utils import flatten_tir_graph
from tvm.tensor_graph.core.space import ForwardGraphSpace
from tvm.tensor_graph.core.tuner import RandomForwardTuner


def test1():
  print("test 1 ########################")
  H = 32
  W = 16
  L = 8

  def _gemm(M, N, K, A, B, requires_grad=True):
    k = tvm.te.reduce_axis([0, K])
    return compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), requires_grad=requires_grad)

  def _add(M, N, A, B, requires_grad=True):
    return compute([M, N], lambda i, j: A[i, j] + B[i, j], requires_grad=requires_grad)

  def _add_const(k):
    def _inner(M, N, A, requires_grad=True):
      return compute([M, N], lambda i, j: A[i, j] + k, requires_grad=requires_grad)
    return _inner

  A = GraphTensor([H, L], name="A")
  B = GraphTensor([L, W], name="B")
  C = GraphOp([H, L], [], [A], _add_const(1), name="C")
  D = GraphOp([L, W], [], [B], _add_const(2), name="D")
  E = GraphOp([H, W], [L], [C, D], _gemm, name="E")
  F = GraphTensor([H, W], name="F")
  G = GraphOp([H, W], [], [E, F], _add, name="G")
  He = GraphOp([H, W], [], [G], _add_const(3), name="H")

  label = GraphTensor([H, W], name="label")

  mse_loss = MSELoss(label)
  sgd = SGD(0.002)
  fgraph = ForwardGraph([A], [G], [B, F])

  forward_space = ForwardGraphSpace()
  forward_tuner = RandomForwardTuner(forward_space)

  layout_generator = LayoutTransform(fgraph, forward_space, forward_tuner)

  beg = time.time()
  for i in range(10):
    fgraph = layout_generator.generate()

    bgraph = fgraph.make_backward(mse_loss, sgd)

    inputs = [x.tvm_tensor for x in bgraph.inputs]
    weights = [x.tvm_tensor for x in bgraph.weights]
    outputs = [x.tvm_tensor for x in bgraph.outputs]
    labels = [x.tvm_tensor for x in bgraph.labels]
    loss = bgraph.loss.tvm_tensor
    gradients = [x.tvm_tensor for x in bgraph.gradients]
    lr = bgraph.lr.tvm_tensor
    updates = [x.tvm_tensor for x in bgraph.updates]

    tgraph = PyTIRGraph(
      inputs,
      labels,
      outputs,
      weights,
      loss,
      gradients,
      lr,
      updates)

    op_list, _ = flatten_tir_graph([x.op for x in outputs + [loss] + gradients + updates])

    print("num ops =", len(op_list))

  end = time.time()
  print("average time=", (end - beg) / 100 * 1e3, "ms")
  print("Success!")


if __name__ == "__main__":
  test1()