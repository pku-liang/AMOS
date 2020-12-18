import tvm
import time
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp
from tensor_graph.nn import MSELoss, SGD


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

  A = GraphTensor([H, L], name="A")
  B = GraphTensor([L, W], name="B")
  C = GraphOp([H, W], [L], [A, B], _gemm, name="C")
  bias = GraphTensor([H, W], name="bias")
  D = GraphOp([H, W], [], [C, bias], _add, name="D")
  E = GraphTensor([L, W], name="E")
  F = GraphOp([H, W], [L], [A, E], _gemm, name="F")
  G = GraphOp([H, W], [], [D, F], _add, name="G")
  label = GraphTensor([H, W], name="label")

  mse_loss = MSELoss(label)
  sgd = SGD(0.002)
  beg = time.time()
  fgraph = ForwardGraph([A], [G], [B, bias, E])
  bgraph = fgraph.make_backward(mse_loss, sgd)
  sch, bufs = bgraph.create_schedule()
  func = bgraph.build(sch, bufs, "llvm")
  # print(tvm.lower(sch, bufs, simple_mode=True))
  end = time.time()
  print("Cost %f ms" % ((end - beg) * 1e3), "to make graph")


if __name__ == "__main__":
  test1()