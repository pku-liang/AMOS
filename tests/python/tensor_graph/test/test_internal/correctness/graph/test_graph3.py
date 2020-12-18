import tvm
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
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
  fgraph = ForwardGraph([A], [G], [B, bias, E])
  bgraph = fgraph.make_backward(mse_loss, sgd)
  
  tgraph = PyTIRGraph(
    [x.tvm_tensor for x in bgraph.inputs],
    [x.tvm_tensor for x in bgraph.labels],
    [x.tvm_tensor for x in bgraph.outputs],
    [x.tvm_tensor for x in bgraph.weights],
    bgraph.loss.tvm_tensor,
    [x.tvm_tensor for x in bgraph.gradients],
    bgraph.lr.tvm_tensor,
    [x.tvm_tensor for x in bgraph.updates]
    )
  tgraph._analyze()



if __name__ == "__main__":
  test1()