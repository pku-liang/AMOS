import tvm

from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, ForwardGraph, \
                              GraphMutator
from tvm.tensor_graph.core.transform import LayoutChangeFinder, LayoutChangeApplier, apply_layout_change


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

  inputs = [A]
  weights = [B, bias, E]
  outputs = [G]

  fgraph = ForwardGraph(inputs, outputs, weights)
  
  new_graph = apply_layout_change(fgraph)

  A = new_graph.inputs[0]
  B, bias, E = new_graph.weights
  G = new_graph.outputs[0]

  out_tensor, params = G({})
  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)
  tensors = [params[x].tvm_tensor for x in [A, B, bias, E, G]]
  print(tvm.lower(s, tensors, simple_mode=True))
  func = tvm.build(s, tensors, "llvm")

  

if __name__ == "__main__":
  test1()