import tvm

from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, ForwardGraph, \
                              GraphMutator


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
  C = GraphOp([H, W], [L], [A, B], _gemm)
  bias = GraphTensor([H, W], name="bias")
  D = GraphOp([H, W], [], [C, bias], _add)

  params = {}
  out_tensor, params = D(params)

  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)
  tensors = [params[x].tvm_tensor for x in [A, B, bias, D]]
  print(tvm.lower(s, tensors, simple_mode=True))
  func = tvm.build(s, tensors, "llvm")


def test2():
  print("test 2 ########################")
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
  C = GraphOp([H, W], [L], [A, B], _gemm)
  bias = GraphTensor([H, W], name="bias")
  D = GraphOp([H, W], [], [C, bias], _add)
  E = GraphTensor([L, W], name="E")
  F = GraphOp([H, W], [L], [A, E], _gemm)
  G = GraphOp([H, W], [], [D, F], _add)

  params = {}
  out_tensor, params = G(params)

  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)

  tensors = [params[x].tvm_tensor for x in [A, B, bias, E, G]]
  print(tvm.lower(s, tensors, simple_mode=True))
  func = tvm.build(s, tensors, "llvm")


def test3():
  print("test 3 ########################")
  H = 32
  W = 16
  L = 8

  def _gemm(M, N, K, A, B, requires_grad=True):
    k = tvm.te.reduce_axis([0, K])
    return compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), requires_grad=requires_grad)

  def _add(M, N, A, B, requires_grad=True):
    return compute([M, N], lambda i, j: A[i, j] + B[i, j], requires_grad=requires_grad)

  A = GraphTensor([H, L], name="A")
  A.layout_transform = [1, 0]
  B = GraphTensor([L, W], name="B")
  C = GraphOp([H, W], [L], [A, B], _gemm)
  C.layout_transform = [1, 0]
  bias = GraphTensor([H, W], name="bias")
  D = GraphOp([H, W], [], [C, bias], _add)
  E = GraphTensor([L, W], name="E")
  E.layout_transform = [1, 0]
  F = GraphOp([H, W], [L], [A, E], _gemm)
  G = GraphOp([H, W], [], [D, F], _add)

  params = {}
  out_tensor, params = G(params)

  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)

  tensors = [params[x].tvm_tensor for x in [A, B, bias, E, G]]
  print(tvm.lower(s, tensors, simple_mode=True))
  func = tvm.build(s, tensors, "llvm")


def test4():
  print("test 4 ########################")
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
  C = GraphOp([H, W], [L], [A, B], _gemm)
  C.layout_transform = [1, 0]
  bias = GraphTensor([H, W], name="bias")
  D = GraphOp([H, W], [], [C, bias], _add)
  E = GraphTensor([L, W], name="E")
  F = GraphOp([H, W], [L], [A, E], _gemm)
  G = GraphOp([H, W], [], [D, F], _add)

  inputs = [A]
  weights = [B, bias, E]
  outputs = [G]

  fgraph = ForwardGraph(inputs, outputs, weights)
  print(isinstance(fgraph, ForwardGraph))
  new_graph = fgraph.make_new(inputs, outputs, weights)
  print(isinstance(fgraph, ForwardGraph))
  mutator = GraphMutator("up")
  new_graph = mutator(new_graph)
  mutator = GraphMutator("up")
  new_graph = mutator(new_graph)

  A = new_graph.inputs[0]
  B, bias, E = new_graph.weights
  G = new_graph.outputs[0]

  params = {}
  out_tensor, params = G(params)

  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)

  tensors = [params[x].tvm_tensor for x in [A, B, bias, E, G]]
  print(tvm.lower(s, tensors, simple_mode=True))
  func = tvm.build(s, tensors, "llvm")



if __name__ == "__main__":
  test1()
  test2()
  test3()
  test4()