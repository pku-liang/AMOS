import tvm 
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, ForwardGraph

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
  out_tensor, params = C(params)

  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)
  # tensors = [params[x].tvm_tensor for x in [A, B, bias, D]]
  # print(tvm.lower(s, tensors, simple_mode=True))
  # func = tvm.build(s, tensors, "llvm")

  batch_dim = tvm.tg.get_batch_like_dim(out_tensor.tvm_tensor)
  op = out_tensor.tvm_tensor.op
  for b in batch_dim:
    print(op.axis[b.value])


def test2():
  print("test 2 ########################")
  H = 32
  W = 16
  N = 8
  C = 12
  K = 6
  R = 3
  S = 3

  def _conv2d(N, K, H, W, C, R, S, A, B, requires_grad=True):
    rc = tvm.te.reduce_axis([0, C])
    rr = tvm.te.reduce_axis([0, R])
    rs = tvm.te.reduce_axis([0, S])
    return compute([N, K, H, W],
      lambda b, k, p, q: tvm.te.sum(A[b, rc, p+rr, q+rs] * B[k, rc, rr, rs], axis=[rc, rr, rs]),
      requires_grad=requires_grad)

  def _add(N, K, H, W, A, B, requires_grad=True):
    return compute([N, K, H, W], lambda b, k, h, w: A[b, k, h, w] + B[b, k], requires_grad=requires_grad)

  A = GraphTensor([N, C, H, W], name="A")
  B = GraphTensor([K, C, R, S], name="B")
  C = GraphOp([N, K, H, W], [C, R, S], [A, B], _conv2d)
  bias = GraphTensor([N, K], name="bias")
  D = GraphOp([N, K, H, W], [], [C, bias], _add)

  params = {}
  out_tensor, params = C(params)

  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)
  # tensors = [params[x].tvm_tensor for x in [A, B, bias, D]]
  # print(tvm.lower(s, tensors, simple_mode=True))
  # func = tvm.build(s, tensors, "llvm")

  batch_dim = tvm.tg.get_batch_like_dim(out_tensor.tvm_tensor)
  op = out_tensor.tvm_tensor.op
  for b in batch_dim:
    print(op.axis[b.value])


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
  B = GraphTensor([L, W], name="B")
  C = GraphOp([H, W], [L], [A, B], _gemm)
  bias = GraphTensor([H, W], name="bias")
  D = GraphOp([H, W], [], [C, bias], _add)

  params = {}
  out_tensor, params = C(params)

  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)
  # tensors = [params[x].tvm_tensor for x in [A, B, bias, D]]
  # print(tvm.lower(s, tensors, simple_mode=True))
  # func = tvm.build(s, tensors, "llvm")

  batch_dim = tvm.tg.get_batch_like_dim(out_tensor.tvm_tensor)
  op = out_tensor.tvm_tensor.op
  batch_axis = [op.axis[x.value] for x in batch_dim]
  tensor_A = params[A].tvm_tensor
  axis_in = tvm.tg.find_axis_in(batch_axis, tensor_A, out_tensor.tvm_tensor)
  print("batch like dim in A:", axis_in)
  tensor_B = params[B].tvm_tensor
  axis_in = tvm.tg.find_axis_in(batch_axis, tensor_B, out_tensor.tvm_tensor)
  print("batch like dim in B:", axis_in)


def test4():
  print("test 4 ########################")
  H = 32
  W = 16
  N = 8
  C = 12
  K = 6
  R = 3
  S = 3

  def _conv2d(N, K, H, W, C, R, S, A, B, requires_grad=True):
    rc = tvm.te.reduce_axis([0, C])
    rr = tvm.te.reduce_axis([0, R])
    rs = tvm.te.reduce_axis([0, S])
    return compute([N, K, H, W],
      lambda b, k, p, q: tvm.te.sum(A[b, rc, p+rr, q+rs] * B[k, rc, rr, rs], axis=[rc, rr, rs]),
      requires_grad=requires_grad)

  def _add(N, K, H, W, A, B, requires_grad=True):
    return compute([N, K, H, W], lambda b, k, h, w: A[b, k, h, w] + B[b, k], requires_grad=requires_grad)

  A = GraphTensor([N, C, H, W], name="A")
  B = GraphTensor([K, C, R, S], name="B")
  C = GraphOp([N, K, H, W], [C, R, S], [A, B], _conv2d)
  bias = GraphTensor([N, K], name="bias")
  D = GraphOp([N, K, H, W], [], [C, bias], _add)

  params = {}
  out_tensor, params = C(params)

  s = tvm.te.create_schedule(out_tensor.tvm_tensor.op)
  # tensors = [params[x].tvm_tensor for x in [A, B, bias, D]]
  # print(tvm.lower(s, tensors, simple_mode=True))
  # func = tvm.build(s, tensors, "llvm")

  batch_dim = tvm.tg.get_batch_like_dim(out_tensor.tvm_tensor)
  op = out_tensor.tvm_tensor.op
  batch_axis = [op.axis[x.value] for x in batch_dim]
  tensor_A = params[A].tvm_tensor
  axis_in = tvm.tg.find_axis_in(batch_axis, tensor_A, out_tensor.tvm_tensor)
  print("batch like dim in A:", axis_in)
  tensor_B = params[B].tvm_tensor
  axis_in = tvm.tg.find_axis_in(batch_axis, tensor_B, out_tensor.tvm_tensor)
  print("batch like dim in B:", axis_in)


if __name__ == "__main__":
  test1()
  test2()
  test3()
  test4()
  print("Success!")