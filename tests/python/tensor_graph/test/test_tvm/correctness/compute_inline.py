import tvm
import numpy as np


def test1():
  print("test 1 ###############################")
  A = tvm.te.placeholder([4, 4], name="A")
  B = tvm.te.placeholder([4, 4], name="B")
  k = tvm.te.reduce_axis([0, 4], name="k")
  C = tvm.te.compute([4, 4], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), name="C")

  D = tvm.te.compute([4, 4], lambda i, j: C[i, j] + 1, name="D")

  s = tvm.te.create_schedule(D.op)

  # s[C].compute_inline()

  print(tvm.lower(s, [A, B, D], simple_mode=True))

  func = tvm.build(s, [A, B, D], "llvm")


def test2():
  print("test 2 ###############################")
  dtype = "float32"
  A = tvm.te.placeholder([4, 4], name="A", dtype=dtype)
  B = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1, name="B")
  C = tvm.te.compute([4, 4], lambda i, j: B[i, j] * 2, name="C")

  s = tvm.te.create_schedule(C.op)
  # s[B].compute_inline()
  s[B].compute_at(s[C], s[C].op.axis[1])

  print(tvm.lower(s, [A, B, C], simple_mode=True))

  func = tvm.build(s, [A, B, C], "llvm")

  A_np = np.random.uniform(-1, 1, [4, 4]).astype(dtype)
  B_np = A_np + 1
  C_np = B_np * 2

  ctx = tvm.context("llvm", 0)
  A_tvm = tvm.nd.array(A_np, ctx)
  B_tvm = tvm.nd.empty([4, 4])
  C_tvm = tvm.nd.empty([4, 4])

  func(A_tvm, B_tvm, C_tvm)

  tvm.testing.assert_allclose(B_np, B_tvm.asnumpy(), atol=1e-2, rtol=1e-5)


if __name__ == "__main__":
  test1()
  test2()