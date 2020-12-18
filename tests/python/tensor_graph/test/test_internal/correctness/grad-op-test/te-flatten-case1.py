from tvm import testing
import tvm
import numpy as np

H0 = 3
H = 8
W = 16

dtype = "float32"

A = tvm.te.placeholder([H0, H, W], dtype=dtype, name="A")
C = tvm.te.compute([H0, H * W],
  lambda i, j:
    A[i, j//W, j%W], name="C")

dC = tvm.te.placeholder([H0, H * W], dtype=dtype, name="dC")

dA = tvm.tg.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))

func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [H0, H, W]).astype("float32")
dC_np = np.random.uniform(-10, 10, [H0, H * W]).astype("float32")
dA_np = np.zeros([H0, H, W]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

print(dA_tvm)

# =======>
# compare the results with numpy
golden_np = np.reshape(dC_np, (H0, H, W))
testing.assert_allclose(dA_tvm.asnumpy(), golden_np, rtol=1e-30, atol=1e-30)
print("Compare with Numpy success!")