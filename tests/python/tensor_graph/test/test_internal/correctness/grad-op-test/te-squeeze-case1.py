from tvm import testing
import tvm
import numpy as np


H1 = 8
H2 = 3

dtype = "float32"

# squeeze这个算子去除掉大小为1的dim
A = tvm.te.placeholder([H1, 1, 1, H2], dtype=dtype, name="A")
C = tvm.te.compute([H1, H2],
  lambda i, j:
    A[i, 0, 0, j], name="C")

dC = tvm.te.placeholder([H1, H2], dtype=dtype, name="dC")

dA = tvm.tg.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))

func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [H1, 1, 1, H2]).astype("float32")
dC_np = np.random.uniform(-10, 10, [H1, H2]).astype("float32")
dA_np = np.zeros([H1, 1, 1, H2]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

print(dA_tvm)

# =======>
# compare the results with numpy
golden_np = np.reshape(dC_np, (H1, 1, 1, H2))
testing.assert_allclose(dA_tvm.asnumpy(), golden_np, atol=1e-30, rtol=1e-30)
print("Compare with Numpy success!")
