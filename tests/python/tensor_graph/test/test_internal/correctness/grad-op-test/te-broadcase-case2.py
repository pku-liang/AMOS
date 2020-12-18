from tvm import testing
import tvm
import numpy as np


H = 8
W = 16
Ashape = [H, 1]
# Ashape = [H] -> pass
Cshape = [H, W]

dtype = "float32"

A = tvm.te.placeholder(Ashape, dtype=dtype, name="A")
Ashape = [H, 1]
C = tvm.te.compute(Cshape,
  lambda i, j:
    A[i, 0], name="C")

# # Ashape = [H]
# C = tvm.te.compute(Cshape,
#   lambda i, j:
#     A[i], name="C")

dC = tvm.te.placeholder(Cshape, dtype=dtype, name="dC")

dA = tvm.tg.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))

func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, Ashape).astype("float32")
dC_np = np.random.uniform(-10, 10, Cshape).astype("float32")
dA_np = np.zeros(Ashape).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

print(dA_tvm)

# =======>
# compare the results with numpy
golden_np = np.sum(dC_np, axis=1)
testing.assert_allclose(dA_tvm.asnumpy(), golden_np, rtol=1e-6, atol=1e-30)
print("Compare with Numpy success!")
