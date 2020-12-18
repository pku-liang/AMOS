from tvm import testing
import tvm
import numpy as np


input_dim = (20, 32, 6, 6)
inter_dim = (20, 32, 6 * 6)
output_dim = (20, 32 * 6 * 6, 1)

dtype = "float32"

A = tvm.te.placeholder(input_dim, dtype=dtype, name="A")
B = tvm.te.compute(inter_dim,
  lambda i, j, k:
    A[i, j, k // 6, k % 6], name="B")
C = tvm.te.compute(output_dim,
    lambda i, j, k:
    B[i, j // 36, j % 36], name="C")

dC = tvm.te.placeholder(output_dim, dtype=dtype, name="dC")

# dA = tvm.tg.grad_op(A, C, dC)
dA, = tvm.tg.gradient(C, A, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))

func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, input_dim).astype("float32")
dC_np = np.random.uniform(-10, 10, output_dim).astype("float32")
dA_np = np.zeros(input_dim).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

print(dA_tvm)

# =======>
# compare the results with numpy
golden_np = np.reshape(dC_np, input_dim)
testing.assert_allclose(dA_tvm.asnumpy(), golden_np, rtol=1e-30, atol=1e-30)
print("Compare with Numpy success!")