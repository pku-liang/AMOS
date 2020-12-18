from tvm import testing
import tvm
import numpy as np


L =  24
A1, B1 = 2, 12

dtype = "float32"

# 原来shape(N, M, K)下标[n, m, k], 希望输出
# [A1, B1] (A1*B1=N*M*K), 
# Out[i, j] = In[
# (i*B1 + j) // (M * K),
# (i*B1 + j) // K % M,
# (i*B1 + j) % K], 
A = tvm.te.placeholder([L], dtype=dtype, name="A")

# Failure (build passing, but comparison failure):

C = tvm.te.compute([A1, B1],
  lambda i, j:
    A[i*B1 + j], 
  name="C")


dC = tvm.te.placeholder([A1, B1], dtype=dtype, name="dC")

dA = tvm.tg.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))


func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, [L]).astype("float32")
dC_np = np.random.uniform(-10, 10, [A1, B1]).astype("float32")
dA_np = np.zeros([L]).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

#print(dA_tvm)

# =======>
# compare the results with numpy
golden_np = np.reshape(dC_np, (L,))
testing.assert_allclose(dA_tvm.asnumpy(), golden_np, atol=1e-30, rtol=1e-30)
print("Compare with Numpy success!")
