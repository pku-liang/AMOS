from tvm import testing
import tvm
import numpy as np
import torch


Ashape = [20, 32 * 6 * 6, 8]
Cshape = [20, 32 * 6 * 6]
dtype = "float32"

A = tvm.te.placeholder(Ashape, dtype=dtype, name="A")

r=tvm.te.reduce_axis([0, 8])
C = tvm.te.compute(Cshape,
  lambda i, j: tvm.te.sum(tvm.tir.power(A[i,j,r], 2.0), axis=[r]),
  name="C")

dC = tvm.te.placeholder(Cshape, dtype=dtype, name="dC")

dA = tvm.tg.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, dC, dA], simple_mode=True))
func = tvm.build(s, [A, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, Ashape).astype("float32")
dC_np = np.ones(Cshape).astype("float32")
dA_np = np.zeros(Ashape).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, dC_tvm, dA_tvm)

print(dA_tvm)

# =======>
# compare the results with Pytorch
A_torch = torch.tensor(A_np, requires_grad=True)
C_torch = torch.sum(torch.pow(A_torch, 2), dim=2)
loss = C_torch.sum()
loss.backward()
testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), atol=1e-6, rtol=1e-6)
print("Compare with Numpy success!")