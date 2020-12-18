from tvm import testing
import tvm
import numpy as np
import torch


Ashape = [10, 20, 1152]
Cshape = [10, 20, 1152, 16]
dtype = "float32"

A = tvm.te.placeholder(Ashape, dtype=dtype, name="A")
B = tvm.te.placeholder(Cshape, dtype=dtype, name="B")

C = tvm.te.compute(Cshape,
  lambda i, j, k, n: B[i,j,k,n]+A[i,j,k], name="C")

dC = tvm.te.placeholder(Cshape, dtype=dtype, name="dC")

dA = tvm.tg.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, B, dC, dA], simple_mode=True))
func = tvm.build(s, [A, B, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, Ashape).astype("float32")
B_np = np.random.uniform(-10, 10, Cshape).astype("float32")
dC_np = np.ones(Cshape).astype("float32")
dA_np = np.zeros(Ashape).astype("float32")

ctx = tvm.context("llvm", 0)
A_tvm = tvm.nd.array(A_np, ctx)
B_tvm = tvm.nd.array(B_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dA_tvm = tvm.nd.array(dA_np, ctx)

func(A_tvm, B_tvm, dC_tvm, dA_tvm)

print(dA_tvm)

# =======>
# compare the results with Pytorch
A_torch = torch.tensor(A_np, requires_grad=True)
B_torch = torch.tensor(B_np, requires_grad=True)
C_torch = B_torch + A_torch.reshape(Ashape + [1]).expand_as(B_torch)
loss = C_torch.sum()
loss.backward()
testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
print("Compare with Numpy success!")