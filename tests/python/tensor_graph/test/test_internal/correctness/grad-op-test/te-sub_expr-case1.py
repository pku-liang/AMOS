from tvm import testing
import tvm
import numpy as np
import torch

H = 8
W = 9
Ashape = [H, W]
Cshape = [H, W]
dtype = "float32"

A = tvm.te.placeholder(Ashape, dtype=dtype, name="A")

C = tvm.te.compute(Cshape,
  lambda h, w:
    A[h, w] * 4 - A[h, w] * A[h, w], name="C")

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
C_torch = A_torch * 4 - A_torch * A_torch
loss = C_torch.sum()
loss.backward()
testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
print("Compare with Numpy success!")