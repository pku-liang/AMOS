from tvm import testing
from tvm from tvm import topi
import tvm
import numpy as np 
import torch

dim0 = 3
dim1 = 4
dim2 = 1

shape_size1 = [dim0, dim1, dim2]
shape_size2 = [dim0, dim1, dim2 * 2]
dtype = "float32"

cap0 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap0")
cap1 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap1")


cap_list = [cap0, cap1]

C = tvm.te.compute(shape_size2, 
    lambda i, j, k:
        tvm.te.if_then_else(
        #tvm.tir.Select(
            k == 0, 
                cap0[i, j, k],
                cap1[i, j, k - 1]),
            name="concat")

dC = tvm.te.placeholder(C.shape, dtype=dtype, name="dC")
dcap0, dcap1 = tvm.tg.gradient(C, cap_list, dC)

dcap_list = [dcap0, dcap1]

s = tvm.te.create_schedule([C.op, dcap0.op, dcap1.op, dC.op])

print(tvm.lower(s, cap_list + [C, dC] + dcap_list, simple_mode=True))

func = tvm.build(s,  cap_list + [C, dC] + dcap_list, target="llvm")

cap0_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
cap1_np = np.random.uniform(-10, 10, shape_size1).astype("float32")


dC_np = np.ones(shape_size2).astype("float32")
dcap0_np = np.zeros(shape_size1).astype("float32")
dcap1_np = np.zeros(shape_size1).astype("float32")

ctx = tvm.context("llvm", 0)
cap0_tvm = tvm.nd.array(cap0_np, ctx)
cap1_tvm = tvm.nd.array(cap1_np, ctx)

C_tvm = tvm.nd.array(dC_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dcap0_tvm = tvm.nd.array(dcap0_np, ctx)
dcap1_tvm = tvm.nd.array(dcap1_np, ctx)

func(cap0_tvm, cap1_tvm,
    C_tvm, dC_tvm,
    dcap0_tvm, dcap1_tvm)

print("dcap0_tvm", dcap0_tvm)

# =======>
# compare the results with pytorch
cap0_torch = torch.tensor(cap0_np, requires_grad=True)
cap1_torch = torch.tensor(cap1_np, requires_grad=True)
C_torch = torch.cat([cap0_torch, cap1_torch], dim=2)
loss = C_torch.sum()
loss.backward()
print("Pytorch gradient:\n cap0:", cap0_torch.grad.numpy(), "\ncap1:", cap1_torch.grad.numpy())
testing.assert_allclose(dcap0_tvm.asnumpy(), cap0_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
testing.assert_allclose(dcap1_tvm.asnumpy(), cap1_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
print("Compare with PyTorch success!")
