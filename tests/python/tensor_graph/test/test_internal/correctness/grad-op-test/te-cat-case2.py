from tvm import testing
from tvm from tvm import topi
import tvm
import numpy as np 
import torch

dim0 = 3
dim1 = 4
dim2 = 1

shape_size1 = [dim0, dim1, dim2]
shape_size2 = [dim0, dim1, dim2 * 8]
dtype = "float32"

cap0 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap0")
cap1 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap1")
cap2 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap2")
cap3 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap3")
cap4 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap4")
cap5 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap5")
cap6 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap6")
cap7 = tvm.te.placeholder(shape_size1, dtype=dtype, name="cap7")

cap_list = [cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7]

C = tvm.te.compute(shape_size2, 
    lambda i, j, k:
        # tvm.tir.Select(
        tvm.te.if_then_else(
            k == 0, cap0[i, j, k],
            tvm.te.if_then_else(k == 1, cap1[i, j, k-1],
                tvm.te.if_then_else(k == 2, cap2[i, j, k-2],
                    tvm.te.if_then_else(k == 3, cap3[i, j, k-3],
                        tvm.te.if_then_else(k == 4, cap4[i, j, k-4],
                            tvm.te.if_then_else(k == 5, cap5[i, j, k-5], 
                                tvm.te.if_then_else(k == 6, cap6[i, j, k-6],
                                    cap7[i, j, k-7]))))))),
                                    name="concat")

dC = tvm.te.placeholder(C.shape, dtype=dtype, name="dC")
dcap0, dcap1, dcap2, dcap3, dcap4, dcap5, dcap6, dcap7 = tvm.tg.gradient(C, cap_list, dC)

dcap_list = [dcap0, dcap1, dcap2, dcap3, dcap4, dcap5, dcap6, dcap7]

s = tvm.te.create_schedule([C.op, dcap0.op, dcap1.op, dcap2.op, dcap3.op, dcap4.op, 
        dcap5.op, dcap6.op, dcap7.op])

print(tvm.lower(s, cap_list + [C, dC] + dcap_list, simple_mode=True))

func = tvm.build(s,  cap_list + [C, dC] + dcap_list, target="llvm")

cap0_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
cap1_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
cap2_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
cap3_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
cap4_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
cap5_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
cap6_np = np.random.uniform(-10, 10, shape_size1).astype("float32")
cap7_np = np.random.uniform(-10, 10, shape_size1).astype("float32")


dC_np = np.ones(shape_size2).astype("float32")
dcap0_np = np.zeros(shape_size1).astype("float32")
dcap1_np = np.zeros(shape_size1).astype("float32")
dcap2_np = np.zeros(shape_size1).astype("float32")
dcap3_np = np.zeros(shape_size1).astype("float32")
dcap4_np = np.zeros(shape_size1).astype("float32")
dcap5_np = np.zeros(shape_size1).astype("float32")
dcap6_np = np.zeros(shape_size1).astype("float32")
dcap7_np = np.zeros(shape_size1).astype("float32")

ctx = tvm.context("llvm", 0)
cap0_tvm = tvm.nd.array(cap0_np, ctx)
cap1_tvm = tvm.nd.array(cap1_np, ctx)
cap2_tvm = tvm.nd.array(cap2_np, ctx)
cap3_tvm = tvm.nd.array(cap3_np, ctx)
cap4_tvm = tvm.nd.array(cap4_np, ctx)
cap5_tvm = tvm.nd.array(cap5_np, ctx)
cap6_tvm = tvm.nd.array(cap6_np, ctx)
cap7_tvm = tvm.nd.array(cap7_np, ctx)


C_np = np.zeros(shape_size2, dtype="float32")
C_tvm = tvm.nd.array(C_np, ctx)
dC_tvm = tvm.nd.array(dC_np, ctx)
dcap0_tvm = tvm.nd.array(dcap0_np, ctx)
dcap1_tvm = tvm.nd.array(dcap1_np, ctx)
dcap2_tvm = tvm.nd.array(dcap2_np, ctx)
dcap3_tvm = tvm.nd.array(dcap3_np, ctx)
dcap4_tvm = tvm.nd.array(dcap4_np, ctx)
dcap5_tvm = tvm.nd.array(dcap5_np, ctx)
dcap6_tvm = tvm.nd.array(dcap6_np, ctx)
dcap7_tvm = tvm.nd.array(dcap7_np, ctx)

func(cap0_tvm, cap1_tvm, cap2_tvm, cap3_tvm, cap4_tvm, cap5_tvm, cap6_tvm, cap7_tvm,
    C_tvm, dC_tvm,
    dcap0_tvm, dcap1_tvm, dcap2_tvm, dcap3_tvm, dcap4_tvm, dcap5_tvm, dcap6_tvm, dcap7_tvm)

print("dcap0_tvm", dcap0_tvm)

# =======>
# compare the results with pytorch
cap0_torch = torch.tensor(cap0_np, requires_grad=True)
cap1_torch = torch.tensor(cap1_np, requires_grad=True)
cap2_torch = torch.tensor(cap2_np, requires_grad=True)
cap3_torch = torch.tensor(cap3_np, requires_grad=True)
cap4_torch = torch.tensor(cap4_np, requires_grad=True)
cap5_torch = torch.tensor(cap5_np, requires_grad=True)
cap6_torch = torch.tensor(cap6_np, requires_grad=True)
cap7_torch = torch.tensor(cap7_np, requires_grad=True)

C_torch = torch.cat([cap0_torch, cap1_torch, cap2_torch, cap3_torch, cap4_torch,
            cap5_torch, cap6_torch, cap7_torch], dim=2)
loss = C_torch.sum()
loss.backward()
print("Pytorch gradient:\n cap0:", cap0_torch.grad.numpy(), "\ncap1:", cap1_torch.grad.numpy())
testing.assert_allclose(dcap0_tvm.asnumpy(), cap0_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
testing.assert_allclose(dcap1_tvm.asnumpy(), cap1_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
testing.assert_allclose(dcap2_tvm.asnumpy(), cap2_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
testing.assert_allclose(dcap3_tvm.asnumpy(), cap3_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
testing.assert_allclose(dcap4_tvm.asnumpy(), cap4_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
testing.assert_allclose(dcap5_tvm.asnumpy(), cap5_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
testing.assert_allclose(dcap6_tvm.asnumpy(), cap6_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
testing.assert_allclose(dcap7_tvm.asnumpy(), cap7_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
print("Compare with PyTorch success!")
