from tvm import testing
import tvm
import numpy as np
import torch
# (A[i, j] / (A[i, j] + 1)) * B[i, j, k]是会报错的。
# 把上面这个表达式做mutation：
# A[i, j] + (A[i, j] + 1) * B[i, j, k] 能过。
# 另外，(A[i, j] / (A[i, j] + 1))单纯单变量也是不会错的

fail = True
only_forward = False
Ashape = [20, 32 * 6 * 6]
Bshape = [20, 32 * 6 * 6, 8]
Cshape = [20, 32 * 6 * 6, 8]
dtype = "float32"

A = tvm.te.placeholder(Ashape, dtype=dtype, name="A")

B = tvm.te.placeholder(Bshape, dtype=dtype, name="B")
if fail:
    C = tvm.te.compute(Cshape,
        lambda i, j, k: 
            (A[i, j] / (A[i, j] + 1)) * B[i, j, k], 
    name="C")
else:
    C = tvm.te.compute(Cshape,
        lambda i, j, k: 
            A[i, j] + (A[i, j] + 1) * B[i, j, k],
    name="C")

if only_forward:
    s = tvm.te.create_schedule(C.op)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    func = tvm.build(s, [A, B, C], target="llvm")
    A_np = np.random.uniform(-10, 10, Ashape).astype("float32")
    B_np = np.random.uniform(-10, 10, Bshape).astype("float32")
    C_np = np.zeros(Cshape).astype("float32")
    ctx = tvm.context("llvm", 0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)
    print("before", C_tvm)
    func(A_tvm, B_tvm, C_tvm)
    print("after", C_tvm)    
    exit(0)

dC = tvm.te.placeholder(Cshape, dtype=dtype, name="dC")

dA = tvm.tg.grad_op(A, C, dC)

s = tvm.te.create_schedule(dA.op)

print(tvm.lower(s, [A, B, dC, dA], simple_mode=True))
func = tvm.build(s, [A, B, dC, dA], target="llvm")

A_np = np.random.uniform(-10, 10, Ashape).astype("float32")
B_np = np.random.uniform(-10, 10, Bshape).astype("float32")
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
B_torch = torch.tensor(B_np, requires_grad=False)
if fail:
    scaling_A_torch = (A_torch / (A_torch + 1)).reshape([20, 32 * 6 * 6, 1]).expand_as(B_torch)
    C_torch = scaling_A_torch * B_torch
else:
    A_expand = A_torch.reshape([20, 32 * 6 * 6, 1]).expand_as(B_torch)
    C_torch = A_expand + (A_expand + 1) * B_torch

loss = C_torch.sum()
loss.backward()
testing.assert_allclose(dA_tvm.asnumpy(), A_torch.grad.numpy(), rtol=1e-4)
print("Compare with Numpy success!")