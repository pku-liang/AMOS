from tvm import testing
import tvm
import numpy as np
import torch

# Input
u_shape = [10, 20, 1152, 16]
v_shape = [10, 20, 16]
# Output
a_shape = [10, 20, 1152]

dtype = "float32"

u_hat = tvm.te.placeholder(u_shape, dtype=dtype, name="u")
v_j = tvm.te.placeholder(v_shape, dtype=dtype, name="v")

r = tvm.te.reduce_axis([0, 16])
a_ij = tvm.te.compute(a_shape,
  lambda i, j, k: tvm.te.sum(u_hat[i,j,k,r] * v_j[i,j,r],axis=[r]), name="a")

da_ij = tvm.te.placeholder(a_shape, dtype=dtype, name="da")

du_hat = tvm.tg.grad_op(u_hat, a_ij, da_ij)

s = tvm.te.create_schedule(du_hat.op)

print(tvm.lower(s, [u_hat, v_j, da_ij, du_hat], simple_mode=True))


func = tvm.build(s, [u_hat, v_j, da_ij, du_hat], target="llvm")

u_np = np.random.uniform(-10, 10, u_shape).astype("float32")
v_np = np.random.uniform(-10, 10, v_shape).astype("float32")
da_np = np.ones(a_shape).astype("float32")
du_np = np.zeros(u_shape).astype("float32")

ctx = tvm.context("llvm", 0)
u_tvm = tvm.nd.array(u_np, ctx)
v_tvm = tvm.nd.array(v_np, ctx)
da_tvm = tvm.nd.array(da_np, ctx)
du_tvm = tvm.nd.array(du_np, ctx)

func(u_tvm, v_tvm, da_tvm, du_tvm)

print(du_tvm)

# =======>
# compare the results with Pytorch
u_torch = torch.tensor(u_np, requires_grad=True)
v_torch = torch.tensor(v_np, requires_grad=True)
a_torch = (u_torch * v_torch.reshape([10, 20, 1, 16])).sum(dim=-1, keepdim=False)
loss = a_torch.sum()
loss.backward()
testing.assert_allclose(du_tvm.asnumpy(), u_torch.grad.numpy(), atol=1e-30, rtol=1e-30)
print("Compare with Numpy success!")
