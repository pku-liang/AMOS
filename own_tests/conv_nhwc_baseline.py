import numpy as np
import tvm
from tvm import te

batch = 256
in_channel = 256
out_channel = 512
in_size = 14
kernel = 3
pad = 1
stride = 1

# Algorithm
# A = te.placeholder((in_size, in_size, in_channel, batch), name='A', dtype="int8")
A = te.placeholder((batch, in_size, in_size, in_channel), name='A', dtype="int8")
# W = te.placeholder((kernel, kernel, in_channel, out_channel), name='W', dtype="int8")
W = te.placeholder((kernel, kernel, out_channel, in_channel), name='W', dtype="int8")
out_size = (in_size - kernel + 2*pad) // stride + 1
# Pad input
Apad = te.compute(
    # (in_size + 2*pad, in_size + 2*pad, in_channel, batch),
    (batch, in_size + 2*pad, in_size + 2*pad, in_channel),
    # lambda yy, xx, cc, nn: tvm.tir.if_then_else(
    lambda nn, yy, xx, cc: tvm.tir.if_then_else(
        tvm.tir.all(yy >= pad, yy - pad < in_size,
                xx >= pad, xx - pad < in_size),
        # A[yy - pad, xx - pad, cc, nn], tvm.tir.const(0, "int8")),
        A[nn, yy - pad, xx - pad, cc], tvm.tir.const(0, "int8")),
    name='Apad')
# Create reduction variables
rc = te.reduce_axis((0, in_channel), name='rc')
ry = te.reduce_axis((0, kernel), name='ry')
rx = te.reduce_axis((0, kernel), name='rx')
# Compute the convolution
B = te.compute(
    # (out_size, out_size, out_channel, batch),
    (batch, out_size, out_size, out_channel),
    # lambda yy, xx, ff, nn: te.sum(
    lambda nn, yy, xx, ff: te.sum(
        # Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
        Apad[nn, yy * stride + ry, xx * stride + rx, rc] * W[ry, rx, rc, ff],
        axis=[ry, rx, rc]),
    name='B')

s = te.create_schedule(B.op)
s[Apad].compute_inline() # compute Apad inline
AA = s.cache_read(Apad, 'shared', [B])
WW = s.cache_read(W, "shared", [B])
AL = s.cache_read(AA, "local", [B])
WL = s.cache_read(WW, "local", [B])
BL = s.cache_write(B, "local")

# tile consts
tile = 8
num_thread = 8
block_factor = tile * num_thread
step = 8
vthread = 2

# Get the GPU thread indices
block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
thread_xz = te.thread_axis((0, vthread), "vthread", name="vx")
thread_yz = te.thread_axis((0, vthread), "vthread", name="vy")

# Split the workloads
# hi, wi, fi, ni = s[B].op.axis
ni, hi, wi, fi = s[B].op.axis
bz = s[B].fuse(hi, wi)
by, fi = s[B].split(fi, factor=block_factor)
bx, ni = s[B].split(ni, factor=block_factor)

# Bind the iteration variables to GPU thread indices
s[B].bind(bz, block_z)
s[B].bind(by, block_y)
s[B].bind(bx, block_x)

tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
ty, fi = s[B].split(fi, nparts=num_thread)
tx, ni = s[B].split(ni, nparts=num_thread)
# s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
s[B].reorder(bz, by, bx, tyz, txz, ty, tx, ni, fi)

s[B].bind(tyz, thread_yz)
s[B].bind(txz, thread_xz)
s[B].bind(ty, thread_y)
s[B].bind(tx, thread_x)

# Schedule BL local write
s[BL].compute_at(s[B], tx)
# yi, xi, fi, ni = s[BL].op.axis
ni, yi, xi, fi = s[BL].op.axis
ry, rx, rc = s[BL].op.reduce_axis
rco, rci = s[BL].split(rc, factor=step)
# s[BL].reorder(rco, ry, rx, rci, fi, ni)
s[BL].reorder(ni, rco, ry, rx, rci, fi)

# Attach computation to iteration variables
s[AA].compute_at(s[BL], rx)
s[WW].compute_at(s[BL], rx)
s[AL].compute_at(s[BL], rci)
s[WL].compute_at(s[BL], rci)

# Schedule for A's shared memory load
# yi, xi, ci, ni = s[AA].op.axis
ni, yi, xi, ci = s[AA].op.axis
ty, ci = s[AA].split(ci, nparts=num_thread)
tx, ni = s[AA].split(ni, nparts=num_thread)
# _, ni = s[AA].split(ni, factor=4)
# s[AA].reorder(ty, tx, yi, xi, ci, ni)
s[AA].reorder(tx, ni, yi, xi, ty, ci)
s[AA].bind(ty, thread_y)
s[AA].bind(tx, thread_x)
# s[AA].vectorize(ni)  # vectorize memory load

# Schedule for W's shared memory load
# yi, xi, ci, fi = s[WW].op.axis
yi, xi, fi, ci  = s[WW].op.axis
ty, ci = s[WW].split(ci, nparts=num_thread)
tx, fi = s[WW].split(fi, nparts=num_thread)
# _, fi = s[WW].split(fi, factor=4)
# s[WW].reorder(ty, tx, yi, xi, ci, fi)
s[WW].reorder(ty, tx, yi, xi, fi, ci)
s[WW].bind(ty, thread_y)
s[WW].bind(tx, thread_x)
# s[WW].vectorize(fi)  # vectorize memory load


print(tvm.lower(s, [A, W, B]))

from tvm.te import schedule
import time
beg = time.time()
sch = s.normalize()
bounds = schedule.InferBound(sch)
end = time.time()
print((end - beg) * 1e3, "ms")
for iv in s[AA].op.axis:
    print(iv, bounds[iv].extent)
func = tvm.build(s, [A, W, B], 'cuda')
ctx = tvm.gpu(0)
a_np = np.random.randint(0, 10, size=(batch, in_size, in_size, in_channel)).astype("int8")
w_np = np.random.randint(0, 10, size=(kernel, kernel, out_channel, in_channel)).astype("int8")
a = tvm.nd.array(a_np, ctx)
w = tvm.nd.array(w_np, ctx)
b = tvm.nd.array(np.zeros((batch, out_size, out_size, out_channel), dtype=B.dtype), ctx)
func(a, w, b)
evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
print('Convolution without intrinstic: %f ms' % (evaluator(a, w, b).mean * 1e3))
