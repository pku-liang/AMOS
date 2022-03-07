# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _opt-conv-tensorcore:
"""

################################################################
# TensorCore Introduction
# -----------------------
import tvm
from tvm import te
import numpy as np
from tvm.contrib import nvcc
from tvm import auto_tensorize as at
from functools import reduce

# The sizes of inputs and filters
batch_size = 256
height = 14
width = 14
in_channels = 256
out_channels = 512
kernel_h = 3
kernel_w = 3
pad_h = 1
pad_w = 1
stride_h = 1
stride_w = 1

# TensorCore shape
block_size = 16

assert batch_size % block_size == 0
assert in_channels % block_size == 0
assert out_channels % block_size == 0

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size // block_size,
    height,
    width,
    in_channels // block_size,
    block_size,
    block_size,
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    kernel_h,
    kernel_w,
    in_channels // block_size,
    out_channels // block_size,
    block_size,
    block_size,
)
# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size // block_size,
    height,
    width,
    out_channels // block_size,
    block_size,
    block_size,
)

# Reduction axes
kh = te.reduce_axis((0, kernel_h), name="kh")
kw = te.reduce_axis((0, kernel_w), name="kw")
ic = te.reduce_axis((0, in_channels // block_size), name="ic")
ii = te.reduce_axis((0, block_size), name="ii")

# Algorithm
A = te.placeholder(data_shape, name="A", dtype="float16")
W = te.placeholder(kernel_shape, name="W", dtype="float16")
Apad = te.compute(
    (
        batch_size // block_size,
        height + 2 * pad_h,
        width + 2 * pad_w,
        in_channels // block_size,
        block_size,
        block_size,
    ),
    lambda n, h, w, i, nn, ii: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height,
                    w >= pad_w, w - pad_w < width),
        A[n, h - pad_h, w - pad_w, i, nn, ii],
        tvm.tir.const(0.0, "float16"),
    ),
    name="Apad",
)
Conv = te.compute(
    output_shape,
    lambda n, h, w, o, nn, oo: te.sum(
        (Apad[n, h * stride_h + kh, w * stride_w +
             kw, ic, nn, ii]
        * W[kh, kw, ic, o, ii, oo]).astype("float32"),
        axis=[ic, kh, kw, ii],
    ),
    name="Conv",
)

###############################################################################
# Memory Scope
# ------------
#

hw_abs_dag = at.WMMAFp16Fp32()
compute_key = "nnn"
shape_key = "16x16x16"
input_names, output_names, nodes, read_graph, feed_graph = \
    at.construct_dag(
        hw_abs_dag, compute_key, shape_key, [Apad, W], [Conv])

output_tensors = reduce(
    lambda x, y: x + y, [nodes[x] for x in output_names], [])

s = tvm.te.create_schedule([x.op for x in output_tensors])
for cap in hw_abs_dag.hw_abs_dict.keys():
    if cap not in output_names:
        tensors = nodes[cap]
        for t in tensors:
            s[t].set_scope("local")

shared_tensors = []
for inp_name in input_names:
    inps = nodes[inp_name]
    assert len(inps) == 1
    readers = reduce(
        lambda x, y: x + y, [nodes[x] for x in feed_graph[inp_name]], [])
    SS = s.cache_read(inps[0], "shared", readers)
    shared_tensors.append(SS)

AS = shared_tensors[0]
WS = shared_tensors[1]
AF = nodes["load_a"][0]
WF = nodes["load_b"][0]
ConvF = nodes["mma"][0]
Conv = output_tensors[0]

s[Apad].compute_inline()

# Define tiling sizes
block_row_warps = 4
block_col_warps = 2
warp_row_tiles = 2
warp_col_tiles = 4
warp_size = 32
chunk = 2

block_x = te.thread_axis("blockIdx.x")
block_y = te.thread_axis("blockIdx.y")
block_z = te.thread_axis("blockIdx.z")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")
thread_z = te.thread_axis("threadIdx.z")

nc, hc, wc, oc, nnc, ooc = Conv.op.axis
block_k = s[Conv].fuse(hc, wc)
s[Conv].bind(block_k, block_z)
nc, nci = s[Conv].split(nc, factor=warp_row_tiles)
block_i, nc = s[Conv].split(nc, factor=block_row_warps)
oc, oci = s[Conv].split(oc, factor=warp_col_tiles)
block_j, oc = s[Conv].split(oc, factor=block_col_warps)
s[Conv].reorder(block_k, block_i, block_j, nc, oc, nci, oci, nnc, ooc)
s[Conv].bind(block_i, block_x)
s[Conv].bind(block_j, block_y)
s[Conv].bind(nc, thread_y)
s[Conv].bind(oc, thread_z)

# Schedule local computation
s[ConvF].compute_at(s[Conv], oc)
n, h, w, o, nnf, oof = ConvF.op.axis
ko, ki = s[ConvF].split(ic, factor=chunk)
s[ConvF].reorder(ko, kh, ki, kw, n, o, nnf, oof, ii)

# Move intermediate computation into each output compute tile
s[AF].compute_at(s[ConvF], kw)
s[WF].compute_at(s[ConvF], kw)

# Schedule for A's share memory
s[AS].compute_at(s[ConvF], kh)
n, h, w, i, nn, ii = AS.op.axis
tx, xo = s[AS].split(n, nparts=block_row_warps)
ty, yo = s[AS].split(xo, nparts=block_col_warps)
t = s[AS].fuse(nn, ii)
to, ti = s[AS].split(t, factor=warp_size)
s[AS].bind(tx, thread_y)
s[AS].bind(ty, thread_z)
s[AS].bind(ti, thread_x)

# Schedule for W's share memory
s[WS].compute_at(s[ConvF], kh)
kh, kw, ic, o, ii, oo = WS.op.axis
tx, xo = s[WS].split(o, nparts=block_row_warps)
ty, yo = s[WS].split(xo, nparts=block_col_warps)
t = s[WS].fuse(ii, oo)
to, ti = s[WS].split(t, nparts=warp_size)
s[WS].bind(tx, thread_y)
s[WS].bind(ty, thread_z)
s[WS].bind(to, thread_x)
s[WS].vectorize(ti)
# print(tvm.lower(s, [A, W, Conv], simple_mode=True))

load_a = hw_abs_dag.get_intrinsic(compute_key, shape_key, "load_a")
load_b = hw_abs_dag.get_intrinsic(compute_key, shape_key, "load_b")
store = hw_abs_dag.get_intrinsic(compute_key, shape_key, "store")
mma = hw_abs_dag.get_intrinsic(compute_key, shape_key, "mma")
print(load_a)
print(load_b)
print(store)
print(mma)

s[AF].tensorize(AF.op.axis[-2], load_a)
s[WF].tensorize(WF.op.axis[-2], load_b)
s[Conv].tensorize(nnc, store)
s[ConvF].tensorize(nnf, mma)
ir_module = tvm.lower(s, [A, W, Conv], simple_mode=True)
print("Lowered IRModule")
print(ir_module)
func = tvm.build(ir_module, target="cuda")
print("Source Code")
print(func.imported_modules[0].get_source())


ctx = tvm.gpu(0)
if nvcc.have_tensorcore(ctx.compute_version):
    with tvm.transform.PassContext(config={"tir.UnrollLoop":
                                              {"auto_max_step": 16}}):
        func = tvm.build(s, [A, W, Conv], "cuda")
    a_np = np.random.uniform(size=data_shape).astype(A.dtype)
    w_np = np.random.uniform(size=kernel_shape).astype(W.dtype)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(np.zeros(output_shape, dtype=Conv.dtype), ctx)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    print("conv2d with tensor core: %f ms" % (evaluator(a, w, c).mean * 1e3))
