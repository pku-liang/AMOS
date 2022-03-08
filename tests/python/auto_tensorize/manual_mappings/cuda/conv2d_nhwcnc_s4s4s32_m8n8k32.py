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
from tvm import auto_tensorize

# The sizes of inputs and filters
batch_size = 128  # 256
height = 14
width = 14
in_channels = 64  # 256
out_channels = 32  # 512
kernel_h = 3
kernel_w = 3
pad_h = 1
pad_w = 1
stride_h = 1
stride_w = 1

alignment = 32

# TensorCore shape
wmma_m = 8
wmma_n = 8
wmma_k = 32

# data type
a_dtype = "int4"
b_dtype = "int4"
acc_dtype = "int32"
out_dtype = "int32"

assert batch_size % wmma_m == 0
assert in_channels % wmma_k == 0
assert out_channels % wmma_n == 0

# Input feature map: (N, H, W, IC, n, ic)
data_shape = (
    batch_size // wmma_m,
    height,
    width,
    in_channels // wmma_k,
    wmma_m,
    wmma_k,
)
# Kernel: (H, W, IC, OC, ic, oc)
kernel_shape = (
    kernel_h,
    kernel_w,
    in_channels // wmma_k,
    out_channels // wmma_n,
    wmma_k,
    wmma_n,
)
# Output feature map: (N, H, W, OC, n, oc)
output_shape = (
    batch_size // wmma_m,
    height,
    width,
    out_channels // wmma_n,
    wmma_m,
    wmma_n,
)

# Reduction axes
kh = te.reduce_axis((0, kernel_h), name="kh")
kw = te.reduce_axis((0, kernel_w), name="kw")
ic = te.reduce_axis((0, in_channels // wmma_k), name="ic")
ii = te.reduce_axis((0, wmma_k), name="ii")

# Algorithm
A = te.placeholder(data_shape, name="A", dtype=a_dtype)
W = te.placeholder(kernel_shape, name="W", dtype=b_dtype)
Apad = te.compute(
    (
        batch_size // wmma_m,
        height + 2 * pad_h,
        width + 2 * pad_w,
        in_channels // wmma_k,
        wmma_m,
        wmma_k,
    ),
    lambda n, h, w, i, nn, ii: tvm.tir.if_then_else(
        tvm.tir.all(h >= pad_h, h - pad_h < height, w >= pad_w, w - pad_w < width),
        A[n, h - pad_h, w - pad_w, i, nn, ii],
        tvm.tir.const(0.0, a_dtype),
    ),
    name="Apad",
)
Conv = te.compute(
    output_shape,
    lambda n, h, w, o, nn, oo: te.sum(
        Apad[n, h * stride_h + kh, w * stride_w + kw, ic, nn, ii].astype(acc_dtype)
        * W[kh, kw, ic, o, ii, oo].astype(acc_dtype),
        axis=[ic, kh, kw, ii],
    ),
    name="Conv",
)

s = te.create_schedule(Conv.op)
s[Apad].compute_inline()

###############################################################################
# Memory Scope
# ------------
# In traditional GPU schedule, we have global, shared and local memory scope.
# To support TensorCores, we add another three special
#  memory scope: :code:`local`,
# :code:`local` and :code:`local`. On hardware, all fragments scope
# stores at the on-chip registers level, the same place with local memory.

# Designate the memory hierarchy
AS = s.cache_read(Apad, "shared", [Conv])
WS = s.cache_read(W, "shared", [Conv])
AF = s.cache_read(AS, "local", [Conv])
WF = s.cache_read(WS, "local", [Conv])
ConvF = s.cache_write(Conv, "local")

###############################################################################
# Define Tensor Intrinsic
# -----------------------
# In fact, TensorCore is a special hardware operation.
# So, we can just use tensorize
# to replace a unit of computation with the TensorCore instruction.
# The first thing is
# that we need to define tensor intrinsic.
#
# There are four basic operation in TensorCore: :code:`fill_fragment`,
# :code:`load_matrix`,
# :code:`mma_sync` and :code:`store_matrix`. Since :code:`fill_fragment`
# and :code:`mma_sync`
# are both used in matrix multiplication, so we can just write following
# three intrinsics.


def intrin_wmma_load_matrix(scope, operand):
    if operand == "Src":
        frag_shape = (wmma_m, wmma_k)
        frag_dtype = a_dtype
        frag_layout = "nvcuda::wmma::row_major"
        frag_ldm = wmma_k
    elif operand == "Filter":
        frag_shape = (wmma_k, wmma_n)
        frag_dtype = b_dtype
        frag_layout = "nvcuda::wmma::col_major"
        frag_ldm = wmma_k
    else:
        raise ValueError(f"Invalid argument: operand = {operand}")

    offset_factor = frag_shape[0] * frag_shape[1]

    A = te.placeholder(frag_shape, name="A", dtype=frag_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=alignment, offset_factor=offset_factor
    )
    C = te.compute(frag_shape, lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=scope, data_alignment=alignment, offset_factor=offset_factor
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.amos_memory",
                "cuda",
                "wmma_int4_int32",
                "nvcuda::wmma::load_matrix_sync",
                BC.data,
                wmma_m,
                wmma_n,
                wmma_k,
                BC.elem_offset // offset_factor,
                BA.access_ptr("r"),
                frag_ldm,
                frag_layout,
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm():
    A = te.placeholder((wmma_m, wmma_k), name="A", dtype=a_dtype)
    B = te.placeholder((wmma_k, wmma_n), name="B", dtype=b_dtype)
    k = te.reduce_axis((0, wmma_k), name="k")
    C = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(A[ii, k].astype(acc_dtype) * B[k, jj].astype(acc_dtype), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name="BA",
        scope="local",
        data_alignment=alignment,
        offset_factor=wmma_m * wmma_k,
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name="BB",
        scope="local",
        data_alignment=alignment,
        offset_factor=wmma_k * wmma_n,
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="local",
        data_alignment=alignment,
        offset_factor=wmma_m * wmma_n,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.amos_compute",
                    "cuda",
                    "wmma_int4_int32",
                    "nvcuda::wmma::fill_fragment",
                    BC.data,
                    wmma_m,
                    wmma_n,
                    wmma_k,
                    BC.elem_offset // (wmma_m * wmma_n),
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.amos_compute",
                    "cuda",
                    "wmma_int4_int32",
                    "nvcuda::wmma::mma_sync",
                    BC.data,
                    BC.elem_offset // (wmma_m * wmma_n),
                    BA.data,
                    BA.elem_offset // (wmma_m * wmma_k),
                    BB.data,
                    BB.elem_offset // (wmma_k * wmma_n),
                    BC.data,
                    BC.elem_offset // (wmma_m * wmma_n),
                    False,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix():
    A = te.placeholder((wmma_m, wmma_n), name="A", dtype=out_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="local", data_alignment=alignment, offset_factor=(wmma_m * wmma_n)
    )
    C = te.compute((wmma_m, wmma_n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="global", data_alignment=alignment, offset_factor=(wmma_m * wmma_n)
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.amos_memory",
                "cuda",
                "wmma_int4_int32",
                "nvcuda::wmma::store_matrix_sync",
                BA.data,
                wmma_m,
                wmma_n,
                wmma_k,
                BA.elem_offset // (wmma_m * wmma_n),
                BC.access_ptr("w"),
                wmma_n,
                "nvcuda::wmma::mem_row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


###############################################################################
# Scheduling the Computation
# --------------------------
# To use TensorCores in TVM, we must schedule the computation into specific
#   structure
# to match the tensor intrinsic. The same as traditional GPU programs,
#   we can also use
# shared memory to boost the speed. If you have any questions about
#   blocking and shared
# memory, please refer :ref:`opt-conv-gpu`.
#
# In this example, each block contains 2x4 warps, and each warp calls 4x2
#   TensorCore
# instructions. Thus, the output shape of each warp is 64x32 and each block
#   outputs
# 128x128 titles. Due to the limit of shared memory space, we only load 2
#   blocks (2x128x128 tiles)
# one time.
#
# .. note::
#
#   *Warp-level Operation*
#
#   Note that all TensorCore instructions are warp-level instructions,
#   which means all 32 threads
#   in a warp should do this instruction simultaneously. Making theadIdx.x
#   extent=32 is one of the
#   easiest way to solve this. Then We can bind threadIdx.x to any loops except
#   those contain
#   TensorCore intrinsics directly or indirectly. Also note that it is not the
#   unique solution.
#   The only thing we should do is to make sure all threads in a warp can
#   call TensorCore at the same time.

# 256x512x256
# 32x32x32
# 8x8x32
# Define tiling sizes
# block_row_warps = 4
# block_col_warps = 2
block_row_warps = 4
block_col_warps = 2
# warp_row_tiles = 2
# warp_col_tiles = 4
warp_row_tiles = 8
warp_col_tiles = 16
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

###############################################################################
# Lowering Computation to Intrinsics
# ----------------------------------
# The last phase is to lower the computation loops down to TensorCore hardware
# intrinsics
# by mapping the 2D convolution to tensor intrinsics
s[AF].tensorize(AF.op.axis[-2], intrin_wmma_load_matrix("local", "Src"))
s[WF].tensorize(WF.op.axis[-2], intrin_wmma_load_matrix("local", "Filter"))
s[Conv].tensorize(nnc, intrin_wmma_store_matrix())
s[ConvF].tensorize(nnf, intrin_wmma_gemm())
ir_module = tvm.lower(s, [A, W, Conv], simple_mode=True)
print("Lowered IRModule")
print(ir_module)
func = tvm.build(s, [A, W, Conv], target="cuda")
print("Source Code")
print(func.imported_modules[0].get_source())

###############################################################################
# Generate CUDA Kernel
# --------------------
# Finally we use TVM to generate and compile the CUDA kernel, and evaluate the
# latency of convolution.
# Since TensorCores are only supported in NVIDIA GPU
# with Compute Capability 7.0
# or higher, it may not
# be able to run on our build server

ctx = tvm.gpu(0)
if nvcc.have_tensorcore(ctx.compute_version):
    with tvm.transform.PassContext(config={"tir.UnrollLoop": {"auto_max_step": 16}}):
        func = tvm.build(s, [A, W, Conv], "cuda")
    a_np = np.random.uniform(size=data_shape).astype("int8")
    w_np = np.random.uniform(size=kernel_shape).astype("int8")
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(np.zeros(output_shape, dtype="int32"), ctx)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
    print("conv2d with tensor core: %f ms" % (evaluator(a, w, c).mean * 1e3))

###############################################################################
# Summary
# -------
# This tutorial demonstrates how TVM scheduling primitives can be used to
# call TensorCores on specific GPUs.
