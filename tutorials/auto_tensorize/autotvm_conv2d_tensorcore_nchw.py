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
This file is to explore how to use tensor core for convolution
with NCHW layout.
We use autotvm for parameter tuning.
---
Methodology:
    Change layout to adapt to tensor core.
    Use dimension split.
---
Transformation:
    NKPQ = NCHW * KCRS --> NKPQnk = NCHWnc * RSCKck
---
Notes:
    The convolution will finally become two separate kernels,
    one is tensor core convolution, the other is output transform.
    They are scheduled separately.
"""

######################################################################
# Install dependencies
# --------------------
# To use autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make TVM run faster in tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import logging
import sys
import numpy as np

import tvm
import time
from tvm import te
from tvm import topi
from tvm.topi.testing import conv2d_nchw_python

from tvm import autotvm

######################################################################
# Step 1:  Define the search space
# --------------------------------
# There are plenty of useful schedule primitives in tvm. You can also find
# some tutorials that describe them in more details, such as
# (1). :ref:`opt-conv-gpu`
# (2). `Optimizing DepthwiseConv on NVIDIA GPU <https://tvm.apache.org/2017/08/22/Optimize-Deep-Learning-GPU-Operators-with-TVM-A-Depthwise-Convolution-Example>`_
#
# However, their implementations are manually tuned for some special input
# shapes. In this section, we build a large enough space to cover
# the techniques used in these tutorials. Then we rely on the efficient auto-tuner
# to search through this space and pick some good configurations.
#
# If you are familiar with writing cuda schedule, you can find the following
# template is very general. Actually this template can be easily modified
# to tune other operators such as depthwise convolution and gemm.
# In order to fully understand this template, you should be familiar with
# the schedule primitives and auto tuning API. You can refer to the above
# tutorials and :doc:`autotvm tutorial <tune_simple_template>`
#
# It is worth noting that the search space for a conv2d operator
# can be very large (at the level of 10^9 for some input shapes)
#


# The sizes of WMMA
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
WARP_SIZE = 32

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

# data type
dtype = "float16"
out_dtype = "float32"

log_name = "conv2d_nchw_tensorcore.log"
log_transform_name = "conv2d_nchw_tensorcore_transform.log"


def intrin_wmma_load_matrix(scope):
    n = 16
    A = te.placeholder((n, n), name="A", dtype=dtype)
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=256)
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope=scope, data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                n,
                n,
                n,
                BC.elem_offset // 256,
                BA.access_ptr("r"),
                n,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm():
    n = 16
    A = te.placeholder((n, n), name="A", dtype=dtype)
    B = te.placeholder((n, n), name="B", dtype=dtype)
    k = te.reduce_axis((0, n), name="k")
    C = te.compute(
        (n, n),
        lambda ii, jj: te.sum((A[ii, k] * B[k, jj]).astype(out_dtype), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a", data_alignment=32, offset_factor=256
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b", data_alignment=32, offset_factor=256
    )
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="BC", scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle", "tir.tvm_fill_fragment", BC.data, n, n, n, BC.elem_offset // 256, 0.0
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    BC.elem_offset // 256,
                    BA.data,
                    BA.elem_offset // 256,
                    BB.data,
                    BB.elem_offset // 256,
                    BC.data,
                    BC.elem_offset // 256,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix():
    n = 16
    A = te.placeholder((n, n), name="A", dtype=out_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )
    C = te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="global", data_alignment=32, offset_factor=256)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                n,
                n,
                n,
                BA.elem_offset // 256,
                BC.access_ptr("w"),
                n,
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def conv2d_nchw_by_nchwnc(N, C, H, W, K, R, S,
                          stride=1, padding=0, dilation=1,
                          dtype="float16", out_dtype="float32"):
    Src = tvm.te.placeholder([N, C, H, W], name="Src", dtype=dtype)
    Filter = tvm.te.placeholder([K, C, R, S], name="Filter", dtype=dtype)

    Padded = tvm.te.compute(
        [N, C, H+2*padding, W+2*padding],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                h >= padding, h + padding < H, w >= padding, w + padding < W),
            Src[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, dtype)
        ),
        name="Padded"
    )

    ChangedSrc = tvm.te.compute(
        [(N + WMMA_M - 1) // WMMA_M,
         (C + WMMA_K - 1) // WMMA_K,
         H + 2 * padding,
         W + 2 * padding,
         WMMA_M,
         WMMA_K],
        lambda n, c, h, w, nn, cc: tvm.tir.if_then_else(
            tvm.tir.all(n * WMMA_M + nn < N, c * WMMA_K + cc < C),
            Padded[n * WMMA_M + nn, c * WMMA_K + cc, h, w],
            tvm.tir.const(0.0, dtype)
        ),
        name="ChangedSrc"
    )

    ChangedFilter = tvm.te.compute(
        [R,
         S,
         (C + WMMA_K - 1) // WMMA_K,
         (K + WMMA_N - 1) // WMMA_N,
         WMMA_K,
         WMMA_N],
        lambda k, c, r, s, cc, kk: tvm.tir.if_then_else(
            tvm.tir.all(k * WMMA_N + kk < K, c * WMMA_K + cc < C),
            Filter[k * WMMA_N + kk, c * WMMA_K + cc, r, s],
            tvm.tir.const(0.0, dtype)
        ),
        name="ChangedFilter"
    )

    kh = (R - 1) * dilation + 1
    kw = (S - 1) * dilation + 1
    P = (H + 2 * padding - kh) // stride + 1
    Q = (W + 2 * padding - kw) // stride + 1

    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")
    rco = tvm.te.reduce_axis([0, (C + WMMA_K - 1) // WMMA_K], name="rco")
    rci = tvm.te.reduce_axis([0, WMMA_K], name="rci")

    ChangedOutput = tvm.te.compute(
        [(N + WMMA_M - 1) // WMMA_M,
         (K + WMMA_N - 1) // WMMA_N,
         P,
         Q,
         WMMA_M,
         WMMA_N],
        lambda n, k, p, q, nn, kk: tvm.te.sum(
            (ChangedSrc[n, rco, p*stride + rr*dilation,
                        q * stride + rr * dilation, nn, rci]
                * ChangedFilter[rr, rs, rco, k, rci, kk]).astype(out_dtype),
            axis=[rr, rs, rco, rci]
        ),
        name="ChangedOutput"
    )

    # Output = tvm.te.compute(
    #     [N, K, P, Q],
    #     lambda n, k, p, q: ChangedOutput[
    #         (n + WMMA_M - 1) // WMMA_M,
    #         (k + WMMA_N - 1) // WMMA_N,
    #         p, q, n % WMMA_M, k % WMMA_N],
    #     name="Output"
    # )

    # return Output, [Src, Filter, Output]
    return ChangedOutput, [Src, Filter, ChangedOutput]


def transform(N, K, P, Q):
    ChangedOutput = tvm.te.placeholder(
        [(N + WMMA_M - 1) // WMMA_M,
         (K + WMMA_N - 1) // WMMA_N,
         P,
         Q,
         WMMA_M,
         WMMA_N],
        name="ChangedOutput",
        dtype=out_dtype
    )

    Output = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q: ChangedOutput[
            (n + WMMA_M - 1) // WMMA_M,
            (k + WMMA_N - 1) // WMMA_N,
            p, q, n % WMMA_M, k % WMMA_N],
        name="Output"
    )

    return Output, [ChangedOutput, Output]


@autotvm.template("tutorial/conv2d_nchw_tensorcore")
def conv2d_nchw_tensorcore(N, H, W, CO, CI, KH, KW, stride, padding):
    ChangedOutput, args = conv2d_nchw_by_nchwnc(
        N, CI, H, W, CO, KH, KW,
        stride=stride, padding=padding, dtype=dtype, out_dtype=out_dtype)

    Src, Filter, _ = args
    # ChangedOutput = Output.op.input_tensors[0]
    ChangedSrc, ChangedFilter = ChangedOutput.op.input_tensors
    Padded = ChangedSrc.op.input_tensors[0]

    sch = te.create_schedule([ChangedOutput.op])

    # space definition begin
    # first, Output schedule space
    # n, k, p, q = sch[Output].op.axis

    cfg = autotvm.get_config()
    # cfg.define_split("tile_n_output", n, num_outputs=4)
    # cfg.define_split("tile_k_output", k, num_outputs=4)
    # cfg.define_knob("auto_unroll_max_step_output", [0, 512, 1500])
    # cfg.define_knob("unroll_explicit_output", [0, 1])

    # second, ChangedOutput schedule space
    n, k, p, q, nn, kk = sch[ChangedOutput].op.axis
    cfg.define_split("tile_n_changed_output", n, num_outputs=3)
    cfg.define_split("tile_k_changed_output", k, num_outputs=3)
    cfg.define_knob("auto_unroll_max_step", [0, 16, 64, 128])
    cfg.define_knob("unroll_explicit", [0, 1])
    # cfg.define_knob("use_vectorize_AS", [1, 2, 4])
    # cfg.define_knob("use_vectorize_WS", [1, 2, 4])

    _, _, rco, _ = sch[ChangedOutput].op.reduce_axis
    cfg.define_split("tile_rco", rco, num_outputs=2)
    # space definition end

    # schedule inputs and create cache
    sch[Padded].compute_inline()
    sch[ChangedSrc].compute_inline()
    sch[ChangedFilter].compute_inline()
    AS = sch.cache_read(ChangedSrc, "shared", [ChangedOutput])
    WS = sch.cache_read(ChangedFilter, "shared", [ChangedOutput])
    AL = sch.cache_read(AS, "wmma.matrix_a", [ChangedOutput])
    WL = sch.cache_read(WS, "wmma.matrix_b", [ChangedOutput])
    OL = sch.cache_write(ChangedOutput, "wmma.accumulator")

    # schedule Output
    # block_x = tvm.te.thread_axis("blockIdx.x")
    # block_y = tvm.te.thread_axis("blockIdx.y")
    # block_z = tvm.te.thread_axis("blockIdx.z")
    # thread_vx = tvm.te.thread_axis("vthread")
    # thread_vy = tvm.te.thread_axis("vthread")
    # thread_x = tvm.te.thread_axis("threadIdx.x")
    # thread_y = tvm.te.thread_axis("threadIdx.y")
    # thread_z = tvm.te.thread_axis("threadIdx.z")

    # n, k, p, q = sch[Output].op.axis
    # kernel_scope, n = sch[Output].split(n, nparts=1)
    # bn, vn, tn, ni = cfg["tile_n_output"].apply(sch, Output, n)
    # bk, vk, tk, ki = cfg["tile_k_output"].apply(sch, Output, k)
    # pq = sch[Output].fuse(p, q)
    # sch[Output].reorder(bn, bk, pq, vn, vk, tn, tk, ni, ki)
    # sch[Output].bind(bn, block_z)
    # sch[Output].bind(bk, block_y)
    # sch[Output].bind(pq, block_x)
    # sch[Output].bind(vn, thread_vy)
    # sch[Output].bind(vk, thread_vx)
    # sch[Output].bind(tn, thread_y)
    # sch[Output].bind(tk, thread_x)
    # sch[Output].pragma(
    #     kernel_scope,
    #     "auto_unroll_max_step",
    #     cfg["auto_unroll_max_step_output"].val)
    # sch[Output].pragma(
    #     kernel_scope,
    #     "unroll_explicit",
    #     cfg["unroll_explicit_output"].val)

    # schedule WMMA
    block_x = tvm.te.thread_axis("blockIdx.x")
    block_y = tvm.te.thread_axis("blockIdx.y")
    block_z = tvm.te.thread_axis("blockIdx.z")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    thread_y = tvm.te.thread_axis("threadIdx.y")
    thread_z = tvm.te.thread_axis("threadIdx.z")
    # schedule ChangedOutput
    n, k, p, q, nn, kk = sch[ChangedOutput].op.axis
    kernel_scope, n = sch[ChangedOutput].split(n, nparts=1)
    bn, tn, ni = cfg["tile_n_changed_output"].apply(sch, ChangedOutput, n)
    bk, tk, ki = cfg["tile_k_changed_output"].apply(sch, ChangedOutput, k)
    pq = sch[ChangedOutput].fuse(p, q)
    sch[ChangedOutput].reorder(pq, bn, bk, tn, tk, ni, ki, nn, kk)
    sch[ChangedOutput].bind(pq, block_z)
    sch[ChangedOutput].bind(bn, block_y)
    sch[ChangedOutput].bind(bk, block_x)
    sch[ChangedOutput].bind(tn, thread_z)
    sch[ChangedOutput].bind(tk, thread_y)
    sch[ChangedOutput].tensorize(nn, intrin_wmma_store_matrix())

    # schedule OL
    sch[OL].compute_at(sch[ChangedOutput], tk)
    n, k, p, q, nn, kk = sch[OL].op.axis
    rr, rs, rco, rci = sch[OL].op.reduce_axis
    rcoo, rcoi = cfg["tile_rco"].apply(sch, OL, rco)
    sch[OL].reorder(rcoo, rr, rcoi, rs, n, k, nn, kk, rci)
    sch[OL].tensorize(nn, intrin_wmma_gemm())

    # schedule AL, WL
    sch[AL].compute_at(sch[OL], rs)
    sch[WL].compute_at(sch[OL], rs)
    sch[AL].tensorize(
        sch[AL].op.axis[-2], intrin_wmma_load_matrix("wmma.matrix_a"))
    sch[WL].tensorize(
        sch[WL].op.axis[-2], intrin_wmma_load_matrix("wmma.matrix_b"))

    # schedule AS
    sch[AS].compute_at(sch[OL], rr)
    n, c, h, w, nn, cc = sch[AS].op.axis
    no, ni = sch[AS].split(n, nparts=cfg["tile_n_changed_output"].size[1])
    co, ci = sch[AS].split(c, nparts=cfg["tile_k_changed_output"].size[1])
    t = sch[AS].fuse(nn, cc)
    to, ti = sch[AS].split(t, factor=WARP_SIZE)
    sch[AS].bind(no, thread_z)
    sch[AS].bind(co, thread_y)
    sch[AS].bind(ti, thread_x)
    # if cfg["use_vectorize_AS"].val > 1:
    _, ti = sch[AS].split(ti, factor=4)
    sch[AS].vectorize(ti)

    # schedule WS
    sch[WS].compute_at(sch[OL], rr)
    r, s, c, k, cc, kk = sch[WS].op.axis
    ko, ki = sch[WS].split(k, nparts=cfg["tile_n_changed_output"].size[1])
    co, ci = sch[WS].split(c, nparts=cfg["tile_k_changed_output"].size[1])
    t = sch[WS].fuse(cc, kk)
    to, ti = sch[WS].split(t, factor=WARP_SIZE)
    sch[WS].bind(ko, thread_z)
    sch[WS].bind(co, thread_y)
    sch[WS].bind(ti, thread_x)
    # if cfg["use_vectorize_WS"].val > 1:
    _, ti = sch[WS].split(ti, factor=4)
    sch[WS].vectorize(ti)

    # tune unroll
    sch[ChangedOutput].pragma(
        kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    sch[ChangedOutput].pragma(
        kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    return sch, [Src, Filter, ChangedOutput]


@autotvm.template("tutorial/conv2d_nchw_tensorcore_transform")
def conv2d_transform(N, K, P, Q):
    Output, args = transform(N, K, P, Q)

    ChangedOutput, _ = args

    sch = te.create_schedule([Output.op])

    AS = sch.cache_read(args[0], "shared", [Output])
    # OL = sch.cache_write(Output, "local")

    # space definition begin
    # first, Output schedule space
    n, k, p, q = sch[Output].op.axis

    cfg = autotvm.get_config()
    cfg.define_split("tile_n_output", n, num_outputs=4)
    cfg.define_split("tile_k_output", k, num_outputs=4)
    # cfg.define_knob("auto_unroll_max_step_output", [0, 512, 1500])
    # cfg.define_knob("unroll_explicit_output", [0, 1])
    # space definition end

    # schedule Output
    block_x = tvm.te.thread_axis("blockIdx.x")
    block_y = tvm.te.thread_axis("blockIdx.y")
    block_z = tvm.te.thread_axis("blockIdx.z")
    thread_vx = tvm.te.thread_axis("vthread")
    thread_vy = tvm.te.thread_axis("vthread")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    thread_y = tvm.te.thread_axis("threadIdx.y")
    thread_z = tvm.te.thread_axis("threadIdx.z")

    n, k, p, q = sch[Output].op.axis
    # kernel_scope, n = sch[Output].split(n, nparts=1)
    bn, vn, tn, ni = cfg["tile_n_output"].apply(sch, Output, n)
    bk, vk, tk, ki = cfg["tile_k_output"].apply(sch, Output, k)
    pq = sch[Output].fuse(p, q)
    sch[Output].reorder(bn, bk, pq, vn, vk, tn, tk, ni, ki)
    sch[Output].bind(bn, block_z)
    sch[Output].bind(bk, block_y)
    sch[Output].bind(pq, block_x)
    sch[Output].bind(vn, thread_vy)
    sch[Output].bind(vk, thread_vx)
    sch[Output].bind(tn, thread_y)
    sch[Output].bind(tk, thread_x)
    # sch[Output].pragma(
    #     kernel_scope,
    #     "auto_unroll_max_step",
    #     cfg["auto_unroll_max_step_output"].val)
    # sch[Output].pragma(
    #     kernel_scope,
    #     "unroll_explicit",
    #     cfg["unroll_explicit_output"].val)

    sch[AS].compute_at(sch[Output], tk)
    n, k, p, q, nn, ii = sch[AS].op.axis
    no, ni = sch[AS].split(n, nparts=cfg["tile_n_output"].size[2])
    ko, ki = sch[AS].split(k, nparts=cfg["tile_k_output"].size[2])
    sch[AS].bind(no, thread_y)
    sch[AS].bind(ko, thread_x)
    t = sch[AS].fuse(nn, ii)
    to, ti = sch[AS].split(t, nparts=WARP_SIZE)
    sch[AS].bind(to, thread_z)
    _, ti = sch[AS].split(ti, factor=4)
    sch[AS].vectorize(ti)

    # sch[OL].compute_at(sch[Output], tk)

    cfg.add_flop(N * K * P * Q)

    return sch, [ChangedOutput, Output]


######################################################################
# Step 2:  Search through the space
# ---------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBoostTuner` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this template

# logging config (for printing tuning log to screen)
logging.getLogger("autotvm").setLevel(logging.DEBUG)
logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

task_conv = autotvm.task.create(
    "tutorial/conv2d_nchw_tensorcore", args=(
        batch_size, height, width, out_channels, in_channels,
        kernel_h, kernel_w, stride_h, pad_h), target="cuda"
)
print(task_conv.config_space)

# Use local gpu, measure 10 times for every config to reduce variance
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
)

# Begin tuning, log records to file `conv2d.log`
# During tuning we will also try many invalid configs, so you are expected to
# see many error reports. As long as you can see non-zero GFLOPS, it is okay.
tuner = autotvm.tuner.XGBTuner(task_conv)
# tuner.tune(
#     n_trial=1000,
#     measure_option=measure_option,
#     callbacks=[autotvm.callback.log_to_file(log_name)],
# )


task_transform = autotvm.task.create(
    "tutorial/conv2d_nchw_tensorcore_transform", args=(
        batch_size, out_channels,
        (height + 2 * pad_h - kernel_h) // stride_h + 1,
        (width + 2 * pad_w - kernel_w) // stride_w + 1,
    ), target="cuda"
)
print(task_transform.config_space)

# Use local gpu, measure 10 times for every config to reduce variance
measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
)

# Begin tuning, log records to file `conv2d.log`
# During tuning we will also try many invalid configs, so you are expected to
# see many error reports. As long as you can see non-zero GFLOPS, it is okay.
tuner = autotvm.tuner.XGBTuner(task_transform)
# tuner.tune(
#     n_trial=1000,
#     measure_option=measure_option,
#     callbacks=[autotvm.callback.log_to_file(log_transform_name)],
# )

#########################################################################
# Finally we can inspect the best config from log file, check correctness,
# and measure running time.

# inspect the best config
dispatch_context = autotvm.apply_history_best(log_name)
best_config = dispatch_context.query(task_conv.target, task_conv.workload)
print("\nBest config conv:")
print(best_config)

dispatch_context = autotvm.apply_history_best(log_transform_name)
best_config = dispatch_context.query(
    task_transform.target, task_transform.workload)
print("\nBest config transform:")
print(best_config)

# apply history best from log file
with autotvm.apply_history_best(log_name):
    with autotvm.apply_history_best(log_transform_name):
        with tvm.target.Target("cuda"):
            s, arg_bufs_conv = conv2d_nchw_tensorcore(
                batch_size, height, width, out_channels, in_channels,
                kernel_h, kernel_w, stride_h, pad_h)
            func_conv = tvm.build(s, arg_bufs_conv)
            s, arg_bufs_trans = conv2d_transform(
                batch_size, out_channels,
                (height + 2 * pad_h - kernel_h) // stride_h + 1,
                (width + 2 * pad_w - kernel_w) // stride_w + 1,
            )
            print(tvm.lower(s, arg_bufs_trans, simple_mode=True))
            func_trans = tvm.build(s, arg_bufs_trans)

# check correctness
a_np = np.random.uniform(
    size=(batch_size, in_channels, height, width)).astype(dtype)
w_np = np.random.uniform(
    size=(out_channels, in_channels, kernel_h, kernel_w)).astype(dtype)
# c_np = conv2d_nchw_python(a_np, w_np, stride_h, pad_h)

ctx = tvm.gpu(0)
a_tvm = tvm.nd.array(a_np, ctx=ctx)
w_tvm = tvm.nd.array(w_np, ctx=ctx)
output_tvm = tvm.nd.empty([int(x) for x in arg_bufs_conv[-1].shape], ctx=ctx)
c_tvm = tvm.nd.empty([int(x) for x in arg_bufs_trans[-1].shape], ctx=ctx)

beg = time.time()
func_conv(a_tvm, w_tvm, output_tvm)
func_trans(output_tvm, c_tvm)
end = time.time()

# tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)
print("Native host time passed %f ms" % ((end - beg) * 1e3))

evaluator_conv = func_conv.time_evaluator(
    func_conv.entry_name, ctx, number=400)
evaluator_trans = func_trans.time_evaluator(
    func_trans.entry_name, ctx, number=400)
time_conv = evaluator_conv(a_tvm, w_tvm, output_tvm).mean * 1e3
time_trans = evaluator_trans(output_tvm, c_tvm).mean * 1e3
print("Ideal time cost of this operator: %f" % (time_conv + time_trans))
