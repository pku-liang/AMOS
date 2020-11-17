"""
This file is to explore how to use tensor core for convolution
with NCHW layout.
We use implicit GEMM method with packing.
We use autotvm to tune paramters.
---
Methodology:
    Change layout to [CRS/x, NHW, x] * [CRS/x, K, x].
    Use dimension fuse.
---
Notes:
    We don't consider output transformation.
"""


import tvm
import numpy as np
import logging
import sys

import time
from tvm import te
from tvm import topi
from tvm.topi.testing import conv2d_nchw_python

from tvm import autotvm


# The sizes of WMMA
WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
WARP_SIZE = 32
MAX_THREADS = 1024

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

# pack factor
# shared memory 128 bit per access
XX = 128 // int(dtype[-2:])

log_name = "conv2d_implicit_gemm_tensorcore.log"


def conv2d_implicit_gemm(
    N, C, H, W, K, R, S, stride=1, padding=0,
        dtype="float16", out_dtype="float32"):
    assert C % WMMA_K == 0, C
    pC = C // WMMA_K

    Src = tvm.te.placeholder([N, pC, H, W, WMMA_K], name="Src", dtype=dtype)
    Filter = tvm.te.placeholder(
        [K, pC, R, S, WMMA_K], name="Filter", dtype=dtype)

    Padded = tvm.te.compute(
        [N, pC, H + 2 * padding, W + 2 * padding, WMMA_K],
        lambda n, c, h, w, cc:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    h >= padding,
                    h + padding < H,
                    w >= padding,
                    w + padding < W),
                Src[n, c, h - padding, w - padding, cc],
                tvm.tir.const(0.0, dtype)
            ),
        name="Padded"
    )

    pH = H + 2 * padding
    pW = W + 2 * padding

    def get_n(nhw):
        return nhw // (pH * pW)

    def get_c(crs):
        return crs // (R * S)

    def get_r(crs):
        c = get_c(crs)
        rs = crs - c * (R * S)
        r = rs // S
        return r

    def get_s(crs):
        c = get_c(crs)
        rs = crs - c * (R * S)
        r = rs // S
        s = rs - r * S
        return s

    def compute_offset_h(nhw, crs):
        n = get_n(nhw)
        hw = nhw - n * (pH * pW)
        h = hw // pW
        w = hw - h * pW
        c = get_c(crs)
        rs = crs - c * (R * S)
        r = rs // S
        s = rs - r * S
        return h + r

    def compute_offset_w(nhw, crs):
        n = get_n(nhw)
        hw = nhw - n * (pH * pW)
        h = hw // pW
        w = hw - h * pW
        c = get_c(crs)
        rs = crs - c * (R * S)
        r = rs // S
        s = rs - r * S
        return w + s

    oH = (pH - R) // stride + 1
    oW = (pW - S) // stride + 1

    ChangedSrc = tvm.te.compute(
        [pC * R * S, N * oH * oW, WMMA_K],
        lambda k, i, cc:
            Padded[
                get_n(i),
                get_c(k),
                compute_offset_h(i, k),
                compute_offset_w(i, k),
                cc],
        name="ChangedSrc"
    )

    ChangedFilter = tvm.te.compute(
        [pC * R * S, K, WMMA_K],
        lambda k, i, cc:
            Filter[
                i,
                get_c(k),
                get_r(k),
                get_s(k),
                cc
            ],
        name="ChangedFilter"
    )

    rk1 = tvm.te.reduce_axis([0, pC * R * S], name="rk1")
    rk2 = tvm.te.reduce_axis([0, WMMA_K], name="rk2")

    # def assemble(xs, exts):
    #     assert len(xs) > 0
    #     ret = xs[0]
    #     assert len(xs) == len(exts)
    #     for x, ext in zip(xs[1:], exts[1:]):
    #         ret = ret * ext + x
    #     return ret

    Conv = tvm.te.compute(
        [N * oH * oW, K],
        lambda nhw, k:
            tvm.te.sum(
                (ChangedSrc[
                    rk1,
                    nhw,
                    rk2
                ] * ChangedFilter[rk1, k, rk2]).astype(out_dtype),
                axis=[rk1, rk2]
            ),
        name="Conv"
    )

    # The convolution should contain an output compute to
    # change back to NCHW layout
    # but the explicit layout change is not necessary here
    # because it could be eliminated by inlining the transform
    # if a successive layer consumes Conv.

    return Conv, [Src, Filter, Conv]


def intrin_wmma_load_matrix_a():
    A = tvm.te.placeholder((WMMA_M, WMMA_K), name="A", dtype=dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=8)
    C = tvm.te.compute((WMMA_M, WMMA_K), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="wmma.matrix_a", data_alignment=32, offset_factor=8)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                WMMA_M,
                WMMA_N,
                WMMA_K,
                BC.elem_offset // (WMMA_M * WMMA_K),
                BA.access_ptr("r"),
                WMMA_K,
                "row_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_b():
    A = tvm.te.placeholder((WMMA_N, WMMA_K), name="A", dtype=dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=8)
    C = tvm.te.compute((WMMA_N, WMMA_K), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="wmma.matrix_b", data_alignment=32, offset_factor=8)

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_load_matrix_sync",
                BC.data,
                WMMA_M,
                WMMA_N,
                WMMA_K,
                BC.elem_offset // (WMMA_N * WMMA_K),
                BA.access_ptr("r"),
                WMMA_K,
                "col_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(out_strides):
    n = 16
    A = tvm.te.placeholder((n, n), name="A", dtype=dtype)
    B = tvm.te.placeholder((n, n), name="B", dtype=dtype)
    k = tvm.te.reduce_axis((0, n), name="k")
    C = tvm.te.compute(
        (n, n),
        lambda ii, jj:
            tvm.te.sum((A[ii, k] * B[jj, k]).astype(out_dtype), axis=k),
        name="C",
    )
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, name="BA", scope="wmma.matrix_a",
        data_alignment=32, offset_factor=8
    )
    BB = tvm.tir.decl_buffer(
        B.shape, B.dtype, name="BB", scope="wmma.matrix_b",
        data_alignment=32, offset_factor=8
    )
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, name="BC", scope="wmma.accumulator",
        data_alignment=32, offset_factor=8,
        strides=out_strides
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle", "tir.tvm_fill_fragment",
                    BC.data, n, n, n, BC.elem_offset // 256, 0.0
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

    return tvm.te.decl_tensor_intrin(
        C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix(in_strides, out_strides):
    n = 16
    A = tvm.te.placeholder((n, n), name="A", dtype=out_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator",
        data_alignment=32, offset_factor=8,
        strides=in_strides
    )
    C = tvm.te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope="global",
        data_alignment=32, offset_factor=8,
        strides=out_strides)

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

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


class ConfigWMMA(object):
    def __init__(self, threadblock=None, warp=None):
        threadblock_gemm = (
            (128, 128, 128) if threadblock is None else threadblock)
        warp_gemm = (64, 64, 64) if warp is None else warp
        instruction_gemm = (WMMA_M, WMMA_N, WMMA_K)
        self.threadblock_gemm = threadblock_gemm
        self.warp_gemm = warp_gemm
        self.instruction_gemm = instruction_gemm

    def valid(self):
        num_threads = (
            (self.threadblock_gemm[0] // self.warp_gemm[0]) *
            (self.threadblock_gemm[1] // self.warp_gemm[1])) * WARP_SIZE
        return (self.threadblock_gemm[0] % self.warp_gemm[0] == 0
                and self.threadblock_gemm[1] % self.warp_gemm[1] == 0
                and self.threadblock_gemm[2] % self.warp_gemm[2] == 0
                and self.warp_gemm[0] % self.instruction_gemm[0] == 0
                and self.warp_gemm[1] % self.instruction_gemm[1] == 0
                and self.warp_gemm[2] % self.instruction_gemm[2] == 0
                and num_threads <= MAX_THREADS)

    def __repr__(self):
        return ("threadblock=" + str(self.threadblock_gemm) + "\n"
                + "warp=" + str(self.warp_gemm) + "\n")


@autotvm.template("tutorial/conv2d_implicit_gemm_tensorcore")
def schedule_conv2d_implicit_gemm(
    N, C, H, W, K, R, S, stride=1, padding=0,
        dtype="float16", out_dtype="float32", cfg=None):
    Conv, args = conv2d_implicit_gemm(
        N, C, H, W, K, R, S, stride=stride, padding=padding,
        dtype=dtype, out_dtype=out_dtype
    )

    sch = tvm.te.create_schedule(Conv.op)
    Src, Filter, _ = args
    ChangedSrc, ChangedFilter = Conv.op.input_tensors
    Padded = ChangedSrc.op.input_tensors[0]
    sch[Padded].compute_inline()
    sch[ChangedFilter].compute_inline()
    sch[ChangedSrc].compute_inline()

    AS = sch.cache_read(ChangedSrc, "shared", [Conv])
    BS = sch.cache_read(ChangedFilter, "shared", [Conv])
    AL = sch.cache_read(AS, "wmma.matrix_a", [Conv])
    BL = sch.cache_read(BS, "wmma.matrix_b", [Conv])
    OL = sch.cache_write(Conv, "wmma.accumulator")

    block_x = tvm.te.thread_axis("blockIdx.x")
    block_y = tvm.te.thread_axis("blockIdx.y")
    block_z = tvm.te.thread_axis("blockIdx.z")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    thread_y = tvm.te.thread_axis("threadIdx.y")
    thread_z = tvm.te.thread_axis("threadIdx.z")

    if cfg is not None:
        threadblock_gemm = cfg.threadblock_gemm
        warp_gemm = cfg.warp_gemm
        instruction_gemm = cfg.instruction_gemm
    else:
        cfg = autotvm.get_config()
        cfg.define_knob("tbx", [2**x for x in range(4, 11)])
        cfg.define_knob("tby", [2**x for x in range(4, 11)])
        cfg.define_knob("tbz", [2**x for x in range(4, 11)])
        cfg.define_knob("tx", [2**x for x in range(4, 11)])
        cfg.define_knob("ty", [2**x for x in range(4, 11)])
        cfg.define_knob("tz", [2**x for x in range(4, 11)])
        cfg.define_knob("unroll_step", [1, 128, 512, 1500])
        cfg.define_knob("unroll_explicit", [0, 1])
        cfg.define_knob("double_buffer_A", [0, 1])
        cfg.define_knob("double_buffer_B", [0, 1])
        threadblock_gemm = (
            cfg["tbx"].val, cfg["tby"].val, cfg["tbz"].val)
        warp_gemm = (
            cfg["tx"].val, cfg["ty"].val, cfg["tz"].val)
        instruction_gemm = (WMMA_M, WMMA_N, WMMA_K)

    nhw, k = sch[Conv].op.axis
    kernel_scope, nhw = sch[Conv].split(nhw, nparts=1)
    ko, k = sch[Conv].split(k, factor=threadblock_gemm[1])  # ko = 4
    kv, k = sch[Conv].split(k, factor=warp_gemm[1])  # kv = 2
    kt, ki = sch[Conv].split(k, factor=instruction_gemm[1])  # kt = 4
    nhwo, nhw = sch[Conv].split(nhw, factor=threadblock_gemm[0])  # nhwo = 392
    nhwv, nhw = sch[Conv].split(nhw, factor=warp_gemm[0])  # nhwv = 2
    nhwt, nhwi = sch[Conv].split(nhw, factor=instruction_gemm[0])  # nhwt = 4
    sch[Conv].reorder(nhwo, ko, nhwv, kv, nhwt, kt, nhwi, ki)
    # we bind xxxv to threads
    # there are 2 x 2 = 4 warps in a thread block
    # totally 2 x 2 x 32 = 128 threads in a thread block
    # each warp does 4 x 4 = 16 wmma store instructions
    sch[Conv].bind(ko, block_x)
    sch[Conv].bind(nhwo, block_y)
    sch[Conv].bind(kv, thread_y)
    sch[Conv].bind(nhwv, thread_z)
    sch[Conv].tensorize(
        nhwi, intrin_wmma_store_matrix([warp_gemm[1], 1], [out_channels, 1]))

    sch[OL].compute_at(sch[Conv], kv)
    nhw, k = sch[OL].op.axis
    crs, cc = sch[OL].op.reduce_axis
    # each warp does 4 x 4 x 4 = 64 wmma mma instructions
    kt, ki = sch[OL].split(k, factor=instruction_gemm[1])  # kt = 4
    nhwt, nhwi = sch[OL].split(nhw, factor=instruction_gemm[0])  # nhwt = 4
    crso, crst = sch[OL].split(
        crs, factor=warp_gemm[2] // instruction_gemm[2])  # crso = 36
    sch[OL].reorder(crso, nhwt, kt, crst, nhwi, ki, cc)
    OL_crso = crso
    sch[OL].tensorize(nhwi, intrin_wmma_gemm([warp_gemm[1], 1]))

    sch[AL].compute_at(sch[OL], OL_crso)
    crs, nhw, cc = sch[AL].op.axis
    # each warp loads 4 x 4 = 16 fragments
    nhwt, nhwi = sch[AL].split(nhw, factor=instruction_gemm[0])  # nhwt = 4
    sch[AL].reorder(nhwt, crs, nhwi, cc)
    sch[AL].tensorize(
        nhwi, intrin_wmma_load_matrix_a())

    sch[BL].compute_at(sch[OL], OL_crso)
    crs, k, cc = sch[BL].op.axis
    # each warp loads 4 x 4 = 16 fragments
    kt, ki = sch[BL].split(k, factor=instruction_gemm[1])  # kt = 4
    sch[BL].reorder(kt, crs, ki, cc)
    sch[BL].tensorize(
        ki, intrin_wmma_load_matrix_b())

    sch[AS].compute_at(sch[OL], OL_crso)
    # nhw = threadblock_gemm[0],
    # crs = warp_gemm[2] // instruction_gemm[2],
    # cc = instruction_gemm[2]
    num_threads = (
        (threadblock_gemm[0] // warp_gemm[0]) *
        (threadblock_gemm[1] // warp_gemm[1])) * WARP_SIZE
    crs, nhw, cc = sch[AS].op.axis
    # compute_at here so we can use vectorize
    sch[Padded].compute_at(sch[AS], nhw)
    # each warp loads 128 * 64 elements
    # two access each vector
    cco, cci = sch[AS].split(cc, factor=XX)  # cco = 2
    # nhwo, nhwi = sch[AS].split(nhw, factor=WARP_SIZE)  # nhwo = 4
    # crso, crsi = sch[AS].split(crs, factor=2)  # crso = 2
    # sch[AS].bind(crso, thread_z)
    # sch[AS].bind(crsi, thread_y)
    # sch[AS].bind(nhwi, thread_x)
    fused = sch[AS].fuse(crs, nhw)
    outer, fused = sch[AS].split(fused, factor=num_threads)
    tz, fused = sch[AS].split(
        fused, nparts=threadblock_gemm[0] // warp_gemm[0])
    ty, fused = sch[AS].split(
        fused, nparts=threadblock_gemm[1] // warp_gemm[1])
    tx, fused = sch[AS].split(
        fused, nparts=WARP_SIZE)
    sch[AS].bind(tz, thread_z)
    sch[AS].bind(ty, thread_y)
    sch[AS].bind(tx, thread_x)
    sch[AS].vectorize(cci)

    sch[BS].compute_at(sch[OL], OL_crso)
    crs, k, cc = sch[BS].op.axis  # k = 128, crs = 4, cc = 16
    # each warp loads 128 * 64 elements
    # two access each vector
    cco, cci = sch[BS].split(cc, factor=XX)  # crs = 8
    # ko, ki = sch[BS].split(k, factor=WARP_SIZE)  # ko = 4
    # crso, crsi = sch[BS].split(crs, factor=2)  # crso = 2
    # sch[BS].bind(crso, thread_z)
    # sch[BS].bind(crsi, thread_y)
    # sch[BS].bind(ki, thread_x)
    fused = sch[BS].fuse(crs, k)
    outer, fused = sch[BS].split(fused, factor=num_threads)
    tz, fused = sch[BS].split(
        fused, nparts=threadblock_gemm[0] // warp_gemm[0])
    ty, fused = sch[BS].split(
        fused, nparts=threadblock_gemm[1] // warp_gemm[1])
    tx, fused = sch[BS].split(
        fused, nparts=WARP_SIZE)
    sch[BS].bind(tz, thread_z)
    sch[BS].bind(ty, thread_y)
    sch[BS].bind(tx, thread_x)
    sch[BS].vectorize(cci)

    sch[Conv].pragma(
        kernel_scope, "auto_unroll_max_step", cfg["unroll_step"].val)
    sch[Conv].pragma(
        kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)
    if cfg["double_buffer_A"].val == 1:
        sch[AL].double_buffer()
    if cfg["double_buffer_B"].val == 1:
        sch[BL].double_buffer()

    return sch, args


def run(cfg=None):
    sch, args = schedule_conv2d_implicit_gemm(
        batch_size,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        stride=stride_h,
        padding=pad_h,
        dtype=dtype,
        out_dtype=out_dtype,
        cfg=cfg
    )
    ctx = tvm.gpu(0)
    data_shape = [int(x) for x in args[0].shape]
    kernel_shape = [int(x) for x in args[1].shape]
    output_shape = [int(x) for x in args[2].shape]
    a_np = np.random.uniform(size=data_shape).astype(dtype)
    w_np = np.random.uniform(size=kernel_shape).astype(dtype)
    a = tvm.nd.array(a_np, ctx)
    w = tvm.nd.array(w_np, ctx)
    c = tvm.nd.array(np.zeros(output_shape, dtype=out_dtype), ctx)
    # print(tvm.lower(sch, args, simple_mode=True))
    func = tvm.build(sch, args, target="cuda")
    evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
    cost = evaluator(a, w, c).mean * 1e3
    return cost


def brute_force_run():
    config_lst = []
    m = 4
    extent = 10
    for i in range(m, extent+1):
        for j in range(m, extent+1):
            for k in range(m, extent+1):
                for ii in range(m, i+1):
                    for jj in range(m, j+1):
                        for kk in range(m, k+1):
                            cfg = ConfigWMMA(
                                    [2**i, 2**j, 2**k],
                                    [2**ii, 2**jj, 2**kk]
                                )
                            if cfg.valid():
                                config_lst.append(cfg)
    min_cost = 9999999999
    with open("brute_force_tune_implicit_gemm.log", "w") as fout:
        for cfg in config_lst:
            try:
                cost = run(cfg)
                if cost < min_cost:
                    min_cost = cost
                    print(
                        "Known minimal: %f" % min_cost, file=fout, flush=True)
                    print(cfg, file=fout, flush=True)
            except Exception as e:
                continue

        print("Minimum cost is %f ms." % min_cost, file=fout, flush=True)


def autotvm_tune():
    # logging config (for printing tuning log to screen)
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    task_conv = autotvm.task.create(
        "tutorial/conv2d_implicit_gemm_tensorcore", args=(
            batch_size, in_channels, height, width, out_channels,
            kernel_h, kernel_w, stride_h, pad_h, dtype, out_dtype),
            target="cuda"
    )
    print(task_conv.config_space)

    # Use local gpu, measure 10 times for every config to reduce variance
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4),
    )

    # Begin tuning, log records to file `conv2d.log`
    tuner = autotvm.tuner.XGBTuner(task_conv)
    # tuner.tune(
    #     n_trial=1000,
    #     measure_option=measure_option,
    #     callbacks=[autotvm.callback.log_to_file(log_name)],
    # )

    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(log_name)
    best_config = dispatch_context.query(task_conv.target, task_conv.workload)
    print("\nBest config conv:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(log_name):
        with tvm.target.Target("cuda"):
            s, arg_bufs_conv = schedule_conv2d_implicit_gemm(
                batch_size, in_channels, height, width, out_channels,
                kernel_h, kernel_w, stride_h, pad_h, dtype, out_dtype)
            func_conv = tvm.build(s, arg_bufs_conv)

    # check correctness
    a_np = np.random.uniform(
        size=[int(x) for x in arg_bufs_conv[0].shape]).astype(dtype)
    w_np = np.random.uniform(
        size=[int(x) for x in arg_bufs_conv[1].shape]).astype(dtype)
    # c_np = conv2d_nchw_python(a_np, w_np, stride_h, pad_h)

    ctx = tvm.gpu(0)
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    output_tvm = tvm.nd.empty(
        [int(x) for x in arg_bufs_conv[-1].shape], ctx=ctx)

    beg = time.time()
    func_conv(a_tvm, w_tvm, output_tvm)
    end = time.time()

    print("Native host time passed %f ms" % ((end - beg) * 1e3))

    evaluator_conv = func_conv.time_evaluator(
        func_conv.entry_name, ctx, number=400)
    time_conv = evaluator_conv(a_tvm, w_tvm, output_tvm).mean * 1e3
    print("Time cost of this operator: %f" % (time_conv))


if __name__ == "__main__":
    autotvm_tune()