import logging
import sys
import numpy as np

import tvm
from tvm import topi
from tvm.topi import cuda

from tvm import autotvm


def intrin_wmma_load_matrix_a():
    n = 16
    A = tvm.te.placeholder((n, n), name="A", dtype="float16")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=256)
    C = tvm.te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="wmma.matrix_a", data_alignment=32, offset_factor=256)

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

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_b():
    n = 16
    A = tvm.te.placeholder((n, n), name="A", dtype="float16")
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, scope="shared", data_alignment=32, offset_factor=256)
    C = tvm.te.compute((n, n), lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, scope="wmma.matrix_b", data_alignment=32, offset_factor=256)

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
                "col_major",
            )
        )
        return ib.get()

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm():
    n = 16
    A = tvm.te.placeholder((n, n), name="A", dtype="float16")
    B = tvm.te.placeholder((n, n), name="B", dtype="float16")
    k = tvm.te.reduce_axis((0, n), name="k")
    C = tvm.te.compute(
        (n, n),
        lambda ii, jj: tvm.te.sum(A[ii, k].astype("float") * B[jj, k].astype("float"), axis=k),
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

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_wmma_store_matrix():
    n = 16
    A = tvm.te.placeholder((n, n), name="A", dtype="float32")
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="wmma.accumulator", data_alignment=32, offset_factor=256
    )
    C = tvm.te.compute((n, n), lambda i, j: A[i, j], name="C")
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

    return tvm.te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


@autotvm.template("conv2d_nchw_tensorcore_test")
def conv2d_nchw_tensorcore_test(N, H, W, CO, CI, KH, KW, stride, padding):
    P = (H + 2 * padding - KH) // stride + 1
    Q = (W + 2 * padding - KW) // stride + 1
    data = tvm.te.placeholder((N, CI, H, W), name='data', dtype="float16")
    padding = tvm.te.compute(
        [N, CI, H + 2 * padding, W + 2 * padding],
        lambda n, c, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h < H + padding, w >= padding, w < W + padding),
                data[n, c, h - padding, w - padding],
                tvm.tir.const(0.0, data.dtype)
            ),
        name="padding"
    )
    kernel = tvm.te.placeholder((CO, CI, KH, KW), name='kernel', dtype="float16")
    
    MM = N * P * Q
    MN = CO
    MK = CI * KH * KW
    data_matrix = tvm.te.compute(
        [MM, MK],
        lambda i, j:
            padding[
                i//(P*Q),
                j//(KH*KW),
                i%(P*Q)//Q*stride+j%(KH*KW)//KW,
                i%Q*stride+j%KW
            ],
        name="data_matrix"
    )
    kernel_matrix = tvm.te.compute(
        [MN, MK],
        lambda i, j:
            kernel[i, j//(KH*KW), j%(KH*KW)//KW, j%KW],
        name="kernel_matrix"
    )

    data_pack = tvm.te.compute(
        [(MM + 15) // 16, (MK + 15) // 16, 16, 16],
        lambda i, j, ii, jj:
            tvm.tir.if_then_else(
                tvm.tir.all(i * 16 + ii < MM, j * 16 + jj < MK),
                data_matrix[i * 16 + ii, j * 16 + jj],
                tvm.tir.const(0.0, data_matrix.dtype)
            ),
        name="data_pack"
    )
    kernel_pack = tvm.te.compute(
        [(MN + 15) // 16, (MK + 15) // 16, 16, 16],
        lambda i, j, ii, jj:
            tvm.tir.if_then_else(
                tvm.tir.all(i * 16 + ii < MN, j * 16 + jj < MK),
                kernel_matrix[i * 16 + ii, j * 16 + jj],
                tvm.tir.const(0.0, kernel_matrix.dtype)
            ),
        name="kernel_pack"
    )
    rk = tvm.te.reduce_axis([0, (MK + 15) // 16], name="rk")
    rki = tvm.te.reduce_axis([0, 16], name="rki")
    output_pack = tvm.te.compute(
        [(MM + 15) // 16, (MN + 15) // 16, 16, 16],
        lambda i, j, ii, jj:
            tvm.te.sum(
                data_pack[i, rk, ii, rki].astype("float32") * kernel_pack[j, rk, jj, rki].astype("float32"),
                axis=[rk, rki]
            ),
        name="output_pack"
    )

    if ((MM + 15) // 16 * 16 > MM) or ((MN + 15) // 16 * 16 > MN):
            output_matrix = tvm.te.compute(
                [MM, MN],
                lambda i, j:
                    output_pack[i//16, j//16, i%16, j%16] + output_pack[(MM + 15) // 16 * 16 - 1, (MN + 15) // 16 * 16 - 1],
                name="output_matrix"
            )
    else:
        output_matrix = tvm.te.compute(
            [MM, MN],
            lambda i, j:
                output_pack[i//16, j//16, i%16, j%16],
            name="output_matrix"
        )

    output = tvm.te.compute(
        [N, CO, P, Q],
        lambda n, c, p, q:
            output_matrix[n * (P*Q) + p * Q + q, c],
        name="output"
    )

    sch = tvm.te.create_schedule([output.op])

    ##### space definition begin #####
    cfg = autotvm.get_config()
    cfg.define_knob("output_thread", [32, 32 * 2, 32 * 4])
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])

    mm, mn, _, _ = sch[output_pack].op.axis
    cfg.define_split("tile_mm", mm, num_outputs=3)
    cfg.define_split("tile_mn", mn, num_outputs=3)
    mk, _ = sch[output_pack].op.reduce_axis
    cfg.define_split("tile_mk", mk, num_outputs=2)
    ##### space definition end #####

    # inline padding
    sch[padding].compute_inline()
    sch[data_matrix].compute_inline()
    sch[kernel_matrix].compute_inline()
    sch[data_pack].compute_inline()
    sch[kernel_pack].compute_inline()
    sch[output_matrix].compute_inline()

    OL = sch.cache_write(output_pack, "wmma.accumulator")

    # create cache stage
    AA = sch.cache_read(data_pack, "shared", [OL])
    WW = sch.cache_read(kernel_pack, "shared", [OL])
    AL = sch.cache_read(AA, "wmma.matrix_a", [OL])
    WL = sch.cache_read(WW, "wmma.matrix_b", [OL])

    # tile and bind spatial axes
    n, f, y, x = sch[output].op.axis
    factor = cfg["output_thread"].val
    fused = sch[output].fuse(n, f, y, x)
    block, thread = sch[output].split(fused, factor=factor)
    sch[output].bind(block, tvm.te.thread_axis("blockIdx.x"))
    sch[output].bind(thread, tvm.te.thread_axis("threadIdx.x"))
    
    # schedule output_matrix
    mm, mn, ttm, ttn = sch[output_pack].op.axis
    kernel_scope, mm = sch[output_pack].split(mm, nparts=1)
    bm, vm, tm = cfg["tile_mm"].apply(sch, output_pack, mm)
    bn, vn, tn = cfg["tile_mn"].apply(sch, output_pack, mn)
    sch[output_pack].reorder(bm, bn, vm, vn, tm, tn, ttm, ttn)

    sch[output_pack].bind(bm, tvm.te.thread_axis("blockIdx.y"))
    sch[output_pack].bind(bn, tvm.te.thread_axis("blockIdx.x"))
    sch[output_pack].bind(tm, tvm.te.thread_axis("threadIdx.z"))
    sch[output_pack].bind(tn, tvm.te.thread_axis("threadIdx.y"))
    sch[output_pack].tensorize(ttm, intrin_wmma_store_matrix())

    # schedule OL
    sch[OL].compute_at(sch[output_pack], tn)
    mm, mn, ttm, ttn = sch[OL].op.axis
    rk, ttk = sch[OL].op.reduce_axis
    rko, rki = cfg["tile_mk"].apply(sch, OL, rk)
    sch[OL].reorder(rko, rki, mm, mn, ttm, ttn, ttk)
    sch[OL].tensorize(ttm, intrin_wmma_gemm())

    # schedule AA, WW, AL, WL
    sch[AA].compute_at(sch[OL], rko)
    sch[WW].compute_at(sch[OL], rko)
    sch[AL].compute_at(sch[OL], rki)
    sch[WL].compute_at(sch[OL], rki)

    mm, mk, ttm, ttk = sch[AL].op.axis
    sch[AL].tensorize(ttm, intrin_wmma_load_matrix_a())
    mn, mk, ttn, ttk = sch[WL].op.axis
    sch[WL].tensorize(ttn, intrin_wmma_load_matrix_b())

    # cooperative fetching
    for load in [AA, WW]:
        axis = sch[load].op.axis
        fused = sch[load].fuse(*axis)
        tz, fused = sch[load].split(fused, nparts=cfg["tile_mm"].size[2])
        ty, fused = sch[load].split(fused, nparts=cfg["tile_mn"].size[2])
        tx, fused = sch[load].split(fused, nparts=32)
        fused, vec = sch[load].split(fused, factor=4)
        sch[load].bind(tz, tvm.te.thread_axis("threadIdx.z"))
        sch[load].bind(ty, tvm.te.thread_axis("threadIdx.y"))
        sch[load].bind(tx, tvm.te.thread_axis("threadIdx.x"))
        sch[load].vectorize(vec)

    # tune unroll
    sch[output_pack].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    sch[output_pack].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    return sch, [data, kernel, output]

######################################################################
# Step 2:  Search through the space
# ---------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBoostTuner` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this template

# logging config (for printing tuning log to screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


# the last layer in yolo
def run(name, N, H, W, CO, CI, KH, KW, stride, pad):
    N, H, W, CO, CI, KH, KW, strides, padding = N, H, W, CO, CI, KH, KW, stride, pad
    task = autotvm.task.create("conv2d_nchw_tensorcore_test",
                               args=(N, H, W, CO, CI, KH, KW, strides, padding),
                               target='cuda')
    print(task.config_space)
    logfile = "conv2d_" + name + ".log"

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=1000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(logfile)])

    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(logfile)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(logfile):
        with tvm.target.create("cuda"):
            s, arg_bufs = conv2d_nchw_tensorcore_test(N, H, W, CO, CI, KH, KW, strides, padding)
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float16)
    w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float16)
    # c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty((N, CO, (H + 2 * pad - KH) // stride + 1, (W + 2 * pad - KW) // stride + 1), ctx=ctx)
    # func(a_tvm, w_tvm, c_tvm)

    # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=400, min_repeat_ms=500)
    cost = evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3
    # print('Time cost of this operator: %f' % cost)
    # with open("autotvm_conv_nhwc.txt", "a") as f:
    #     f.write("name, {}\n".format(cost))
    return cost


res18_shapes_b1 = [
    # resnet-18
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (16, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (16, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (16, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (16, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    (16, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (16, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (16, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    (16, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (16, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (16, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (16, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
    # (256, 512, 7, 7, 512, 512, 1, 1, 1, 1, 0, 1, 1)
]


if __name__ == "__main__":
    costs = []
    for i, args in enumerate(res18_shapes_b1):
        name = "resnet-18-layer-" + str(i+1)
        N, CI, H, W, CO, _, KW, KH, _, stride, pad, _, _ = args
        try:
            cost = run(name, N, H, W, CO, CI, KH, KW, stride, pad)
        except Exception as e:
            print(e, flush=True)
            cost = float("inf")
        costs.append(cost)
    print("The costs:")
    for cost in costs:
        print(cost)
