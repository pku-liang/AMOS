import logging
import sys
import argparse

import numpy as np
import tvm
from tvm import te

from tvm import autotvm, topi
from tvm.contrib import nvcc

from tvm.topi import nn
from tvm.topi.util import get_const_int, get_const_tuple, traverse_inline
from tvm.topi.nn.winograd_util import winograd_transform_matrices


def _infer_tile_size(data, kernel):
    N, CI, H, W = get_const_tuple(data.shape)

    if H % 8 == 0:
        return 4
    return 2


def winograd_cuda(data, kernel, strides, padding, dilation, out_dtype, pre_computed):
    """Compute declaration for winograd"""
    tile_size = _infer_tile_size(data, kernel)

    N, CI, H, W = get_const_tuple(data.shape)

    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")

    if not isinstance(H, int) or not isinstance(W, int):
        raise RuntimeError(
            "cuda winograd conv2d doesn't support dynamic input\
                           height or width."
        )

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation
    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides

    if not pre_computed:  # kernel tensor is raw tensor, do strict check
        if dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (1, 1, dilation_h, dilation_w))
        CO, CI, KH, KW = get_const_tuple(kernel.shape)
        alpha = KW + tile_size - 1
        assert HSTR == 1 and WSTR == 1 and KH == KW
    else:
        # kernel tensor is pre-transfomred. this op is created by alter op layout.
        # dilation is not supported
        alpha, _, CI, CO = get_const_tuple(kernel.shape)
        KH = KW = alpha + 1 - tile_size
        assert HSTR == 1 and WSTR == 1 and dilation_h == 1 and dilation_w == 1

    pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))
    data_pad = nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")

    r = KW
    m = tile_size
    A, B, G = winograd_transform_matrices(m, r, out_dtype)

    H = (H + pt + pb - KH) // HSTR + 1
    W = (W + pl + pr - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m

    P = N * nH * nW if isinstance(N, int) else nH * nW

    # transform kernel
    if not pre_computed:
        r_kh = te.reduce_axis((0, KH), name="r_kh")
        r_kw = te.reduce_axis((0, KW), name="r_kw")
        kernel_pack = te.compute(
            (alpha, alpha, CI, CO),
            lambda eps, nu, ci, co: te.sum(
                kernel[co][ci][r_kh][r_kw] * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
            ),
            name="kernel_pack",
        )
    else:
        kernel_pack = kernel

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    # pack input tile
    input_tile = te.compute(
        (CI, P, alpha, alpha),
        lambda c, p, eps, nu: data_pad[idxdiv(p, (nH * nW))][c][
            idxmod(idxdiv(p, nW), nH) * m + eps
        ][idxmod(p, nW) * m + nu],
        name="d",
    )

    # transform data
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_a")
    data_pack = te.compute(
        (alpha, alpha, CI, P),
        lambda eps, nu, ci, p: te.sum(
            input_tile[ci][p][r_a][r_b] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="data_pack",
    )

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    bgemm = te.compute(
        (alpha, alpha, CO, P),
        lambda eps, nu, co, p: te.sum(
            kernel_pack[eps][nu][ci][co] * data_pack[eps][nu][ci][p], axis=[ci]
        ),
        name="bgemm",
    )

    # inverse transform
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_a")
    inverse = te.compute(
        (CO, P, m, m),
        lambda co, p, vh, vw: te.sum(
            bgemm[r_a][r_b][co][p] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
        ),
        name="inverse",
    )

    # output
    output = te.compute(
        (N, CO, H, W),
        lambda n, co, h, w: inverse[
            co, n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m), idxmod(h, m), idxmod(w, m)
        ],
        name="output",
        tag="conv2d_nchw_winograd",
    )

    return output


@autotvm.template("example/winograd")
def schedule_winograd_cuda(N, C, H, W, K, R, S, stride, padding, dilation, dtype, out_dtype):
    """Schedule winograd template"""
    # cfg, s, output, pre_computed
    # data, kernel, strides, padding, dilation, out_dtype, pre_computed
    data = tvm.te.placeholder([N, C, H, W], name="data", dtype=dtype)
    kernel = tvm.te.placeholder([K, C, R, S], name="kernel", dtype=dtype)
    pre_computed = False
    output = winograd_cuda(data, kernel, stride, padding, dilation, out_dtype, pre_computed)
    s = tvm.te.create_schedule(output.op)
    cfg = autotvm.get_config()
    # get stages
    inverse = s[output].op.input_tensors[0]
    bgemm, A = s[inverse].op.input_tensors
    kernel_pack, data_pack = s[bgemm].op.input_tensors
    input_tile, B = s[data_pack].op.input_tensors
    pad_data = s[input_tile].op.input_tensors[0]

    # data transform
    s[B].compute_inline()

    data_l = s.cache_write(data_pack, "local")
    eps, nu, c, p = s[data_l].op.axis
    r_a, r_b = s[data_l].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        s[data_l].unroll(axis)

    eps, nu, c, p = s[data_pack].op.axis
    p, pi = s[data_pack].split(p, 1)
    fused = s[data_pack].fuse(c, p)
    bb, tt = s[data_pack].split(fused, 128)
    s[data_pack].reorder(bb, tt, pi, eps, nu)
    s[data_pack].bind(bb, te.thread_axis("blockIdx.x"))
    s[data_pack].bind(tt, te.thread_axis("threadIdx.x"))

    s[data_l].compute_at(s[data_pack], pi)
    s[input_tile].compute_at(s[data_pack], pi)
    s[pad_data].compute_inline()

    # transform kernel
    if not pre_computed:
        kernel, G = s[kernel_pack].op.input_tensors
        eps, nu, ci, co = s[kernel_pack].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during pre-compute optimization pass
            s[G].pragma(s[G].op.axis[0], "debug_skip_region")
            s[kernel_pack].pragma(eps, "debug_skip_region")
        else:
            s[G].compute_inline()
            r_a, r_b = s[kernel_pack].op.reduce_axis
            for axis in [eps, nu, r_a, r_b]:
                s[kernel_pack].unroll(axis)

            fused = s[kernel_pack].fuse(ci, co)
            bb, tt = s[kernel_pack].split(fused, 128)
            s[kernel_pack].reorder(bb, tt, eps, nu, r_a, r_b)
            s[kernel_pack].bind(bb, te.thread_axis("blockIdx.x"))
            s[kernel_pack].bind(tt, te.thread_axis("threadIdx.x"))
    else:
        kernel = kernel_pack

    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    ##### space definition begin #####
    b1, b2, y, x = s[bgemm].op.axis
    rc = s[bgemm].op.reduce_axis[0]
    alpha = get_const_int(b1.dom.extent)

    cfg.define_split(
        "tile_b", cfg.axis(alpha * alpha), num_outputs=4, filter=lambda x: x.size[-3:] == [1, 1, 1]
    )
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 128, 1500])
    target = tvm.target.Target.current()
    if target.kind.name in ["nvptx", "rocm"]:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    # batch gemm
    C = bgemm
    A0, B0 = kernel_pack, data_pack

    OL = s.cache_write(C, "local")
    AA = s.cache_read(A0, "shared", [OL])
    BB = s.cache_read(B0, "shared", [OL])

    b = s[bgemm].fuse(b1, b2)

    # tile and bind spatial axes
    bgemm_scope, b = s[bgemm].split(b, nparts=1)
    bz, vz, tz, zi = cfg["tile_b"].apply(s, C, b)
    by, vy, ty, yi = cfg["tile_y"].apply(s, C, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, C, x)
    s[C].bind(bz, te.thread_axis("blockIdx.z"))
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(vz, te.thread_axis("vthread"))
    s[C].bind(vy, te.thread_axis("vthread"))
    s[C].bind(vx, te.thread_axis("vthread"))
    s[C].bind(tz, te.thread_axis("threadIdx.z"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi)

    # tile reduction axes
    s[OL].compute_at(s[C], tx)
    b1, b2, y, x = s[OL].op.axis
    b = s[OL].fuse(b1, b2)
    (rc,) = s[OL].op.reduce_axis
    rco, rci = cfg["tile_rc"].apply(s, OL, rc)
    s[OL].reorder(rco, rci, b, y, x)

    s[AA].compute_at(s[OL], rco)
    s[BB].compute_at(s[OL], rco)

    # cooperative fetching
    for load in [AA, BB]:
        fused = s[load].fuse(*list(s[load].op.axis))
        fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
        fused, tz = s[load].split(fused, cfg["tile_b"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))

    s[C].pragma(bgemm_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[C].pragma(bgemm_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    # schedule inverse, output and fusion
    if output.op in s.outputs:
        OL = None
    else:
        OL = output
        s[OL].set_scope("local")
        output = s.outputs[0]

    m = alpha - 3 + 1
    n, co, h, w = s[output].op.axis
    ho, wo, hi, wi = s[output].tile(h, w, m, m)
    inverse_scope, n = s[output].split(n, nparts=1)

    fused = s[output].fuse(n, co, ho, wo)
    bb, tt = s[output].split(fused, 128)

    s[output].bind(bb, te.thread_axis("blockIdx.x"))
    s[output].bind(tt, te.thread_axis("threadIdx.x"))

    if OL is not None:
        s[OL].compute_at(s[output], tt)

    s[A].compute_inline()
    co, p, vh, vw = s[inverse].op.axis
    r_a, r_b = s[inverse].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        s[inverse].unroll(axis)
    s[inverse].compute_at(s[output], tt)

    return s, [data, kernel, output]


def tune_and_evaluate(task_name, task_config, target, dev_id, log_name, task_func, naive_func=None, trials=1000, measure_trials=100, tune=True):
    task = autotvm.task.create(
        task_name, args=(*task_config,), target=target
    )
    print(task.config_space)

    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    measure_option = autotvm.measure_option(builder="local", runner=autotvm.LocalRunner(number=5))

    tuner = autotvm.tuner.XGBTuner(task)
    if tune:
        tuner.tune(
            n_trial=trials,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(log_name)],
        )

    dispatch_context = autotvm.apply_history_best(log_name)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)
    with autotvm.apply_history_best(log_name):
        with tvm.target.Target(target):
            s, arg_bufs = task_func(*task_config)
            print(tvm.lower(s, arg_bufs, simple_mode=True))
            func = tvm.build(s, arg_bufs)
    dev_module = func.imported_modules[0]
    with open(log_name + ".src", "w") as fout:
        print(dev_module.get_source(), file=fout)
    
    ctx = tvm.context(target, dev_id)
    if naive_func is not None:
        inputs_data, outputs_data = naive_func(*task_config)
        inputs_tvm = [tvm.nd.array(x, ctx=ctx) for x in inputs_data]
        outputs_tvm = [tvm.nd.array(np.zeros(x.shape, dtype=x.dtype), ctx=ctx) for x in outputs_data]
        func(*inputs_tvm, *outputs_tvm)

        for x, y in zip(outputs_data, outputs_tvm):
            tvm.testing.assert_allclose(x, y.asnumpy(), rtol=1e-3)
    
    feedin_tvm = [tvm.nd.array(np.random.uniform(-1, 1, get_const_tuple(x.shape)).astype(x.dtype), ctx=ctx) for x in arg_bufs]

    evaluator = func.time_evaluator(func.entry_name, ctx, number=measure_trials)
    print("Time cost of this operator: %f ms" % (evaluator(*feedin_tvm).mean * 1e3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()

    task_name = "example/winograd"
    N = 8
    C = 1024
    H = W = 7
    K = 1024
    R = S = 3
    stride = 1
    padding = R//2
    dilation = 1
    dtype = "float32"
    out_dtype = "float32"
    task_config = (N, C, H, W, K, R, S, stride, padding, dilation, dtype, out_dtype)
    target = "cuda"
    dev_id = 0
    log_name = "winograd.log"
    task_func = schedule_winograd_cuda
    tune = args.tune
    tune_and_evaluate(task_name, task_config, target, dev_id, log_name, task_func, naive_func=None, trials=1000, measure_trials=100, tune=tune)