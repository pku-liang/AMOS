import tvm
from tvm import topi
from tvm.topi.util import get_const_int, get_const_tuple
from tvm.topi import nn
from tvm.topi.nn.winograd_util import winograd_transform_matrices
from tvm import te, autotvm
from tvm.topi import util
from tvm.contrib.util import tempdir
import numpy as np


def _infer_tile_size(data, kernel):
    N, CI, H, W = get_const_tuple(data.shape)

    if H % 8 == 0:
        return 4
    return 2


def winograd(
        data, kernel, strides, padding,
        dilation, out_dtype, pre_computed):
    """Compute declaration for winograd"""
    tile_size = _infer_tile_size(data, kernel)

    N, CI, H, W = get_const_tuple(data.shape)

    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")

    if not isinstance(H, int) or not isinstance(W, int):
        raise RuntimeError(
            "winograd conv2d doesn't support dynamic input\
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


def winograd_opencl(N, C, H, W, K, R, S, stride, padding, dilation):
    A = tvm.te.placeholder([N, C, H, W], dtype="float32")
    # B = tvm.te.placeholder([K, C, R, S], dtype="float32")
    tile_size = _infer_tile_size(A, None)
    alpha = tile_size + R - 1
    B = tvm.te.placeholder([alpha, alpha, C, K], dtype="float32")
    C = winograd(A, B, stride, padding, dilation, "float32", True)
    return [A, B, C]


def tile_axis(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))


def schedule_winograd(Input, Filter, Conv, pre_computed=True):
    sch = tvm.te.create_schedule(Conv.op)

    # get stages
    output = Conv
    inverse = output.op.input_tensors[0]
    bgemm, A = inverse.op.input_tensors
    kernel_pack, data_pack = bgemm.op.input_tensors
    input_tile, B = data_pack.op.input_tensors
    data_pad = input_tile.op.input_tensors[0]

    # inline const matrix
    sch[B].compute_inline()

    data_l = sch.cache_write(data_pack, "local")
    eps, nu, c, p = sch[data_l].op.axis
    r_a, r_b = sch[data_l].op.reduce_axis
    for axis in [eps, nu, r_a, r_b]:
        sch[data_l].unroll(axis)
    # sch[data_l].vectorize(c)
    sch[data_l].vectorize(p)

    eps, nu, c, p = sch[data_pack].op.axis
    p, pi = sch[data_pack].split(p, 1)
    fused = sch[data_pack].fuse(c, p)
    bb, tt = sch[data_pack].split(fused, 128)
    sch[data_pack].reorder(bb, tt, pi, eps, nu)
    sch[data_pack].bind(bb, te.thread_axis("blockIdx.x"))
    sch[data_pack].bind(tt, te.thread_axis("threadIdx.x"))

    sch[data_l].compute_at(sch[data_pack], pi)
    # put input tile at some position
    sch[input_tile].compute_at(sch[data_pack], pi)
    # inline padding
    sch[data_pad].compute_inline()

    if not pre_computed:
        kernel, G = sch[kernel_pack].op.input_tensors
        eps, nu, ci, co = sch[kernel_pack].op.axis
        sch[G].compute_inline()
        r_a, r_b = sch[kernel_pack].op.reduce_axis
        for axis in [eps, nu, r_a, r_b]:
            sch[kernel_pack].unroll(axis)
        # sch[kernel_pack].vectorize(eps)
        # sch[kernel_pack].vectorize(nu)

        fused = sch[kernel_pack].fuse(ci, co)
        bb, tt = sch[kernel_pack].split(fused, 128)
        sch[kernel_pack].bind(bb, te.thread_axis("blockIdx.x"))
        sch[kernel_pack].bind(tt, te.thread_axis("threadIdx.x"))

    else:
        kernel = kernel_pack

    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        sch[kernel].compute_inline()

    # schedule batch gemm
    b1, b2, y, x = sch[bgemm].op.axis
    rc = sch[bgemm].op.reduce_axis[0]
    alpha = get_const_int(b1.dom.extent)

    # factors
    b_factors = [alpha * alpha, 1, 1, 1]
    y_factors = [-1, 1, 16, 2]
    x_factors = [-1, 1, 20, 2]
    rc_factors = [-1, 20]
    unroll_step = 32
    unroll_explicit = 1
    m = alpha - 3 + 1

    OL = sch.cache_write(bgemm, "local")
    AA = sch.cache_read(kernel_pack, "shared", [OL])
    BB = sch.cache_read(data_pack, "shared", [OL])
    b = sch[bgemm].fuse(b1, b2)
    kernel_scope, b = sch[bgemm].split(b, nparts=1)
    bb, vb, tb, ib = tile_axis(sch, bgemm, b, b_factors)
    by, vy, ty, iy = tile_axis(sch, bgemm, y, y_factors)
    bx, vx, tx, ix = tile_axis(sch, bgemm, x, x_factors)
    sch[bgemm].bind(bb, te.thread_axis("blockIdx.z"))
    sch[bgemm].bind(by, te.thread_axis("blockIdx.y"))
    sch[bgemm].bind(bx, te.thread_axis("blockIdx.x"))
    sch[bgemm].bind(tb, te.thread_axis("threadIdx.z"))
    sch[bgemm].bind(ty, te.thread_axis("threadIdx.y"))
    sch[bgemm].bind(tx, te.thread_axis("threadIdx.x"))
    sch[bgemm].reorder(bb, by, bx, vb, vy, vx, tb, ty, tx, ib, iy, ix)
    # sch[bgemm].vectorize(ix)

    sch[OL].compute_at(sch[bgemm], tx)
    b1, b2, y, x = sch[OL].op.axis
    b = sch[OL].fuse(b1, b2)
    rc = sch[OL].op.reduce_axis[0]
    rco, rci = tile_axis(sch, OL, rc, rc_factors)
    sch[OL].reorder(rco, rci, b, y, x)
    # use vectorize
    # _, xi = sch[OL].split(x, factor=x_factors[-1])
    # sch[OL].vectorize(xi)

    # schedule shared memory
    sch[AA].compute_at(sch[OL], rco)
    sch[BB].compute_at(sch[OL], rco)
    # cooperative fetching
    for load in [AA, BB]:
        fused = sch[load].fuse(*list(sch[load].op.axis))
        # fused, iv = sch[load].split(fused, 4)
        # sch[load].vectorize(iv)
        fused, tx = sch[load].split(fused, x_factors[-2])
        fused, ty = sch[load].split(fused, y_factors[-2])
        fused, tz = sch[load].split(fused, b_factors[-2])
        sch[load].bind(tz, te.thread_axis("threadIdx.z"))
        sch[load].bind(ty, te.thread_axis("threadIdx.y"))
        sch[load].bind(tx, te.thread_axis("threadIdx.x"))

    sch[bgemm].pragma(kernel_scope, "auto_unroll_max_step", unroll_step)
    sch[bgemm].pragma(kernel_scope, "unroll_explicit", unroll_explicit)

    # schedule inverse, output and fusion
    n, co, h, w = sch[output].op.axis
    ho, wo, hi, wi = sch[output].tile(h, w, m, m)

    fused = sch[output].fuse(n, co, ho, wo)
    bb, tt = sch[output].split(fused, 128)

    sch[output].bind(bb, te.thread_axis("blockIdx.x"))
    sch[output].bind(tt, te.thread_axis("threadIdx.x"))

    sch[A].compute_inline()
    co, p, vh, vw = sch[inverse].op.axis
    r_a, r_b = sch[inverse].op.reduce_axis
    for axis in [vh, vw, r_a, r_b]:
        sch[inverse].unroll(axis)
    # sch[inverse].vectorize(vh)
    # sch[inverse].vectorize(vw)
    sch[inverse].compute_at(sch[output], tt)

    return sch


if __name__ == "__main__":
    N, C, H, W, K, R, S, stride, padding, dilation = (
        1, 128, 28, 28, 128, 3, 3, 1, 1, 1
    )
    use_android = True
    target = tvm.target.Target("opencl -device=mali")

    # Replace "aarch64-linux-gnu" with the correct target of your board.
    # This target host is used for cross compilation. You can query it by :code:`gcc -v` on your device.
    target_host = "llvm -mtriple=aarch64-linux-android"

    # Also replace this with the device key in your tracker
    device_key = "android"
    A, B, Conv = winograd_opencl(
        N, C, H, W, K, R, S, stride, padding, dilation)
    sch = schedule_winograd(A, B, Conv)
    # print(tvm.lower(sch, [A, B, Conv], simple_mode=True))
    func = tvm.build(sch, [A, B, Conv], target=target, target_host=target_host)
    print(func.imported_modules[0].get_source())
    # func = tvm.build(sch, [A, B, Conv], target="cuda")
    # print(func.imported_modules[0].get_source())

    tmp = tempdir()
    if use_android:
        from tvm.contrib import ndk

        filename = "net.so"
        func.export_library(tmp.relpath(filename), ndk.create_shared)
    else:
        filename = "net.tar"
        func.export_library(tmp.relpath(filename))

    # upload module to device
    print("Upload...")
    remote = autotvm.measure.request_remote(device_key, "0.0.0.0", 9190, timeout=10000)
    remote.upload(tmp.relpath(filename))
    module = remote.load_module(filename)

    # upload parameters to device
    ctx = remote.context(str(target), 0)

    # evaluate
    print("Evaluate inference time cost...")
    tvm_arys = [tvm.nd.empty(get_const_tuple(x.shape), x.dtype, ctx) for x in [A, B, Conv]]
    ftimer = module.time_evaluator(module.entry_name, ctx, number=400, repeat=1, min_repeat_ms=600)
    prof_res = np.array(ftimer(*tvm_arys).results) * 1000  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)"
        % (np.mean(prof_res), np.std(prof_res))
    )
