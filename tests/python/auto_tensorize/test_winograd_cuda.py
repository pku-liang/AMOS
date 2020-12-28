import tvm
from tvm import topi
from tvm.topi.util import get_const_int, get_const_tuple
from tvm.topi import nn
from tvm.topi.nn.winograd_util import winograd_transform_matrices
from tvm import te, autotvm, auto_tensorize as at
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
    assert data.dtype == kernel.dtype
    A, B, G = winograd_transform_matrices(
        m, r, (out_dtype, str(data.dtype), str(data.dtype)))

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
            lambda co, ci, eps, nu: te.sum(
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
        (CI, P, alpha, alpha),
        lambda ci, p, eps, nu: te.sum(
            input_tile[ci][p][r_a][r_b] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="data_pack",
    )

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    bgemm = te.compute(
        (CO, P, alpha, alpha),
        lambda co, p, eps, nu: te.sum(
            (kernel_pack[co][ci][eps][nu] * data_pack[ci][p][eps][nu]).astype(out_dtype), axis=[ci]
        ),
        name="bgemm",
    )

    # inverse transform
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_a")
    inverse = te.compute(
        (CO, P, m, m),
        lambda co, p, vh, vw: te.sum(
            bgemm[co][p][r_a][r_b] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
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


def winograd_cuda(N, C, H, W, K, R, S, stride, padding, dilation):
    A = tvm.te.placeholder([N, C, H, W], dtype="float16")
    B = tvm.te.placeholder([K, C, R, S], dtype="float16")
    # tile_size = _infer_tile_size(A, None)
    # alpha = tile_size + R - 1
    # B = tvm.te.placeholder([alpha, alpha, C, K], dtype="float32")
    C = winograd(A, B, stride, padding, dilation, "float16", False)
    return [A, B, C]


def tensorize_tensorcore_fp16fp16(
    N, C, H, W, K, R, S, stride,
    padding, dilation, layer
):
    A, B, Conv = winograd_cuda(
        N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "conv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

    trials = 1000
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)

    result = at.auto_tensorize_v2(
        target_dag, target, log_file, measure_opt,
        trials=trials, verbose=True, transform_dump=True)
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app

    # load from file
    schedule_gen.load_from_file(log_file, clear=True)
    entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in file
    params, value = entry.record, 1 / entry.value
    print(value)
    print(params.to_json())

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)
    return cost


def run(N, C, H, W, K, R, S, stride,
        padding, dilation, layer):
    return tensorize_tensorcore_fp16fp16(
        N, C, H, W, K, R, S, stride,
        padding, dilation, layer)


res18_shapes_b1 = [
    # resnet-18
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 5
    num = 1
    for batch in batches:
        costs = []
        for i, shape in enumerate(res18_shapes_b1[beg:beg+num]):
            (_, C, H, W, K, _, R, S, _, stride,
                padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            # try:
            cost = run(
                N, C, H, W, K, R, S, stride,
                padding, dilation,
                i + beg + 1
            )
            costs.append(cost)
            # except Exception as e:
            #     print("Fail to run\n", str(e))
            #     costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)