import tvm
from tvm import auto_tensorize as at


def conv2d_nchw(N, C, H, W, K, R, S, CI, KI, stride, padding, dilation):
    pR = (R - 1) * dilation + 1
    pS = (S - 1) * dilation + 1
    P = (H + 2 * padding - pR) // stride + 1
    Q = (W + 2 * padding - pS) // stride + 1
    CO = (C + CI - 1) // CI
    KO = (K + KI - 1) // KI
    ELE = 4
    assert CI % ELE == 0
    A = tvm.te.placeholder(
        [N, C, H, W],
        name="A",
        dtype="uint8"
    )
    B = tvm.te.placeholder(
        [K, C, R, S],
        name="B",
        dtype="int8"
    )
    data_vec = tvm.te.compute(
        [N, CO, H, W, CI],
        lambda n, co, h, w, ci:
            tvm.tir.if_then_else(
                co * CI + ci < C,
                A[n, co * CI + ci, h, w],
                tvm.tir.const(0.0, A.dtype)
        ),
        name="data_vec"
    )
    padded = tvm.te.compute(
        [N, CO, H + 2 * padding, W + 2 * padding, CI],
        lambda n, co, h, w, ci:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h < H + padding,
                            w >= padding, w < W + padding),
                data_vec[n, co, h - padding, w - padding, ci],
                tvm.tir.const(0.0, A.dtype)
        ),
        name="padded"
    )
    filter_vec = tvm.te.compute(
        [KO, CO, R, S, CI//ELE, KI, ELE],
        lambda ko, co, r, s, cie, ki, ele:
            tvm.tir.if_then_else(
                tvm.tir.all(ko * KI + ki < K, co * CI + cie * ELE + ele < C),
                B[ko * KI + ki, co * CI + cie * ELE + ele, r, s],
                tvm.tir.const(0.0, B.dtype)
        ),
        name="filter_vec"
    )
    rco = tvm.te.reduce_axis([0, CO], "rco")
    rcie = tvm.te.reduce_axis([0, CI], "rcie")
    rele = tvm.te.reduce_axis([0, ELE], "rele")
    rr = tvm.te.reduce_axis([0, R], "rr")
    rs = tvm.te.reduce_axis([0, S], "rs")
    Output = tvm.te.compute(
        [N, KO, P, Q, KI],
        lambda n, ko, p, q, ki:
            tvm.te.sum(
                padded[n, rco, p * stride + rr * dilation, q * stride +
                       rs * dilation, rcie * ELE + rele].astype("int32")
                * filter_vec[ko, rco, rr, rs, rcie, ki, rele].astype("int32"),
                axis=[rr, rs, rco, rcie, rele]
        ),
        name="Output"
    )

    return [A, B], [Output]


def schedule_conv2d_nvvi(A, B, Output):
    padded, kernel_vec = Output.op.input_tensors
    data_vec = padded.op.input_tensors[0]
    sch = tvm.te.create_schedule(Output.op)

    reg_n, unroll_kw = 2, False
    ic_bn = CI
    oc_bn = KI

    # schedule pad
    batch, ic_chunk, ih, iw, ic_block = sch[padded].op.axis
    parallel_axis = sch[padded].fuse(batch, ic_chunk, ih)
    sch[padded].parallel(parallel_axis)

    # schedule data vec
    batch, ic_chunk, ih, iw, ic_block = sch[data_vec].op.axis
    parallel_axis = sch[data_vec].fuse(batch, ic_chunk, ih)
    sch[data_vec].parallel(parallel_axis)

    oc_chunk, ic_chunk, oh, ow, ic_block, oc_block, _ = sch[kernel_vec].op.axis
    sch[kernel_vec].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
    oc_bn = 2
    if oc_bn > 1:
        sch[kernel_vec].vectorize(oc_block)
    parallel_axis = sch[kernel_vec].fuse(oc_chunk, oh)
    sch[kernel_vec].parallel(parallel_axis)

    # schedule 5-D NCHW[x]c conv
    C, O = Output, Output
    CC = sch.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = sch[C].op.axis
    ow_chunk, ow_block = sch[C].split(ow, factor=reg_n)
    sch[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = sch[C].fuse(batch, oc_chunk, oh)
    sch[C].vectorize(oc_block)
    if C == O:
        sch[C].parallel(parallel_axis)

    sch[CC].compute_at(sch[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = sch[CC].op.axis
    kh, kw, ic_outer, ic_f_inner, ic_s_inner = sch[CC].op.reduce_axis

    ow_chunk, ow_block = sch[CC].split(ow, factor=reg_n)

    oc_f_inner, oc_s_inner = sch[CC].split(oc_block, factor=16)

    if unroll_kw:
        sch[CC].reorder(
            oc_chunk,
            oh,
            ow_chunk,
            ic_outer,
            kh,
            ic_f_inner,
            kw,
            ow_block,
            oc_f_inner,
            oc_s_inner,
            ic_s_inner,
        )
        sch[CC].unroll(kw)
    else:
        sch[CC].reorder(
            oc_chunk,
            oh,
            ow_chunk,
            ic_outer,
            kh,
            kw,
            ic_f_inner,
            ow_block,
            oc_f_inner,
            oc_s_inner,
            ic_s_inner,
        )

    intrin = at.AVX512SkylakeGemv("").get_intrinsic()
    if intrin is not None:
        sch[CC].tensorize(oc_s_inner, intrin)
    sch[CC].unroll(ow_block)
    sch[CC].unroll(oc_f_inner)

    return sch


def run(N, C, H, W, K, R, S, CI, KI, stride, padding, dilation):
    (A, B), (Output,) = conv2d_nchw(N, C, H, W,
                                    K, R, S, CI, KI, stride, padding, dilation)
    sch = schedule_conv2d_nvvi(A, B, Output)
    measure_opt = at.MeasureOptions(
        target_host="llvm -mcpu=skylake-avx512",
        target="llvm -mcpu=skylake-avx512", timeout=10, number=20, min_repeat_ms=150)
    # print(tvm.lower(sch, [A, B, Output], simple_mode=True))
    cost = at.evaluate_schedule(sch, [A, B, Output], measure_opt)
    return cost


def verify(N, C, H, W, K, R, S, CI, KI, stride, padding, dilation):
    import numpy as np
    (A, B), (Output,) = conv2d_nchw(N, C, H, W,
                                    K, R, S, CI, KI, stride, padding, dilation)
    sch1 = schedule_conv2d_nvvi(A, B, Output)
    sch2 = tvm.te.create_schedule(Output.op)
    target = "llvm -mcpu=skylake-avx512"
    func1 = tvm.build(sch1, [A, B, Output], target)
    func2 = tvm.build(sch2, [A, B, Output], target)

    A_np = np.random.uniform(-10, 10, [int(x)
                             for x in A.shape]).astype("uint8")
    B_np = np.random.uniform(-10, 10, [int(x) for x in B.shape]).astype("int8")
    C_np1 = np.random.uniform(-10, 10, [int(x)
                              for x in Output.shape]).astype("int32")
    C_np2 = np.random.uniform(-10, 10, [int(x)
                              for x in Output.shape]).astype("int32")

    ctx = tvm.context(target)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm1 = tvm.nd.array(C_np1, ctx)
    C_tvm2 = tvm.nd.array(C_np2, ctx)
    func1(A_tvm, B_tvm, C_tvm1)
    func2(A_tvm, B_tvm, C_tvm2)
    from tvm import testing
    testing.assert_allclose(C_tvm1.asnumpy(), C_tvm2.asnumpy())


res18_shapes_b1 = [
    # resnet-18
    (1, 16, 24, 24, 32, 16, 3, 3, 1, 2, 1, 1, 1),
    # (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    # (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    # (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    # (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    # (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    # (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    # (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    # (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    # (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    # (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    # (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    # (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]


if __name__ == "__main__":
    batch = 1
    print("N,C,H,W,K,R,S,CI,KI,stride,padding,dilation,cost")
    for shape in res18_shapes_b1:
        N, C, H, W, K, _, R, S, _, stride, padding, dilation, groups = shape
        CI = 4
        KI = 16
        N = batch
        verify(N, C, H, W, K, R, S, CI, KI, stride, padding, dilation)
        # cost = run(N, C, H, W, K, R, S, CI, KI, stride, padding, dilation)
        # print(f"{N},{C},{H},{W},{K},{R},{S},{CI},{KI},{stride},{padding},{dilation},{cost}")
