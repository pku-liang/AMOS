import tvm
import numpy as np
from conv2d_nhwc_implicit_gemm_pro_epilogue import Conv2d
from conv2d_nhwc_implicit_gemm_tensorcore_pro_epilogue import Conv2dTC
from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[:len(prefix)] == prefix
    try:
        number = int(name[len(prefix):])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)
        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix):])


@register_test
def test1():
    R = S = 3
    N = tvm.tir.Var("N", "int32")
    K = tvm.tir.Var("K", "int32")
    H = tvm.tir.Var("H", "int32")
    W = tvm.tir.Var("W", "int32")
    C = tvm.tir.Var("C", "int32")
    # st_h = tvm.tir.Var("st_h", "int32")
    # st_w = tvm.tir.Var("st_w", "int32")
    # ph = tvm.tir.Var("ph", "int32")
    # pw = tvm.tir.Var("pw", "int32")
    # dh = tvm.tir.Var("dh", "int32")
    # dw = tvm.tir.Var("dw", "int32")

    # N = 1
    # K = 1024
    # H = 14
    # W = 14
    # C = 1024
    st_h = 1
    st_w = 1
    ph = 1
    pw = 1
    dh = 1
    dw = 1

    pH = H + 2 * ph
    pW = W + 2 * pw
    dR = (R - 1) * dh + 1
    dS = (S - 1) * dw + 1
    P = (pH - dR) // st_h + 1
    Q = (pW - dS) // st_w + 1

    mN = K
    mM = N * P * Q
    mK = C * R * S

    tb_m = 128
    tb_n = 64
    tb_k = 32
    wp_m = 64
    wp_n = 32
    wp_k = 32
    volta_m = 16
    volta_n = 16
    volta_k = 16
    ptx_m = 8
    ptx_n = 8
    ptx_k = 4

    mMo = (mM + tb_m - 1) // tb_m
    mNo = (mN + tb_n - 1) // tb_n
    mKo = (mK + tb_k - 1) // tb_k
    mMi = (tb_m + wp_m - 1) // wp_m
    mNi = (tb_n + wp_n - 1) // wp_n
    mKi = (tb_k + wp_k - 1) // wp_k
    mMii = (wp_m + volta_m - 1) // volta_m
    mNii = (wp_n + volta_n - 1) // volta_n
    mKii = (wp_k + volta_k - 1) // volta_k
    mMiii = volta_m
    mNiii = volta_n
    mKiii = volta_k

    Image = tvm.te.placeholder([N, H, W, C], dtype="float16", name="Image")
    Filter = tvm.te.placeholder([K, C, R, S], dtype="float16", name="Filter")
    pImage = tvm.te.compute(
        [N, pH, pW, C],
        lambda n, h, w, c:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= ph, h < pH - ph, w >= pw, w < pW - pw),
                Image[n, h-ph, w-pw, c],
                tvm.tir.const(0, Image.dtype)
        ),
        name="pImage")

    def get_mm(mo, mi, mii, miii):
        return mo * tb_m + mi * wp_m + mii * volta_m + miii

    def get_mn(no, ni, nii, niii):
        return no * tb_n + ni * wp_n + nii * volta_n + niii

    def get_mk(ko, ki, kii, kiii):
        return ko * tb_k + ki * wp_k + kii * volta_k + kiii

    def get_n(mm, mk):
        return mm // (P * Q)

    def get_p(mm, mk):
        return mm % (P * Q) // Q

    def get_q(mm, mk):
        return mm % Q

    def get_c(mm, mk):
        return mk // (R * S)

    def get_r(mm, mk):
        return mk % (R * S) // S

    def get_s(mm, mk):
        return mk % S
    A = tvm.te.compute(
        [mMo, mKo, mMi, mKi, mMii, mKii, mMiii, mKiii],
        lambda mo, ko, mi, ki, mii, kii, miii, kiii:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_mm(mo, mi, mii, miii) < mM,
                    get_mk(ko, ki, kii, kiii) < mK
                ),
                pImage[
                    get_n(
                        get_mm(mo, mi, mii, miii),
                        get_mk(ko, ki, kii, kiii)
                    ),
                    get_p(
                        get_mm(mo, mi, mii, miii),
                        get_mk(ko, ki, kii, kiii)
                    ) * dh + get_r(
                                get_mm(mo, mi, mii, miii),
                                get_mk(ko, ki, kii, kiii)) * st_h,
                    get_q(
                        get_mm(mo, mi, mii, miii),
                        get_mk(ko, ki, kii, kiii)
                    ) * dw + get_s(
                                get_mm(mo, mi, mii, miii),
                                get_mk(ko, ki, kii, kiii)) * st_w,
                    get_c(
                        get_mm(mo, mi, mii, miii),
                        get_mk(ko, ki, kii, kiii)
                    )],
                tvm.tir.const(0, Image.dtype)
            ),
        name="A"
    )
    B = tvm.te.compute(
        [mNo, mKo, mNi, mKi, mNii, mKii, mNiii, mKiii],
        lambda no, ko, ni, ki, nii, kii, niii, kiii:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_mn(no, ni, nii, niii) < mN,
                    get_mk(ko, ki, kii, kiii) < mK
                ),
                Filter[
                    get_mn(no, ni, nii, niii),
                    get_c(
                        get_mn(no, ni, nii, niii),
                        get_mk(ko, ki, kii, kiii)),
                    get_r(
                        get_mn(no, ni, nii, niii),
                        get_mk(ko, ki, kii, kiii)),
                    get_s(
                        get_mn(no, ni, nii, niii),
                        get_mk(ko, ki, kii, kiii))
                ],
                tvm.tir.const(0, Filter.dtype)
            ),
        name="B"
    )
    rko = tvm.te.reduce_axis([0, mKo], name="rko")
    rki = tvm.te.reduce_axis([0, mKi], name="rki")
    rkii = tvm.te.reduce_axis([0, mKii], name="rkii")
    rkiii = tvm.te.reduce_axis([0, mKiii], name="rkiii")

    def assemble(n, p, q):
        return n * P * Q + p * Q + q
    def get_ceil(a, b):
        return (a + b - 1) // b

    C = tvm.te.compute(
        [mMo, mNo, mMi, mNi, mMii, mNii, mMiii, mNiii],
        lambda mo, no, mi, ni, mii, nii, miii, niii:
            tvm.te.sum(
                (A[
                    mo,
                    rko,
                    mi,
                    rki,
                    mii,
                    rkii,
                    miii,
                    rkiii] *
                    B[
                        no,
                        rko,
                        ni,
                        rki,
                        nii,
                        rkii,
                        niii,
                        rkiii]).astype("float32"),
                axis=[rko, rki, rkii, rkiii]
        ),
        name="C"
    )

    Output = tvm.te.compute(
        [N, P, Q, K],
        lambda n, p, q, k:
            C[
                get_ceil(assemble(n, p, q), tb_m),
                get_ceil(k, tb_n),
                get_ceil(assemble(n, p, q) % tb_m, wp_m),
                get_ceil(k % tb_n, wp_n),
                get_ceil(assemble(n, p, q) % wp_m, volta_m),
                get_ceil(k % wp_n, volta_n),
                assemble(n, p, q) % volta_m,
                k % volta_n
            ],
        name="Output"
    )

    sch = tvm.te.create_schedule(Output.op)
    print("Initial schedule")
    print(tvm.lower(sch, [Image, Filter, Output], simple_mode=True))
    # AF = sch.cache_read(A, "local", [C])
    # BF = sch.cache_read(B, "local", [C])
    AS = sch.cache_read(A, "shared", [C])
    BS = sch.cache_read(B, "shared", [C])
    AFF = sch.cache_read(AS, "local", [C])
    BFF = sch.cache_read(BS, "local", [C])

    # AF2 = sch.cache_read(A, "local", [C])
    # BF2 = sch.cache_read(B, "local", [C])
    # AS2 = sch.cache_read(AF2, "shared", [C])
    # BS2 = sch.cache_read(BF2, "shared", [C])
    # AFF2 = sch.cache_read(AS2, "local", [C])
    # BFF2 = sch.cache_read(BS2, "local", [C])

    sch[pImage].compute_inline()
    sch[A].compute_inline()
    sch[B].compute_inline()
    sch[C].set_scope("local")

    print("After memory hierarchy")
    print(tvm.lower(sch, [Image, Filter, Output], simple_mode=True))

    def tile_axis(s, op, axis, factors):
        ret = []
        for f in factors:
            outer, axis = s[op].split(axis, factor=f)
            ret.append(outer)
        ret.append(axis)
        return ret

    n, p, q, k = sch[Output].op.axis
    npq = sch[Output].fuse(n, p, q)
    mo, mi, mii, miii = tile_axis(sch, Output, npq, [tb_m, wp_m, volta_m])
    no, ni, nii, niii = tile_axis(sch, Output, k, [tb_n, wp_n, volta_n])
    # mo, mo_ = sch[Output].split(mo, nparts=mMo)
    # no, no_ = sch[Output].split(no, nparts=mNo)

    sch[Output].reorder(mo, no, mi, ni, mii, nii, miii, niii)

    print("After tile Output")
    print(tvm.lower(sch, [Image, Filter, Output], simple_mode=True))

    cmo, cno, cmi, cni, cmii, cnii, cmiii, cniii = sch[C].op.axis
    cko, cki, ckii, ckiii = sch[C].op.reduce_axis
    sch[C].compute_at(sch[Output], ni)
    # sch[C].reorder(cko, cki, n, p, q, k, ckii, ckiii)

    print("After compute at C")
    print(tvm.lower(sch, [Image, Filter, Output], simple_mode=True))
    # sch[AF].compute_at(sch[C], cko)
    # sch[BF].compute_at(sch[C], cko)
    sch[AS].compute_at(sch[C], cko)
    sch[BS].compute_at(sch[C], cko)
    sch[AFF].compute_at(sch[C], cki)
    sch[BFF].compute_at(sch[C], cki)

    print("After compute at cache")
    print(tvm.lower(sch, [Image, Filter, Output], simple_mode=True))


    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")
    sch[Output].bind(mo, by)
    sch[Output].bind(no, bx)
    sch[Output].bind(mi, tz)
    sch[Output].bind(ni, ty)
    # sch[C].bind(cmo, by)
    # sch[C].bind(cno, bx)
    # sch[C].bind(cmi, tz)
    # sch[C].bind(cni, ty)

    # from tvm.te import schedule
    # tmp_sch = sch.normalize()
    # bounds = schedule.InferBound(tmp_sch)
    # print(bounds[no])
    # print(bounds[cno])

    print(tvm.lower(sch, [Image, Filter, Output], simple_mode=True))


@register_test
def test2():
    R = S = 3
    N = tvm.tir.Var("N", "int32")
    K = tvm.tir.Var("K", "int32")
    H = tvm.tir.Var("H", "int32")
    W = tvm.tir.Var("W", "int32")
    C = tvm.tir.Var("C", "int32")
    st_h = tvm.tir.Var("st_h", "int32")
    st_w = tvm.tir.Var("st_w", "int32")
    ph = tvm.tir.Var("ph", "int32")
    pw = tvm.tir.Var("pw", "int32")
    dh = tvm.tir.Var("dh", "int32")
    dw = tvm.tir.Var("dw", "int32")

    pH = H + 2 * ph
    pW = W + 2 * pw
    dR = (R - 1) * dh + 1
    dS = (S - 1) * dw + 1
    P = (pH - dR) // st_h + 1
    Q = (pW - dS) // st_w + 1

    Image = tvm.te.placeholder([N, H, W, C], dtype="float16", name="Image")
    Filter = tvm.te.placeholder([K, C, R, S], dtype="float16", name="Filter")
    pImage = tvm.te.compute(
        [N, pH, pW, C],
        lambda n, h, w, c:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= ph, h < pH - ph, w >= pw, w < pW - pw),
                Image[n, h-ph, w-pw, c],
                tvm.tir.const(0, Image.dtype)
        ),
        name="pImage")
    c = tvm.te.reduce_axis([0, C], "c")
    r = tvm.te.reduce_axis([0, R], "r")
    s = tvm.te.reduce_axis([0, S], "s")
    Output = tvm.te.compute(
        [N, P, Q, K],
        lambda n, p, q, k:
            tvm.te.sum(
                pImage[n, p * st_h + r * dh, q * st_w + s * dw, c] * Filter[k, c, r, s],
            axis=[c, r, s]),
        name="Output"
    )

    (tb_n, tb_p, tb_q, tb_k, tb_c, tb_r, tb_s) = (1, 8, 8, 64, 32, 1, 1)
    (wp_n, wp_p, wp_q, wp_k, wp_c, wp_r, wp_s) = (1, 8, 4, 32, 32, 1, 1)
    (d_n, d_p, d_q, d_k, d_c, d_r, d_s) = (1, 4, 4, 16, 4, 1, 1)


@register_test
def test3():
    N, H, W, C, K, R, S = 1, 7, 7, 512, 512, 3, 3
    pad, stride, dilation = 1, 1, 1
    conv_op = Conv2d([R, S], pad, stride, dilation, target="cuda")
    conv_op.compile([], [], [])
    result = conv_op.test(N, H, W, C, K, min_repeat_ms=500, verify=True)
    print("Time cost is", result, "ms")

@register_test
def test4():
    N, H, W, C, K, R, S = 1, 14, 14, 256, 512, 1, 1
    pad, stride, dilation = 0, 2, 1
    conv_op = Conv2d([R, S], pad, stride, dilation)
    conv_op.compile([], [], [])
    result = conv_op.test(N, H, W, C, K, min_repeat_ms=500)
    print("Time cost is", result, "ms")

@register_test
def test5():
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
    print("N,C,H,W,C,R,S,pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,time(ms)")
    for shape in res18_shapes_b1:
        N, H, W, C, K, R, S = shape[0], shape[2], shape[3], shape[1], shape[4], shape[6], shape[7]
        pad, stride, dilation = shape[10], shape[9], shape[11]
        conv_op = Conv2d([R, S], pad, stride, dilation)
        conv_op.compile([], [], [])
        result = conv_op.test(N, H, W, C, K, min_repeat_ms=500)
        print(",".join([str(x) for x in [
            N, C, H, W, C, R, S,
            pad, pad, stride, stride, dilation, dilation, result
        ]]))


@register_test
def test6():
    N, H, W, C, K, R, S = 1, 7, 7, 512, 512, 3, 3
    pad, stride, dilation = 1, 1, 1
    conv_op = Conv2dTC([R, S], pad, stride, dilation, target="cuda")
    conv_op.compile([], [], [])
    result = conv_op.test(N, H, W, C, K, min_repeat_ms=500, verify=True)
    print("Time cost is", result, "ms")


@register_test
def test7():
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
    print("N,C,H,W,C,R,S,pad_h,pad_w,stride_h,stride_w,dilation_h,dilation_w,time(ms)")
    for shape in res18_shapes_b1:
        N, H, W, C, K, R, S = shape[0], shape[2], shape[3], shape[1], shape[4], shape[6], shape[7]
        pad, stride, dilation = shape[10], shape[9], shape[11]
        conv_op = Conv2dTC([R, S], pad, stride, dilation)
        conv_op.compile([], [], [])
        result = conv_op.test(N, H, W, C, K, min_repeat_ms=500)
        print(",".join([str(x) for x in [
            N, C, H, W, C, R, S,
            pad, pad, stride, stride, dilation, dilation, result
        ]]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="test case", type=int, default=1)
    parser.add_argument("--all", help="test all", action="store_true")

    args = parser.parse_args()
    if args.all:
        for k, v in TEST_CASES.items():
            print("############################################")
            print("test", k)
            v()
            print("Pass!")
    else:
        assert args.case in TEST_CASES, "Can't find case %s." % (
            str(args.case))
        case = TEST_CASES[args.case]
        case()
