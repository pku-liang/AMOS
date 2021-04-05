import tvm
from ...threadblock import (
    threadblock_gemm_cuda_tensorcore,
    threadblock_gemm_cuda_tensorcore_split_K)
from ...utils import index


def kernel_conv2d_nchw_implicit_gemm_tensorcore_perfect(
    threadblock_problem_size,
    warp_problem_size,
    tensorize_problem_size,
    epilogues,
    A_dtype="float16",
    B_dtype="float16",
    C_dtype="float32",
    stride=1,
    padding=0,
    dilation=1,
):
    N = index("N")
    C = index("C")
    H = index("H")
    W = index("W")
    K = index("W")
    R = index("R")
    S = index("S")

    Params = [N, C, H, W, K, R, S]

    pH = H + 2 * padding
    pW = W + 2 * padding
    dR = (R - 1) * dilation + 1
    dS = (S - 1) * dilation + 1
    P = (pH - dR) // stride + 1
    Q = (pW - dS) // stride + 1

    MM = N * P * Q
    MN = K
    MK = C * R * S

    A = tvm.te.placeholder([N, C, H, W], dtype=A_dtype, name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype=B_dtype, name="B")
    padded = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, ph, pw:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    ph >= padding, ph < pH - padding,
                    pw >= padding, pw < pW - padding
                ),
                A[n, c, ph - padding, pw - padding],
                tvm.tir.const(0, A.dtype)
            ),
        name="padded"
    )
    A_M = tvm.te.compute(
        [MM, MK],
        lambda m, k:
            padded[m//(P*Q), k//(R*S),
              m%(P*Q)//Q*stride + k%(R*S)//S*dilation,
              m%Q*stride + k%S*dilation],
        name="A_M"
    )
    B_M = tvm.te.compute(
        [MN, MK],
        lambda n, k:
            B[n, k//(R*S), k%(R*S)//S, k%S],
        name="B_M"
    )

    (
        Output,
        schedule_func,
        (M1, N1, K1),
        parse_func
    ) = threadblock_gemm_cuda_tensorcore(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A_M, B_M,
        C_dtype=C_dtype
    )

    TBM = threadblock_problem_size[0]
    TBN =  threadblock_problem_size[1]

    Conv2d = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
            parse_func((n*(P*Q)+p*Q+q), k),
        name="Conv2d"
    )

    def schedule_pro(sch):
        sch[padded].compute_inline()
        sch[A_M].compute_inline()
        sch[B_M].compute_inline()

    def schedule_epi(sch):
        # sch[C_M].compute_inline()
        n, k, p, q = sch[Conv2d].op.axis
        fused = sch[Conv2d].fuse(n, k, p, q)
        num_threads = (
            (threadblock_problem_size[0] // warp_problem_size[0])
            * (threadblock_problem_size[1] // warp_problem_size[1])
        ) * 32
        fused, threads = sch[Conv2d].split(fused, factor=num_threads)
        sch[Conv2d].bind(fused, tvm.te.thread_axis("blockIdx.x"))
        sch[Conv2d].bind(threads, tvm.te.thread_axis("threadIdx.x"))

    return (
        Conv2d,
        [A, B],
        [schedule_pro] + schedule_func + [schedule_epi],
        Params,
        (M1, N1, K1)
    )


def kernel_conv2d_nhwc_implicit_gemm_tensorcore_perfect(
    threadblock_problem_size,
    warp_problem_size,
    tensorize_problem_size,
    epilogues,
    A_dtype="float16",
    B_dtype="float16",
    C_dtype="float32",
    stride=1,
    padding=0,
    dilation=1,
):
    N = index("N")
    C = index("C")
    H = index("H")
    W = index("W")
    K = index("W")
    R = index("R")
    S = index("S")

    Params = [N, C, H, W, K, R, S]

    pH = H + 2 * padding
    pW = W + 2 * padding
    dR = (R - 1) * dilation + 1
    dS = (S - 1) * dilation + 1
    P = (pH - dR) // stride + 1
    Q = (pW - dS) // stride + 1

    MM = N * P * Q
    MN = K
    MK = C * R * S

    A = tvm.te.placeholder([N, H, W, C], dtype=A_dtype, name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype=B_dtype, name="B")
    padded = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, ph, pw:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    ph >= padding, ph < pH - padding,
                    pw >= padding, pw < pW - padding
                ),
                A[n, ph - padding, pw - padding, c],
                tvm.tir.const(0, A.dtype)
            ),
        name="padded"
    )
    A_M = tvm.te.compute(
        [MM, MK],
        lambda m, k:
            padded[m//(P*Q),
              m%(P*Q)//Q*stride + k%(R*S)//S*dilation,
              m%Q*stride + k%S*dilation,
              k//(R*S)],
        name="A_M"
    )
    B_M = tvm.te.compute(
        [MN, MK],
        lambda n, k:
            B[n, k//(R*S), k%(R*S)//S, k%S],
        name="B_M"
    )

    (
        Output,
        schedule_func,
        (M1, N1, K1),
        parse_func
    ) = threadblock_gemm_cuda_tensorcore(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A_M, B_M,
        C_dtype=C_dtype
    )

    TBM = threadblock_problem_size[0]
    TBN =  threadblock_problem_size[1]

    Conv2d = tvm.te.compute(
        [N, P, Q, K],
        lambda n, p, q, k:
            parse_func((n*(P*Q)+p*Q+q), k),
        name="Conv2d"
    )

    def schedule_pro(sch):
        sch[padded].compute_inline()
        sch[A_M].compute_inline()
        sch[B_M].compute_inline()

    def schedule_epi(sch):
        # sch[C_M].compute_inline()
        n, p, q, k = sch[Conv2d].op.axis
        fused = sch[Conv2d].fuse(n, p, q, k)
        num_threads = (
            (threadblock_problem_size[0] // warp_problem_size[0])
            * (threadblock_problem_size[1] // warp_problem_size[1])
        ) * 32
        fused, threads = sch[Conv2d].split(fused, factor=num_threads)
        sch[Conv2d].bind(fused, tvm.te.thread_axis("blockIdx.x"))
        sch[Conv2d].bind(threads, tvm.te.thread_axis("threadIdx.x"))

    return (
        Conv2d,
        [A, B],
        [schedule_pro] + schedule_func + [schedule_epi],
        Params,
        (M1, N1, K1)
    )
