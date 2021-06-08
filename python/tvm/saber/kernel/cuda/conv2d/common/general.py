import tvm
from ...threadblock_implementation_route import (
    get_gemm_implementation_cuda)
from ...utils import index


def kernel_conv2d_nchw_implicit_gemm_general_perfect_common_common(arch, code, tag="double_buffer"):
    def _kernel_conv2d_nchw_implicit_gemm_general_perfect(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A_dtype="float32",
        B_dtype="float32",
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
            [MK, MM],
            lambda k, m:
                padded[m//(P*Q), k//(R*S),
                       m % (P*Q)//Q*stride + k % (R*S)//S*dilation,
                       m % Q*stride + k % S*dilation],
            name="A_M"
        )
        B_M = tvm.te.compute(
            [MK, MN],
            lambda k, n:
                B[n, k//(R*S), k % (R*S)//S, k % S],
            name="B_M"
        )

        (
            Output,
            schedule_func,
            (M1, N1, K1),
            parse_func
        ) = get_gemm_implementation_cuda("general", arch, code, tag)(
            threadblock_problem_size,
            warp_problem_size,
            tensorize_problem_size,
            epilogues,
            A_M, B_M,
            C_dtype=C_dtype
        )

        TBM, TBN, *_ = threadblock_problem_size
        TWM, TWN, *_ = warp_problem_size

        Conv2d = tvm.te.compute(
            [N, K, P, Q],
            lambda n, k, p, q:
                parse_func((n*(P*Q)+p*Q+q), k),
            name="Conv2d"
        )

        def schedule_pro(sch, ctx=None):
            ctx = {} if ctx is None else ctx
            sch[padded].compute_inline()
            sch[A_M].compute_inline()
            sch[B_M].compute_inline()
            return ctx

        def schedule_epi(sch, ctx=None):
            ctx = {} if ctx is None else ctx
            n, k, p, q = sch[Conv2d].op.axis
            fused = sch[Conv2d].fuse(n, k, p, q)
            n_threads_per_warp = 32
            n_warps_per_block = (TBM // TWM) * (TBN // TWN)
            n_threads_per_block = n_threads_per_warp * n_warps_per_block
            fused, threads = sch[Conv2d].split(
                fused, factor=n_threads_per_block)
            sch[Conv2d].bind(fused, tvm.te.thread_axis("blockIdx.x"))
            sch[Conv2d].bind(threads, tvm.te.thread_axis("threadIdx.x"))
            return ctx

        return (
            Conv2d,
            [A, B],
            [schedule_pro] + schedule_func + [schedule_epi],
            Params,
            (M1, N1, K1)
        )

    return _kernel_conv2d_nchw_implicit_gemm_general_perfect


def kernel_conv2d_nhwc_implicit_gemm_general_perfect_common_common(arch, code, tag="double_buffer"):
    def _kernel_conv2d_nhwc_implicit_gemm_general_perfect(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A_dtype="float32",
        B_dtype="float32",
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
            [MK, MM],
            lambda k, m:
                padded[m//(P*Q),
                       m % (P*Q)//Q*stride + k % (R*S)//S*dilation,
                       m % Q*stride + k % S*dilation,
                       k//(R*S)],
            name="A_M"
        )
        B_M = tvm.te.compute(
            [MK, MN],
            lambda k, n:
                B[n, k//(R*S), k % (R*S)//S, k % S],
            name="B_M"
        )

        (
            Output,
            schedule_func,
            (M1, N1, K1),
            parse_func
        ) = get_gemm_implementation_cuda("general", arch, code, tag)(
            threadblock_problem_size,
            warp_problem_size,
            tensorize_problem_size,
            epilogues,
            A_M, B_M,
            C_dtype=C_dtype
        )

        TBM, TBN, *_ = threadblock_problem_size
        TWM, TWN, *_ = warp_problem_size

        Conv2d = tvm.te.compute(
            [N, P, Q, K],
            lambda n, p, q, k:
                parse_func((n*(P*Q)+p*Q+q), k),
            name="Conv2d"
        )

        def schedule_pro(sch, ctx=None):
            sch[padded].compute_inline()
            sch[A_M].compute_inline()
            sch[B_M].compute_inline()

        def schedule_epi(sch, ctx=None):
            n, p, q, k = sch[Conv2d].op.axis
            fused = sch[Conv2d].fuse(n, p, q, k)
            n_threads_per_warp = 32
            n_warps_per_block = (TBM // TWM) * (TBN // TWN)
            n_threads_per_block = n_threads_per_warp * n_warps_per_block
            fused, threads = sch[Conv2d].split(
                fused, factor=n_threads_per_block)
            sch[Conv2d].bind(fused, tvm.te.thread_axis("blockIdx.x"))
            sch[Conv2d].bind(threads, tvm.te.thread_axis("threadIdx.x"))

        return (
            Conv2d,
            [A, B],
            [schedule_pro] + schedule_func + [schedule_epi],
            Params,
            (M1, N1, K1)
        )

    return _kernel_conv2d_nhwc_implicit_gemm_general_perfect
