import tvm
from ...threadblock_implementation_route import (
    get_gemm_implementation_mali)
from ...utils import index


def kernel_conv2d_nchw_implicit_gemm_general_perfect_bifrost_g71(
    threadblock_problem_size,
    warp_problem_size,
    instruction_problem_size,
    epilogues,
    A_dtype="float32",
    B_dtype="float32",
    C_dtype="float32",
    stride=1,
    padding=0,
    dilation=1,
    tag="single_buffer"
):
    N = index("N")
    C = index("C")
    H = index("H")
    W = index("W")
    K = index("W")
    R = index("R")
    S = index("S")

    M1 = index("M1")
    N1 = index("N1")
    K1 = index("K1")

    A_vec_L = 128 // tvm.runtime.DataType(A_dtype).bits
    B_vec_L = 128 // tvm.runtime.DataType(B_dtype).bits
    C_vec_L = 128 // tvm.runtime.DataType(C_dtype).bits

    M2, N2, K2 = threadblock_problem_size
    M3, N3, K3 = warp_problem_size
    M4, N4, K4 = instruction_problem_size
    assert M2 % M3 == 0
    assert N2 % N3 == 0
    assert K2 % K3 == 0
    assert M3 % M4 == 0
    assert N3 % N4 == 0
    assert K3 % K4 == 0
    M2 = M2 // M3
    N2 = N2 // N3
    K2 = K2 // K3
    M3 = M3 // M4
    N3 = N3 // N4
    K3 = K3 // K4

    block_x = lambda *_: tvm.te.thread_axis("blockIdx.x")
    block_y = lambda *_: tvm.te.thread_axis("blockIdx.y")
    block_z = lambda *_: tvm.te.thread_axis("blockIdx.z")
    thread_x = lambda *_: tvm.te.thread_axis("threadIdx.x")
    thread_y = lambda *_: tvm.te.thread_axis("threadIdx.y")
    thread_z = lambda *_: tvm.te.thread_axis("threadIdx.z")

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
        parse_func
    ) = get_gemm_implementation_mali("general", "bifrost", "g71", tag)(
        [M1, N1, K1],
        threadblock_problem_size,
        warp_problem_size,
        instruction_problem_size,
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

    def schedule_pro(sch, ctx=None):
        ctx = {} if ctx is None else ctx
        sch[padded].compute_inline()
        # sch[A_M].compute_inline()
        # sch[B_M].compute_inline()
        m, k = sch[A_M].op.axis
        m1, mi = sch[A_M].pred_split(m, nparts=M1, factor=M2*M3*M4)
        m2, mi = sch[A_M].split(mi, factor=M3*M4)
        m3, m4 = sch[A_M].split(mi, factor=M4)
        k1, ki = sch[A_M].pred_split(k, nparts=K1, factor=K2*K3*K4)
        k2, ki = sch[A_M].split(ki, factor=K3*K4)
        k3, k4 = sch[A_M].split(ki, factor=K4)
        sch[A_M].reorder(m1, k1, m2, k2, m3, k3, m4, k4)
        sch[A_M].bind(m1, block_y())
        sch[A_M].bind(k1, block_x())
        sch[A_M].bind(m3, thread_z())
        sch[A_M].bind(k3, thread_y())
        sch[A_M].bind(m4, thread_x())
        # sch[A_M].unroll(k4)

        n, k = sch[B_M].op.axis
        n1, ni = sch[B_M].pred_split(n, nparts=N1, factor=N2*N3*N4)
        n2, ni = sch[B_M].split(ni, factor=N3*N4)
        n3, n4 = sch[B_M].split(ni, factor=N4)
        k1, ki = sch[B_M].pred_split(k, nparts=K1, factor=K2*K3*K4)
        k2, ki = sch[B_M].split(ki, factor=K3*K4)
        k3, k4 = sch[B_M].split(ki, factor=K4)
        sch[B_M].reorder(n1, k1, n2, k2, n3, k3, n4, k4)
        sch[B_M].bind(n1, block_y())
        sch[B_M].bind(k1, block_x())
        sch[B_M].bind(n3, thread_z())
        sch[B_M].bind(k3, thread_y())
        sch[B_M].bind(n4, thread_x())
        # sch[B_M].vectorize(k4)
        return ctx

    def schedule_epi(sch, ctx=None):
        # sch[C_M].compute_inline()
        ctx = {} if ctx is None else ctx
        n, k, p, q = sch[Conv2d].op.axis
        sch[Conv2d].reorder(n, p, q, k)
        fused = sch[Conv2d].fuse(n, p, q)
        m1, mi = sch[Conv2d].pred_split(fused, factor=M2*M3*M4, nparts=M1)
        n1, ni = sch[Conv2d].pred_split(k, factor=N2*N3*N4, nparts=N1)
        m2, mi = sch[Conv2d].split(mi, factor=M3*M4)
        m3, m4 = sch[Conv2d].split(mi, factor=M4)
        n2, ni = sch[Conv2d].split(ni, factor=N3*N4)
        n3, n4 = sch[Conv2d].split(ni, factor=N4)
        sch[Conv2d].reorder(m1, n1, m2, n2, m3, n3, m4, n4)
        sch[Conv2d].bind(m1, block_y())
        sch[Conv2d].bind(n1, block_x())
        sch[Conv2d].bind(m3, thread_z())
        sch[Conv2d].bind(n3, thread_y())
        # fused = sch[Conv2d].fuse(m3, n3)
        # fused, threads = sch[Conv2d].split(fused, factor=4)
        sch[Conv2d].bind(m4, thread_x())
        # sch[Conv2d].unroll(m4)
        unroll, vec = sch[Conv2d].split(n4, factor=C_vec_L)
        sch[Conv2d].unroll(unroll)
        sch[Conv2d].vectorize(vec)
        # sch[Conv2d].unroll(n4)
        
        new_ctx = {}
        new_ctx.update(ctx)
        # new_ctx["Output"] = {
        #     "tensor": Conv2d,
        #     "axis": [m1, n1, m2, m3, n2, n3, m4, n4]
        # }
        return new_ctx
    
    def schedule_epi_(sch, ctx=None):
        ctx = {} if ctx is None else ctx
        # sch[C_M].compute_inline()
        n, k, p, q = sch[Conv2d].op.axis
        fused = sch[Conv2d].fuse(n, k, p, q)
        num_threads = (
            (warp_problem_size[0] // instruction_problem_size[0])
            * (warp_problem_size[1] // instruction_problem_size[1])
        ) * instruction_problem_size[0]
        fused, threads = sch[Conv2d].split(fused, factor=num_threads)
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
