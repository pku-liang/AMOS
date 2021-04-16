import tvm
from tvm import te
from ...utils import (
    index,
    multi_index,
    multi_reduce_axis,
    return_conv2d_vars,
    ceil,
    reduce_mul
)


def threadblock_gemm_general(
    Vars,
    threadblock_problem_size,
    warp_problem_size,
    instruction_problem_size,
    epilogues,
    A, B,
    C_dtype="float32"
):
    K, M = A.shape
    _, N = B.shape

    M1, N1, K1 = Vars

    A_vec_L = 128 // tvm.runtime.DataType(A.dtype).bits
    B_vec_L = 128 // tvm.runtime.DataType(B.dtype).bits
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

    def get_m(m1, m2, m3, m4):
        return m1 * M2 * M3 * M4 + m2 * M3 * M4 + m3 * M4 + m4

    def get_n(n1, n2, n3, n4):
        return n1 * N2 * N3 * N4 + n2 * N3 * N4 + n3 * N4 + n4

    def get_k(k1, k2, k3, k4):
        return k1 * K2 * K3 * K4 + k2 * K3 * K4 + k3 * K4 + k4

    def get_ko(k1, k2, k3):
        return k1 * K2 * K3 + k2 * K3 + k3

    A_operand = te.compute(
        [K1, K2, K3, K4, M1, M2, M3, M4],
        lambda k1, k2, k3, k4, m1, m2, m3, m4:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_m(m1, m2, m3, m4) < M,
                    get_ko(k1, k2, k3) < ceil(K, K4)),
                A[get_k(k1, k2, k3, k4), get_m(m1, m2, m3, m4)],
                tvm.tir.const(0, A.dtype)
        ),
        name="A_operand"
    )
    B_operand = te.compute(
        [K1, K2, K3, K4, N1, N2, N3, N4],
        lambda k1, k2, k3, k4, n1, n2, n3, n4:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_n(n1, n2, n3, n4) < N,
                    get_ko(k1, k2, k3) < ceil(K, K4)),
                B[get_k(k1, k2, k3, k4), get_n(n1, n2, n3, n4)],
                tvm.tir.const(0, B.dtype)
        ),
        name="B_operand"
    )

    rk1, rk2, rk3, rk4 = multi_reduce_axis([K1, K2, K3, K4], "rk")
    C = te.compute(
        [M1, M2, M3, M4, N1, N2, N3, N4],
        lambda m1, m2, m3, m4, n1, n2, n3, n4:
            te.sum(
                (A_operand[rk1, rk2, rk3, rk4, m1, m2, m3, m4] *
                 B_operand[rk1, rk2, rk3, rk4, n1, n2, n3, n4]).astype(C_dtype),
                axis=[rk1, rk2, rk3, rk4]
        ),
        name="C"
    )

    Epilogues = []
    Epi = C
    for epi in epilogues:
        Epi = epi(Epi)
        Epilogues.append(Epi)

    def parse_logic_Output_to_physical_Output(*args):
        if len(args) == 4:
            m1, n1, m, n = args
            # m1 = m // (M2 * M3 * M4)
            m2 = m % (M2 * M3 * M4) // (M3 * M4)
            m3 = m % (M3 * M4) // M4
            m4 = m % M4
            # n1 = n // (N2 * N3 * N4)
            n2 = n % (N2 * N3 * N4) // (N3 * N4)
            n3 = n % (N3 * N4) // N4
            n4 = n % N4
            return Epi[m1, m2, m3, m4, n1, n2, n3, n4]
        elif len(args) == 2:
            m, n = args
            m1 = m // (M2 * M3 * M4)
            m2 = m % (M2 * M3 * M4) // (M3 * M4)
            m3 = m % (M3 * M4) // M4
            m4 = m % M4
            n1 = n // (N2 * N3 * N4)
            n2 = n % (N2 * N3 * N4) // (N3 * N4)
            n3 = n % (N3 * N4) // N4
            n4 = n % N4
            return Epi[m1, m2, m3, m4, n1, n2, n3, n4]
        else:
            raise RuntimeError("Invalid args: " + str(args))

    def block(x): return te.thread_axis(f"blockIdx.{x}")
    def thread(x): return te.thread_axis(f"threadIdx.{x}")
    def vthread(): return te.thread_axis("vthread")

    def schedule_threadblock_gemm_thread_map_double_buffer(sch):
        nonlocal C

        BM, BN, BK = M2 * M3 * M4, N2 * N3 * N4, K2 * K3 * K4
        WM, WN = M3 * M4, N3 * N4
        TM, TN = M4, N4

        n_warps_per_block = M2 * N2
        n_threads_per_warp = M4 * N4
        n_threads_per_block = n_threads_per_warp * n_warps_per_block
        assert n_threads_per_warp == 32
        assert (BK * BM) % n_threads_per_block == 0
        assert (BK * BN) % n_threads_per_block == 0

        # build graph
        if len(Epilogues) == 0:
            cache_write = sch.cache_write(C, "local")
            Last = C
            C = cache_write
        else:
            Last = Epilogues[-1]
            sch[C].set_scope("local")
            for epi in Epilogues[:-1]:
                sch[epi].compute_inline()

        s = sch
        CL = C
        AL = sch.cache_read(A_operand, "local", [CL])
        BL = sch.cache_read(B_operand, "local", [CL])
        AA = A_operand
        BB = B_operand
        sch[AA].set_scope("shared")
        sch[BB].set_scope("shared")

        by, bx = (block(x) for x in "yx")
        tz, ty, tx = (thread(x) for x in "zyx")

        m1, m2, m3, m4, n1, n2, n3, n4 = s[Last].op.axis
        s[Last].reorder(m1, n2, m2, n2, m3, n3, m4, n4)
        s[Last].bind(m1, by)
        s[Last].bind(n1, bx)
        s[Last].bind(m2, tz)
        s[Last].bind(n2, ty)
        s[Last].bind(m3, vthread())
        s[Last].bind(n3, vthread())
        warp = s[Last].fuse(m4, n4)
        s[Last].bind(warp, tx)

        s[CL].compute_at(s[Last], warp)
        m1, m2, m3, m4, n1, n2, n3, n4 = s[CL].op.axis
        k1, k2, k3, k4 = s[CL].op.reduce_axis
        k2 = s[C].fuse(k2, k3, k4)
        s[CL].reorder(m1, m2, m3, m4, n1, n2, n3, n4, k1, k2)
        s[CL].unroll(k2)

        s[AL].compute_at(s[CL], k2)
        s[AL].double_buffer()

        s[BL].compute_at(s[CL], k2)
        s[BL].double_buffer()

        def handle_shared(XX, BX):
            s[XX].compute_at(s[CL], k1)
            if BX <= n_threads_per_warp:
                assert n_threads_per_warp % BX == 0
                k0, k, x0, x = s[XX].op.axis
                kv, k = s[XX].split(k, nparts=(BK * BX) // n_threads_per_block)
                ko, ki = s[XX].split(k, nparts=(BM // WM))
                ki, kw = s[XX].split(ki, nparts=(BN // WN))
                s[XX].reorder(k0, x0, kv, ko, ki, kw, x)
                kwx = s[XX].fuse(kw, x)
                s[XX].bind(kv, vthread())
                s[XX].bind(ko, tz)
                s[XX].bind(ki, ty)
                s[XX].bind(kwx, tx)
                s[XX].double_buffer()
            else:
                assert BX % n_threads_per_warp == 0
                k0, k, x0, x = s[XX].op.axis
                kv, k = s[XX].split(k, nparts=(BK * BX) // n_threads_per_block)
                xo, xi = s[XX].split(x, factor=n_threads_per_warp)
                s[XX].reorder(k0, x0, kv, k, xo, xi)
                kxo = s[XX].fuse(k, xo)
                k, xo = s[XX].split(kxo, nparts=(BM // WM))
                s[XX].bind(kv, vthread())
                s[XX].bind(k, tz)
                s[XX].bind(xo, ty)
                s[XX].bind(xi, tx)
                s[XX].double_buffer()

        handle_shared(AA, BM)
        handle_shared(BB, BN)

    return (
        Epi,
        [schedule_threadblock_gemm_thread_map_double_buffer],
        parse_logic_Output_to_physical_Output
    )
