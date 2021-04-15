import tvm
from ...utils import (
    index,
    multi_index,
    multi_reduce_axis,
    return_conv2d_vars,
    ceil,
    reduce_mul
    )


def threadblock_gemm_general_bifrost_g76_single_buffer(
    Vars,
    threadblock_problem_size,
    warp_problem_size,
    instruction_problem_size,
    epilogues,
    A, B,
    C_dtype="float32"
):
    M, K = A.shape
    N, _ = B.shape

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

    A_operand = tvm.te.compute(
        [M1, M2, M3, M4, K1, K2, K3, K4],
        lambda m1, m2, m3, m4, k1, k2, k3, k4:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_m(m1, m2, m3, m4) < M,
                    get_ko(k1, k2, k3) < ceil(K, K4)),
                A[get_m(m1, m2, m3, m4), get_k(k1, k2, k3, k4)],
                tvm.tir.const(0, A.dtype)
            ),
        name="A_operand"
    )
    B_operand = tvm.te.compute(
        [N1, N2, N3, N4, K1, K2, K3, K4],
        lambda n1, n2, n3, n4, k1, k2, k3, k4:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_n(n1, n2, n3, n4) < N,
                    get_ko(k1, k2, k3) < ceil(K, K4)),
                B[get_n(n1, n2, n3, n4), get_k(k1, k2, k3, k4)],
                tvm.tir.const(0, B.dtype)
            ),
        name="B_operand"
    )
    
    rk1, rk2, rk3, rk4 = multi_reduce_axis([K1, K2, K3, K4], "rk")
    C = tvm.te.compute(
        [M1, M2, M3, M4, N1, N2, N3, N4],
        lambda m1, m2, m3, m4, n1, n2, n3, n4:
            tvm.te.sum(
                (A_operand[m1, m2, m3, m4, rk1, rk2, rk3, rk4] *
                B_operand[n1, n2, n3, n4, rk1, rk2, rk3, rk4]).astype(C_dtype),
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
            m, n= args
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

    block_x = lambda *_: tvm.te.thread_axis("blockIdx.x")
    block_y = lambda *_: tvm.te.thread_axis("blockIdx.y")
    block_z = lambda *_: tvm.te.thread_axis("blockIdx.z")
    thread_x = lambda *_: tvm.te.thread_axis("threadIdx.x")
    thread_y = lambda *_: tvm.te.thread_axis("threadIdx.y")
    thread_z = lambda *_: tvm.te.thread_axis("threadIdx.z")

    def schedule_threadblock_gemm(sch, ctx=None):
        ctx = {} if ctx is None else ctx
        nonlocal C
        if "Output" in ctx:
            Output_ctx = ctx["Output"]
            Output = Output_ctx["tensor"]
            Output_axis = Output_ctx["axis"]
            Last = Output
            sch[C].set_scope("local")
            for epi in Epilogues:
                sch[epi].compute_inline()

            m1, n1, m2, m3, n2, n3, m4, n4 = Output_axis
        else:
            if len(Epilogues) == 0:
                cache_write = sch.cache_write(C, "local")
                Last = C
                C = cache_write
            else:
                Last = Epilogues[-1]
                sch[C].set_scope("local")
                for epi in Epilogues[:-1]:
                    sch[epi].compute_inline()

            m1, m2, m3, m4, n1, n2, n3, n4 = sch[Last].op.axis
            sch[Last].reorder(m1, n1, m2, n2, m3, n3, m4, n4)
            sch[Last].bind(m1, block_y())
            sch[Last].bind(n1, block_x())
            sch[Last].bind(m3, thread_y())
            sch[Last].bind(n3, thread_x())
            sch[Last].bind(m2, tvm.te.thread_axis("vthread"))
            sch[Last].bind(n2, tvm.te.thread_axis("vthread"))
            # fused = sch[Last].fuse(m3, n3)
            # fused, threads = sch[Last].split(fused, factor=4)
            # sch[Last].bind(m4, thread_x())
            sch[Last].unroll(m4)
            unroll, vec = sch[Last].split(n4, factor=C_vec_L)
            sch[Last].unroll(unroll)
            sch[Last].vectorize(vec)
        
        AA = sch.cache_read(A_operand, "local", [C])
        BB = sch.cache_read(B_operand, "local", [C])
        sch[A_operand].set_scope("shared")
        sch[B_operand].set_scope("shared")

        sch[C].compute_at(sch[Last], n3)
        m1, m2, m3, m4, n1, n2, n3, n4 = sch[C].op.axis
        rk1, rk2, rk3, rk4 = sch[C].op.reduce_axis
        sch[C].reorder(m1, n1, rk1, m2, n2, rk2, m3, n3, rk3, rk4, m4, n4)
        # sch[C].unroll(rk1)
        # sch[C].unroll(rk2)
        # sch[C].unroll(rk3)
        sch[C].unroll(rk4)
        sch[C].unroll(m4)
        # unroll, vec = sch[C].split(n4, factor=C_vec_L)
        # sch[C].unroll(unroll)
        # sch[C].vectorize(vec)
        sch[C].unroll(n4)

        sch[AA].compute_at(sch[C], rk3)
        sch[BB].compute_at(sch[C], rk3)

        sch[A_operand].compute_at(sch[C], rk1)
        sch[B_operand].compute_at(sch[C], rk1)
        # sch[A_operand].compute_inline()
        # sch[B_operand].compute_inline()
        # sch[AA].compute_inline()
        # sch[BB].compute_inline()

        m1, m2, m3, m4, k1, k2, k3, k4 = sch[AA].op.axis
        sch[AA].unroll(m4)
        # sch[AA].unroll(k4)
        unroll, vec = sch[AA].split(k4, factor=A_vec_L)
        sch[AA].vectorize(vec)
        sch[AA].unroll(unroll)

        n1, n2, n3, n4, k1, k2, k3, k4 = sch[BB].op.axis
        sch[BB].unroll(n4)
        # sch[BB].unroll(k4)
        unroll, vec = sch[BB].split(k4, factor=B_vec_L)
        sch[BB].vectorize(vec)
        sch[BB].unroll(unroll)

        m1, m2, m3, m4, k1, k2, k3, k4 = sch[A_operand].op.axis
        # unroll, vec = sch[A_operand].split(k4, factor=A_vec_L)
        # sch[A_operand].vectorize(vec)
        # sch[A_operand].unroll(unroll)
        # sch[A_operand].bind(m1, block_y())
        # sch[A_operand].bind(k1, block_x())
        sch[A_operand].reorder(m1, k1, m2, m3, m4, k2, k3, k4)
        fused = sch[A_operand].fuse(m2, m3, m4, k2, k3, k4)
        fused, tx = sch[A_operand].split(fused, factor=N3)
        fused, ty = sch[A_operand].split(fused, factor=M3)
        # fused, tz = sch[A_operand].split(fused, factor=M3)
        sch[A_operand].bind(tx, thread_x())
        sch[A_operand].bind(ty, thread_y())
        # sch[A_operand].bind(tz, thread_z())

        n1, n2, n3, n4, k1, k2, k3, k4 = sch[B_operand].op.axis
        # unroll, vec = sch[B_operand].split(k4, factor=B_vec_L)
        # sch[B_operand].vectorize(vec)
        # sch[B_operand].unroll(unroll)
        # sch[B_operand].bind(n1, block_y())
        # sch[B_operand].bind(k1, block_x())
        sch[B_operand].reorder(n1, k1, n2, n3, n4, k2, k3, k4)
        fused = sch[B_operand].fuse(n2, n3, n4, k2, k3, k4)
        fused, tx = sch[B_operand].split(fused, factor=N3)
        fused, ty = sch[B_operand].split(fused, factor=M3)
        # fused, tz = sch[B_operand].split(fused, factor=M3)
        sch[B_operand].bind(tx, thread_x())
        sch[B_operand].bind(ty, thread_y())
        # sch[B_operand].bind(tz, thread_z())

        new_ctx = {}
        new_ctx.update(ctx)
        return new_ctx

    return (
        Epi,
        [schedule_threadblock_gemm],
        parse_logic_Output_to_physical_Output
    )
