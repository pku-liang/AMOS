import tvm
from ...utils import (
    index,
    multi_index,
    multi_reduce_axis,
    return_conv2d_vars,
    ceil,
    reduce_mul
    )


def threadblock_gemm_general(
    threadblock_problem_size,
    warp_problem_size,
    instruction_problem_size,
    epilogues,
    A, B,
    C_dtype="float32"
):
    M, K = A.shape
    N, _ = B.shape

    M1 = index("M1")
    N1 = index("N1")
    K1 = index("K1")

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
        [M1, K1, M2, M3, K2, K3, M4, K4],
        lambda m1, k1, m2, m3, k2, k3, m4, k4:
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
        [N1, K1, N2, N3, K2, K3, N4, K4],
        lambda n1, k1, n2, n3, k2, k3, n4, k4:
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
        [M1, N1, M2, M3, N2, N3, M4, N4],
        lambda m1, n1, m2, m3, n2, n3, m4, n4:
            tvm.te.sum(
                (A_operand[m1, rk1, m2, m3, rk2, rk3, m4, rk4] *
                B_operand[n1, rk1, n2, n3, rk2, rk3, n4, rk4]).astype(C_dtype),
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
            return Epi[m1, n1, m2, m3, n2, n3, m4, n4]
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
            return Epi[m1, n1, m2, m3, n2, n3, m4, n4]
        else:
            raise RuntimeError("Invalid args: " + str(args))

    block_x = lambda *_: tvm.te.thread_axis("blockIdx.x")
    block_y = lambda *_: tvm.te.thread_axis("blockIdx.y")
    block_z = lambda *_: tvm.te.thread_axis("blockIdx.z")
    thread_x = lambda *_: tvm.te.thread_axis("threadIdx.x")
    thread_y = lambda *_: tvm.te.thread_axis("threadIdx.y")
    thread_z = lambda *_: tvm.te.thread_axis("threadIdx.z")

    def schedule_threadblock_gemm(sch):
        nonlocal C
        if len(Epilogues) == 0:
            cache_write = sch.cache_write(C, "local")
            Last = C
            C = cache_write
        else:
            Last = Epilogues[-1]
            sch[C].set_scope("local")
            for epi in Epilogues[:-1]:
                sch[epi].compute_inline()
        sch[A_operand].set_scope("local")
        sch[B_operand].set_scope("local")

        m1, n1, m2, m3, n2, n3, m4, n4 = sch[Last].op.axis
        sch[Last].reorder(m1, n1, m2, n2, m3, n3, m4, n4)
        sch[Last].bind(m1, block_y())
        sch[Last].bind(n1, block_x())
        sch[Last].bind(m3, thread_z())
        sch[Last].bind(n3, thread_y())
        # fused = sch[Last].fuse(m3, n3)
        # fused, threads = sch[Last].split(fused, factor=4)
        sch[Last].bind(m4, thread_x())
        # sch[Last].unroll(m4)
        unroll, vec = sch[Last].split(n4, factor=C_vec_L)
        # sch[Last].unroll(unroll)
        sch[Last].vectorize(vec)

        sch[C].compute_at(sch[Last], m4)
        m1, n1, m2, m3, n2, n3, m4, n4 = sch[C].op.axis
        rk1, rk2, rk3, rk4 = sch[C].op.reduce_axis
        sch[C].reorder(m1, n1, rk1, m2, n2, rk2, rk3, m3, n3, m4, rk4, n4)
        # sch[C].unroll(rk2)
        # sch[C].unroll(rk3)
        # sch[C].unroll(rk4)
        unroll, vec = sch[C].split(n4, factor=C_vec_L)
        # sch[C].unroll(unroll)
        sch[C].vectorize(vec)

        sch[A_operand].compute_at(sch[C], n3)
        sch[B_operand].compute_at(sch[C], n3)

        m1, k1, m2, m3, k2, k3, m4, k4 = sch[A_operand].op.axis
        unroll, vec = sch[A_operand].split(k4, factor=A_vec_L)
        sch[A_operand].vectorize(vec)
        # sch[A_operand].unroll(unroll)

        n1, k1, n2, n3, k2, k3, n4, k4 = sch[B_operand].op.axis
        unroll, vec = sch[B_operand].split(k4, factor=B_vec_L)
        sch[B_operand].vectorize(vec)
        # sch[B_operand].unroll(unroll)


    Vars = [M1, N1, K1]
    return (
        Epi,
        [schedule_threadblock_gemm],
        Vars,
        parse_logic_Output_to_physical_Output
    )


def threadblock_gemm_general_split_K(
    threadblock_problem_size,
    warp_problem_size,
    instruction_problem_size,
    epilogues,
    A, B,
    split_K = 8,
    C_dtype="float32"
):
    M, K = A.shape
    N, _ = B.shape

    M1 = index("M1")
    N1 = index("N1")
    K1 = index("K1")

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

    A_vec_L = 128 // tvm.runtime.DataType(A.dtype).bits
    B_vec_L = 128 // tvm.runtime.DataType(B.dtype).bits
    C_vec_L = 128 // tvm.runtime.DataType(C_dtype).bits

    assert K2 % split_K == 0
    K2 = K2 // split_K

    def get_m(m1, m2, m3, m4):
        return m1 * M2 * M3 * M4 + m2 * M3 * M4 + m3 * M4 + m4

    def get_n(n1, n2, n3, n4):
        return n1 * N2 * N3 * N4 + n2 * N3 * N4 + n3 * N4 + n4

    def get_k(k1, k2, k3, k4):
        return k1 * split_K * K2 * K3 * K4 + k2 * K3 * K4 + k3 * K4 + k4

    A_operand = tvm.te.compute(
        [M1, K1, M2, M3, split_K, K2, K3, M4, K4],
        lambda m1, k1, m2, m3, sp, k2, k3, m4, k4:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_m(m1, m2, m3, m4) < M,
                    get_k(k1, sp * K2 + k2, k3, k4) < K),
                A[get_m(m1, m2, m3, m4), get_k(k1, sp * K2 + k2, k3, k4)],
                tvm.tir.const(0, A.dtype)
            ),
        name="A_operand"
    )
    B_operand = tvm.te.compute(
        [N1, K1, N2, N3, split_K, K2, K3, N4, K4],
        lambda n1, k1, n2, n3, sp, k2, k3, n4, k4:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_n(n1, n2, n3, n4) < N,
                    get_k(k1, sp * K2 + k2, k3, k4) < K),
                B[get_n(n1, n2, n3, n4), get_k(k1, sp * K2 + k2, k3, k4)],
                tvm.tir.const(0, B.dtype)
            ),
        name="B_operand"
    )
    
    rk1, split, rk2, rk3, rk4 = multi_reduce_axis([K1, split_K, K2, K3, K4], "rk")
    C_split = tvm.te.compute(
        [M1, N1, split_K, M2, M3, N2, N3, M4, N4],
        lambda m1, n1, sp, m2, m3, n2, n3, m4, n4:
            tvm.te.sum(
                (A_operand[m1, rk1, m2, m3, sp, rk2, rk3, m4, rk4] *
                B_operand[n1, rk1, n2, n3, sp, rk2, rk3, n4, rk4]).astype(C_dtype),
                axis=[rk1, rk2, rk3, rk4]
            ),
        name="C_split"
    )

    C = tvm.te.compute(
        [M1, N1, M2, M3, N2, N3, M4, N4],
        lambda m1, n1, m2, m3, n2, n3, m4, n4:
            tvm.te.sum(
                C_split[m1, n1, split, m2, m3, n2, n3, m4, n4],
                axis=[split]
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
            return Epi[m1, n1, m2, m3, n2, n3, m4, n4]
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
            return Epi[m1, n1, m2, m3, n2, n3, m4, n4]
        else:
            raise RuntimeError("Invalid args: " + str(args))

    block_x = lambda *_: tvm.te.thread_axis("blockIdx.x")
    block_y = lambda *_: tvm.te.thread_axis("blockIdx.y")
    block_z = lambda *_: tvm.te.thread_axis("blockIdx.z")
    thread_x = lambda *_: tvm.te.thread_axis("threadIdx.x")
    thread_y = lambda *_: tvm.te.thread_axis("threadIdx.y")
    thread_z = lambda *_: tvm.te.thread_axis("threadIdx.z")

    def schedule_threadblock_gemm_split_K(sch):
        nonlocal C
        if len(Epilogues) == 0:
            cache_write = sch.cache_write(C, "local")
            Last = C
            C = cache_write
        else:
            Last = Epilogues[-1]
            sch[C].set_scope("local")
            for epi in Epilogues[:-1]:
                sch[epi].compute_inline()
        AA = sch.cache_read(A_operand, "local", [C_split])
        BB = sch.cache_read(B_operand, "local", [C_split])
        sch[A_operand].set_scope("shared")
        sch[B_operand].set_scope("shared")
        C_split_register = sch.cache_write(C_split, "local")
        sch[C_split].set_scope("shared")

        m1, n1, m2, m3, n2, n3, m4, n4 = sch[Last].op.axis
        sch[Last].reorder(m1, n1, m2, n2, m3, n3, m4, n4)
        sch[Last].bind(m1, block_y())
        sch[Last].bind(n1, block_x())
        fused = sch[Last].fuse(m4, n4)
        fused, ty = sch[Last].split(fused,factor=split_K)
        fused = sch[Last].fuse(m2, n2)
        sch[Last].bind(fused, thread_z())
        sch[Last].bind(ty, thread_y())

        sch[C].compute_at(sch[Last], fused)
        sch[C_split].compute_at(sch[Last], n1)
        m1, n1, m2, m3, n2, n3, m4, n4 = sch[C].op.axis
        k1, = sch[C].op.reduce_axis
        sch[C].reorder(m1, n1, k1, m2, n2, m3, n3, m4, n4)

        recipe = get_tensorcore_recipe([str(A.dtype), str(B.dtype)], C_dtype)
        compute_key = "ntn"
        shape_key = "x".join([str(x) for x in instruction_problem_size])
        load_a = recipe.get_intrinsic(compute_key, shape_key, "load_a")
        load_b = recipe.get_intrinsic(compute_key, shape_key, "load_b")
        store = recipe.get_intrinsic(compute_key, shape_key, "store", store_scope="shared")
        mma = recipe.get_intrinsic(compute_key, shape_key, "mma")

        m1, n1, sp, m2, m3, n2, n3, m4, n4 = sch[C_split].op.axis
        sch[C_split].reorder(m1, n1, m2, n2, sp, m3, n3, m4, n4)
        fused = sch[C_split].fuse(m2, n2)
        sch[C_split].bind(fused, thread_z())
        sch[C_split].bind(sp, thread_y())
        sch[C_split].tensorize(m4, store)

        sch[C_split_register].compute_at(sch[C_split], sp)
        m1, n1, sp, m2, m3, n2, n3, m4, n4 = sch[C_split_register].op.axis
        k1, k2, k3, k4 = sch[C_split_register].op.reduce_axis
        sch[C_split_register].reorder(m1, n1, k1, m2, n2, sp, k2, m3, n3, k3, m4, n4, k4)
        sch[C_split_register].tensorize(m4, mma)

        sch[AA].compute_at(sch[C_split_register], k2)
        sch[BB].compute_at(sch[C_split_register], k2)
        sch[A_operand].compute_at(sch[C_split_register], k1)
        sch[B_operand].compute_at(sch[C_split_register], k1)
        m1, k1, m2, m3, sp, k2, k3, m4, k4 = sch[AA].op.axis
        sch[AA].tensorize(m4, load_a)
        n1, k1, n2, n3, sp, k2, k3, n4, k4 = sch[BB].op.axis
        sch[BB].tensorize(n4, load_b)

        m1, k1, m2, m3, sp, k2, k3, m4, k4 = sch[A_operand].op.axis
        fused = sch[A_operand].fuse(k2, k3, m4, k4)
        fused, vec = sch[A_operand].split(fused, factor=A_vec_L)
        fused, tx = sch[A_operand].split(fused, factor=32)
        fused = sch[A_operand].fuse(m2, m3)
        fused, tz = sch[A_operand].split(fused, factor=M2*N2)
        sch[A_operand].vectorize(vec)
        sch[A_operand].bind(tx, thread_x())
        sch[A_operand].bind(sp, thread_y())
        sch[A_operand].bind(tz, thread_z())

        n1, k1, n2, n3, sp, k2, k3, n4, k4 = sch[B_operand].op.axis
        fused = sch[B_operand].fuse(k2, k3, n4, k4)
        fused, vec = sch[B_operand].split(fused, factor=B_vec_L)
        fused, tx = sch[B_operand].split(fused, factor=32)
        fused = sch[B_operand].fuse(n2, n3)
        fused, tz = sch[B_operand].split(fused, factor=M2*N2)
        sch[B_operand].vectorize(vec)
        sch[B_operand].bind(tx, thread_x())
        sch[B_operand].bind(sp, thread_y())
        sch[B_operand].bind(tz, thread_z())

    Vars = [M1, N1, K1]
    return (
        Epi,
        [schedule_threadblock_gemm_split_K],
        Vars,
        parse_logic_Output_to_physical_Output
    )
