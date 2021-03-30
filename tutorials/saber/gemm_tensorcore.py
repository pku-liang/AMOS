import tvm
from tvm import auto_tensorize as at
from functools import reduce


def index(name="tmp"):
    return tvm.tir.Var(name, "int32")


def multi_index(num, name="tmp"):
    return [tvm.tir.Var(name + str(i)) for i in range(num)]


def multi_reduce_axis(extents, name="tmp"):
    return [tvm.te.reduce_axis(
        [0, extents[i]], name + str(i)) for i in range(len(extents))]


def return_conv2d_vars(N, K, H, W, C, R, S):
    return [N, K, H, W, C, R, S]


def ceil(a, b):
    return (a + b - 1) // b


def reduce_mul(lst):
    return reduce(lambda i, j: i * j, lst, 1)


def reduce_add(lst):
    return reduce(lambda i, j: i + j, lst, 0)


def tile_axes_outer_to_inner(sch, op, axis, factors):
    ret = []
    for f in factors:
        outer, axis = sch[op].split(axis, factor=f)
        ret.append(outer)
    ret.append(axis)
    return ret


def get_recipe(in_dtypes, out_dtype):
    assert len(in_dtypes) == 2 and in_dtypes[0] == in_dtypes[1]
    if in_dtypes[0] == "float16":
        if out_dtype == "float16":
            return at.WMMAFp16Fp16()
        elif out_dtype == "float32":
            return at.WMMAFp16Fp32()
    if in_dtypes[0] == "int1":
        if out_dtype == "int32":
            return at.WMMABin1Int32()
    if in_dtypes[0] == "int4":
        if out_dtype == "int32":
            return at.WMMAInt4Int32()
    if in_dtypes[0] == "int8":
        if out_dtype == "int32":
            return at.WMMAInt8Int32()
    if in_dtypes[0] == "bfloat16":
        if out_dtype == "float32":
            return at.WMMABf16Fp32()
    if in_dtypes[0] == "float32":
        if out_dtype == "float32":
            return at.WMMATf32Fp32()
    if in_dtypes[0] == "float64":
        if out_dtype == "float64":
            return at.WMMAFp64Fp64()
    raise RuntimeError("Invalid dtype for tensor core: input:" + in_dtypes[0] + ", output:" + out_dtype)


def threadblock_gemm(
    threadblock_problem_size,
    warp_problem_size,
    tensorize_problem_size,
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
    M4, N4, K4 = tensorize_problem_size
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

    A_operand = tvm.te.compute(
        [M1, K1, M2, M3, K2, K3, M4, K4],
        lambda m1, k1, m2, m3, k2, k3, m4, k4:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_m(m1, m2, m3, m4) < M,
                    get_k(k1, k2, k3, k4) < K),
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
                    get_k(k1, k2, k3, k4) < K),
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
        AA = sch.cache_read(A_operand, "local", [C])
        BB = sch.cache_read(B_operand, "local", [C])
        sch[A_operand].set_scope("shared")
        sch[B_operand].set_scope("shared")

        recipe = get_recipe([str(A.dtype), str(B.dtype)], C_dtype)
        compute_key = "ntn"
        shape_key = "x".join([str(x) for x in tensorize_problem_size])
        load_a = recipe.get_intrinsic(compute_key, shape_key, "load_a")
        load_b = recipe.get_intrinsic(compute_key, shape_key, "load_b")
        store = recipe.get_intrinsic(compute_key, shape_key, "store", store_scope="global")
        mma = recipe.get_intrinsic(compute_key, shape_key, "mma")

        m1, n1, m2, m3, n2, n3, m4, n4 = sch[Last].op.axis
        sch[Last].reorder(m1, n1, m2, n2, m3, n3, m4, n4)
        sch[Last].bind(m1, block_y())
        sch[Last].bind(n1, block_x())
        sch[Last].bind(m2, thread_z())
        sch[Last].bind(n2, thread_y())
        sch[Last].tensorize(m4, store)

        sch[C].compute_at(sch[Last], n2)
        m1, n1, m2, m3, n2, n3, m4, n4 = sch[C].op.axis
        rk1, rk2, rk3, rk4 = sch[C].op.reduce_axis
        sch[C].reorder(m1, n1, rk1, m2, n2, rk2, rk3, m3, n3, m4, n4, rk4)
        sch[C].unroll(rk2)
        sch[C].unroll(rk3)
        sch[C].tensorize(m4, mma)

        sch[AA].compute_at(sch[C], rk3)
        sch[BB].compute_at(sch[C], rk3)
        sch[A_operand].compute_at(sch[C], rk1)
        sch[B_operand].compute_at(sch[C], rk1)
        m1, k1, m2, m3, k2, k3, m4, k4 = sch[AA].op.axis
        sch[AA].tensorize(m4, load_a)
        n1, k1, n2, n3, k2, k3, n4, k4 = sch[BB].op.axis
        sch[BB].tensorize(n4, load_b)
        m1, k1, m2, m3, k2, k3, m4, k4 = sch[A_operand].op.axis
        fused = sch[A_operand].fuse(m2, m3, k2, k3, m4, k4)
        fused, vec = sch[A_operand].split(fused, factor=A_vec_L)
        fused, tx = sch[A_operand].split(fused, factor=32)
        fused, ty = sch[A_operand].split(fused, factor=N2)
        fused, tz = sch[A_operand].split(fused, factor=M2)
        sch[A_operand].vectorize(vec)
        sch[A_operand].bind(tx, thread_x())
        sch[A_operand].bind(ty, thread_y())
        sch[A_operand].bind(tz, thread_z())

        n1, k1, n2, n3, k2, k3, n4, k4 = sch[B_operand].op.axis
        fused = sch[B_operand].fuse(n2, n3, k2, k3, n4, k4)
        fused, vec = sch[B_operand].split(fused, factor=B_vec_L)
        fused, tx = sch[B_operand].split(fused, factor=32)
        fused, ty = sch[B_operand].split(fused, factor=N2)
        fused, tz = sch[B_operand].split(fused, factor=M2)
        sch[B_operand].vectorize(vec)
        sch[B_operand].bind(tx, thread_x())
        sch[B_operand].bind(ty, thread_y())
        sch[B_operand].bind(tz, thread_z())


    Vars = [M1, N1, K1]
    return (
        Epi,
        [schedule_threadblock_gemm],
        Vars,
        parse_logic_Output_to_physical_Output
    )


def threadblock_gemm_split_K(
    threadblock_problem_size,
    warp_problem_size,
    tensorize_problem_size,
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
    M4, N4, K4 = tensorize_problem_size

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

        recipe = get_recipe([str(A.dtype), str(B.dtype)], C_dtype)
        compute_key = "ntn"
        shape_key = "x".join([str(x) for x in tensorize_problem_size])
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


def kernel_gemm(
    threadblock_problem_size,
    warp_problem_size,
    tensorize_problem_size,
    epilogues,
    A_dtype="float16",
    B_dtype="float16",
    C_dtype="float32"
):
    M = index("M")
    N = index("N")
    K = index("K")
    Params = [M, N, K]
    A = tvm.te.placeholder([M, K], dtype=A_dtype, name="A")
    B = tvm.te.placeholder([N, K], dtype=B_dtype, name="B")
    (
        Output,
        schedule_func,
        (M1, N1, K1),
        parse_func
    ) = threadblock_gemm(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A, B,
        C_dtype=C_dtype
    )

    Gemm = tvm.te.compute(
        [M1, threadblock_problem_size[0], N1, threadblock_problem_size[1]],
        lambda m1, m, n1, n:
            parse_func(m1, n1, m, n),
        name="Gemm"
    )

    def schedule_kernel(sch):
        m1, m, n1, n = sch[Gemm].op.axis
        num_threads = (
            (threadblock_problem_size[0] // warp_problem_size[0])
            * (threadblock_problem_size[1] // warp_problem_size[1])
        ) * 32
        sch[Gemm].reorder(m1, n1, m, n)
        fused = sch[Gemm].fuse(m, n)
        fused, threads = sch[Gemm].split(fused, factor=num_threads)
        sch[Gemm].bind(threads, tvm.te.thread_axis("threadIdx.x"))
        sch[Gemm].bind(m1, tvm.te.thread_axis("blockIdx.y"))
        sch[Gemm].bind(n1, tvm.te.thread_axis("blockIdx.x"))

    return (
        Gemm,
        [A, B],
        schedule_func + [schedule_kernel],
        Params,
        (M1, N1, K1)
    )


def kernel_gemm_perfect(
    threadblock_problem_size,
    warp_problem_size,
    tensorize_problem_size,
    epilogues,
    A_dtype="float16",
    B_dtype="float16",
    C_dtype="float32"
):
    M = index("M")
    N = index("N")
    K = index("K")
    Params = [M, N, K]
    A = tvm.te.placeholder([M, K], dtype=A_dtype, name="A")
    B = tvm.te.placeholder([N, K], dtype=B_dtype, name="B")
    (
        Output,
        schedule_func,
        (M1, N1, K1),
        parse_func
    ) = threadblock_gemm(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A, B,
        C_dtype=C_dtype
    )

    Gemm = tvm.te.compute(
        [M, N],
        lambda m, n:
            parse_func(m, n),
        name="Gemm"
    )

    def schedule_kernel(sch):
        m, n = sch[Gemm].op.axis
        num_threads = (
            (threadblock_problem_size[0] // warp_problem_size[0])
            * (threadblock_problem_size[1] // warp_problem_size[1])
        ) * 32
        fused = sch[Gemm].fuse(m, n)
        fused, threads = sch[Gemm].split(fused, factor=num_threads)
        sch[Gemm].bind(threads, tvm.te.thread_axis("threadIdx.x"))
        sch[Gemm].bind(fused, tvm.te.thread_axis("blockIdx.x"))

    return (
        Gemm,
        [A, B],
        schedule_func + [schedule_kernel],
        Params,
        (M1, N1, K1)
    )


def kernel_gemm_split_K(
    threadblock_problem_size,
    warp_problem_size,
    tensorize_problem_size,
    epilogues,
    split_K = 8,
    A_dtype="float16",
    B_dtype="float16",
    C_dtype="float32"
):
    M = index("M")
    N = index("N")
    K = index("K")
    Params = [M, N, K]
    A = tvm.te.placeholder([M, K], dtype=A_dtype, name="A")
    B = tvm.te.placeholder([N, K], dtype=B_dtype, name="B")
    (
        Output,
        schedule_func,
        (M1, N1, K1),
        parse_func
    ) = threadblock_gemm_split_K(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A, B,
        split_K=split_K,
        C_dtype=C_dtype
    )

    Gemm = tvm.te.compute(
        [M1, threadblock_problem_size[0], N1, threadblock_problem_size[1]],
        lambda m1, m, n1, n:
            parse_func(m1, n1, m, n),
        name="Gemm"
    )

    def schedule_kernel(sch):
        m1, m, n1, n = sch[Gemm].op.axis
        num_threads = (
            (threadblock_problem_size[0] // warp_problem_size[0])
            * (threadblock_problem_size[1] // warp_problem_size[1])
        ) * 32
        sch[Gemm].reorder(m1, n1, m, n)
        fused = sch[Gemm].fuse(m, n)
        fused, threads = sch[Gemm].split(fused, factor=num_threads)
        sch[Gemm].bind(threads, tvm.te.thread_axis("threadIdx.x"))
        sch[Gemm].bind(m1, tvm.te.thread_axis("blockIdx.y"))
        sch[Gemm].bind(n1, tvm.te.thread_axis("blockIdx.x"))

    return (
        Gemm,
        [A, B],
        schedule_func + [schedule_kernel],
        Params,
        (M1, N1, K1)
    )


def kernel_gemm_split_K_perfect(
    threadblock_problem_size,
    warp_problem_size,
    tensorize_problem_size,
    epilogues,
    split_K = 8,
    A_dtype="float16",
    B_dtype="float16",
    C_dtype="float32"
):
    M = index("M")
    N = index("N")
    K = index("K")
    Params = [M, N, K]
    A = tvm.te.placeholder([M, K], dtype=A_dtype, name="A")
    B = tvm.te.placeholder([N, K], dtype=B_dtype, name="B")
    (
        Output,
        schedule_func,
        (M1, N1, K1),
        parse_func
    ) = threadblock_gemm_split_K(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A, B,
        split_K=split_K,
        C_dtype=C_dtype
    )

    Gemm = tvm.te.compute(
        [M, N],
        lambda m, n:
            parse_func(m, n),
        name="Gemm"
    )

    def schedule_kernel(sch):
        m, n = sch[Gemm].op.axis
        num_threads = (
            (threadblock_problem_size[0] // warp_problem_size[0])
            * (threadblock_problem_size[1] // warp_problem_size[1])
        ) * 32
        fused = sch[Gemm].fuse(m, n)
        fused, threads = sch[Gemm].split(fused, factor=num_threads)
        sch[Gemm].bind(threads, tvm.te.thread_axis("threadIdx.x"))
        sch[Gemm].bind(fused, tvm.te.thread_axis("blockIdx.x"))

    return (
        Gemm,
        [A, B],
        schedule_func + [schedule_kernel],
        Params,
        (M1, N1, K1)
    )


def kernel_conv2d_nchw_implicit_gemm_perfect(
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
    ) = threadblock_gemm(
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


def kernel_conv2d_nhwc_implicit_gemm_perfect(
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
    ) = threadblock_gemm(
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


def run_gemm(
        M, N, K, in_dtype="float16", out_dtype="float32", target="llvm", verify=True, dump=False):
    (
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size
    )    = (
        [64, 64, 16],
        [16, 32, 16],
        [16, 16, 16]
    )

    epilogues = []

    (
        Output,
        (A, B),
        schedule_func,
        Params,
        Vars
    ) = kernel_gemm(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A_dtype=in_dtype,
        B_dtype=in_dtype,
        C_dtype=out_dtype
    )
    sch = tvm.te.create_schedule(Output.op)
    for func in schedule_func:
        func(sch)

    ctx = tvm.context(target)
    import numpy as np
    if dump:
        print(tvm.lower(
            sch, [A, B, Output, *Params, *Vars],
            simple_mode=True
        ))

    gemm_func = tvm.build(
        sch, [A, B, Output, *Params, *Vars], target=target)

    params = [M, N, K]
    vars_ = [
        ceil(M, threadblock_problem_size[0]),
        ceil(N, threadblock_problem_size[1]),
        ceil(K, threadblock_problem_size[2])]

    A_np = np.random.uniform(-1, 1, [M, K])
    B_np = np.random.uniform(-1, 1, [N, K])
    Output_np = np.zeros(
        [vars_[0], threadblock_problem_size[0],
         vars_[1], threadblock_problem_size[1]],
        # [
        #     vars_[0], vars_[1],
        #     threadblock_problem_size[0] // warp_problem_size[0],
        #     warp_problem_size[0] // tensorize_problem_size[0],
        #     threadblock_problem_size[1] // warp_problem_size[1],
        #     warp_problem_size[1] // tensorize_problem_size[1],
        #     tensorize_problem_size[0],
        #     tensorize_problem_size[1]
        # ],
        dtype=Output.dtype)
    A_tvm = tvm.nd.array(A_np.astype(A.dtype), ctx)
    B_tvm = tvm.nd.array(B_np.astype(B.dtype), ctx)
    Output_tvm = tvm.nd.array(Output_np, ctx)

    gemm_func(A_tvm, B_tvm, Output_tvm,
                *params, *vars_)

    if verify:
        import torch
        if torch.cuda.is_available():
            A_torch = torch.tensor(A_np).to("cuda")
            B_torch = torch.tensor(B_np).to("cuda")
        else:
            A_torch = torch.tensor(A_np)
            B_torch = torch.tensor(B_np)
        Output_torch = torch.mm(
            A_torch.type(torch.float16),
            B_torch.type(torch.float16).permute(1, 0))
        _Output_tvm = Output_tvm.asnumpy()
        _Output_tvm = _Output_tvm.reshape(
            _Output_tvm.shape[0] * _Output_tvm.shape[1],
            _Output_tvm.shape[2] * _Output_tvm.shape[3])
        _Output_tvm = _Output_tvm[:M, :N]
        from tvm import testing
        testing.assert_allclose(
            Output_torch.cpu().numpy(), _Output_tvm,
            rtol=1e-1, atol=1e-1)

    timed_func = gemm_func.time_evaluator(
        gemm_func.entry_name, ctx, number=20, min_repeat_ms=500)
    cost = timed_func(A_tvm, B_tvm, Output_tvm, *params, *vars_).mean
    print(",".join(["Gemm", in_dtype, out_dtype] + [str(x) for x in [
        M, N, K, cost * 1e3
    ]]))


def run_gemm_perfect(
        M, N, K, in_dtype="float16", out_dtype="float32", target="llvm", verify=True, dump=False):
    (
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size
    )    = (
        [64, 64, 16],
        [16, 32, 16],
        [16, 16, 16]
    )

    epilogues = []

    (
        Output,
        (A, B),
        schedule_func,
        Params,
        Vars
    ) = kernel_gemm_perfect(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A_dtype=in_dtype,
        B_dtype=in_dtype,
        C_dtype=out_dtype
    )
    sch = tvm.te.create_schedule(Output.op)
    for func in schedule_func:
        func(sch)

    ctx = tvm.context(target)
    import numpy as np
    if dump:
        print(tvm.lower(
            sch, [A, B, Output, *Params, *Vars],
            simple_mode=True
        ))

    gemm_func = tvm.build(
        sch, [A, B, Output, *Params, *Vars], target=target)

    params = [M, N, K]
    vars_ = [
        ceil(M, threadblock_problem_size[0]),
        ceil(N, threadblock_problem_size[1]),
        ceil(K, threadblock_problem_size[2])]

    A_np = np.random.uniform(-1, 1, [M, K])
    B_np = np.random.uniform(-1, 1, [N, K])
    Output_np = np.zeros(
        [M, N],
        dtype=Output.dtype)
    A_tvm = tvm.nd.array(A_np.astype(A.dtype), ctx)
    B_tvm = tvm.nd.array(B_np.astype(B.dtype), ctx)
    Output_tvm = tvm.nd.array(Output_np, ctx)

    gemm_func(A_tvm, B_tvm, Output_tvm,
                *params, *vars_)

    if verify:
        import torch
        if torch.cuda.is_available():
            A_torch = torch.tensor(A_np).to("cuda")
            B_torch = torch.tensor(B_np).to("cuda")
        else:
            A_torch = torch.tensor(A_np)
            B_torch = torch.tensor(B_np)
        Output_torch = torch.mm(
            A_torch.type(torch.float16),
            B_torch.type(torch.float16).permute(1, 0))
        _Output_tvm = Output_tvm.asnumpy()
        _Output_tvm = _Output_tvm[:M, :N]
        from tvm import testing
        testing.assert_allclose(
            Output_torch.cpu().numpy(), _Output_tvm,
            rtol=1e-1, atol=1e-1)

    timed_func = gemm_func.time_evaluator(
        gemm_func.entry_name, ctx, number=20, min_repeat_ms=500)
    cost = timed_func(A_tvm, B_tvm, Output_tvm, *params, *vars_).mean
    print(",".join(["Gemm_perfect", in_dtype, out_dtype] + [str(x) for x in [
        M, N, K, cost * 1e3
    ]]))


def run_gemm_split_K(
        M, N, K, in_dtype="float16", out_dtype="float32", target="llvm", verify=True, dump=False):
    (
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size
    )    = (
        [64, 64, 16],
        [16, 32, 16],
        [16, 16, 16]
    )

    epilogues = []

    (
        Output,
        (A, B),
        schedule_func,
        Params,
        Vars
    ) = kernel_gemm_split_K(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        split_K=1,
        A_dtype=in_dtype,
        B_dtype=in_dtype,
        C_dtype=out_dtype
    )
    sch = tvm.te.create_schedule(Output.op)
    for func in schedule_func:
        func(sch)

    ctx = tvm.context(target)
    import numpy as np
    if dump:
        print(tvm.lower(
            sch, [A, B, Output, *Params, *Vars],
            simple_mode=True
        ))

    gemm_func = tvm.build(
        sch, [A, B, Output, *Params, *Vars], target=target)

    params = [M, N, K]
    vars_ = [
        ceil(M, threadblock_problem_size[0]),
        ceil(N, threadblock_problem_size[1]),
        ceil(K, threadblock_problem_size[2])]

    A_np = np.random.uniform(-1, 1, [M, K])
    B_np = np.random.uniform(-1, 1, [N, K])
    Output_np = np.zeros(
        [vars_[0], threadblock_problem_size[0],
         vars_[1], threadblock_problem_size[1]],
        dtype=Output.dtype)
    A_tvm = tvm.nd.array(A_np.astype(A.dtype), ctx)
    B_tvm = tvm.nd.array(B_np.astype(B.dtype), ctx)
    Output_tvm = tvm.nd.array(Output_np, ctx)

    gemm_func(A_tvm, B_tvm, Output_tvm,
                *params, *vars_)

    if verify:
        import torch
        if torch.cuda.is_available():
            A_torch = torch.tensor(A_np).to("cuda")
            B_torch = torch.tensor(B_np).to("cuda")
        else:
            A_torch = torch.tensor(A_np)
            B_torch = torch.tensor(B_np)
        Output_torch = torch.mm(
            A_torch.type(torch.float16),
            B_torch.type(torch.float16).permute(1, 0))
        _Output_tvm = Output_tvm.asnumpy()
        _Output_tvm = _Output_tvm.reshape(
            _Output_tvm.shape[0] * _Output_tvm.shape[1],
            _Output_tvm.shape[2] * _Output_tvm.shape[3])
        _Output_tvm = _Output_tvm[:M, :N]
        from tvm import testing
        testing.assert_allclose(
            Output_torch.cpu().numpy(), _Output_tvm,
            rtol=1e-1, atol=1e-1)

    timed_func = gemm_func.time_evaluator(
        gemm_func.entry_name, ctx, number=20, min_repeat_ms=500)
    cost = timed_func(A_tvm, B_tvm, Output_tvm, *params, *vars_).mean
    print(",".join(["Gemm_split_k", in_dtype, out_dtype] + [str(x) for x in [
        M, N, K, cost * 1e3
    ]]))


def run_gemm_split_K_perfect(
        M, N, K, in_dtype="float16", out_dtype="float32", target="llvm", verify=True, dump=False):
    (
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size
    )    = (
        [64, 64, 16],
        [16, 32, 16],
        [16, 16, 16]
    )

    epilogues = []

    (
        Output,
        (A, B),
        schedule_func,
        Params,
        Vars
    ) = kernel_gemm_split_K_perfect(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        split_K=1,
        A_dtype=in_dtype,
        B_dtype=in_dtype,
        C_dtype=out_dtype
    )
    sch = tvm.te.create_schedule(Output.op)
    for func in schedule_func:
        func(sch)

    ctx = tvm.context(target)
    import numpy as np
    if dump:
        print(tvm.lower(
            sch, [A, B, Output, *Params, *Vars],
            simple_mode=True
        ))

    gemm_func = tvm.build(
        sch, [A, B, Output, *Params, *Vars], target=target)

    params = [M, N, K]
    vars_ = [
        ceil(M, threadblock_problem_size[0]),
        ceil(N, threadblock_problem_size[1]),
        ceil(K, threadblock_problem_size[2])]

    A_np = np.random.uniform(-1, 1, [M, K])
    B_np = np.random.uniform(-1, 1, [N, K])
    Output_np = np.zeros(
        [M, N],
        dtype=Output.dtype)
    A_tvm = tvm.nd.array(A_np.astype(A.dtype), ctx)
    B_tvm = tvm.nd.array(B_np.astype(B.dtype), ctx)
    Output_tvm = tvm.nd.array(Output_np, ctx)

    gemm_func(A_tvm, B_tvm, Output_tvm,
                *params, *vars_)

    if verify:
        import torch
        if torch.cuda.is_available():
            A_torch = torch.tensor(A_np).to("cuda")
            B_torch = torch.tensor(B_np).to("cuda")
        else:
            A_torch = torch.tensor(A_np)
            B_torch = torch.tensor(B_np)
        Output_torch = torch.mm(
            A_torch.type(torch.float16),
            B_torch.type(torch.float16).permute(1, 0))
        _Output_tvm = Output_tvm.asnumpy()
        _Output_tvm = _Output_tvm[:M, :N]
        from tvm import testing
        testing.assert_allclose(
            Output_torch.cpu().numpy(), _Output_tvm,
            rtol=1e-1, atol=1e-1)

    timed_func = gemm_func.time_evaluator(
        gemm_func.entry_name, ctx, number=20, min_repeat_ms=500)
    cost = timed_func(A_tvm, B_tvm, Output_tvm, *params, *vars_).mean
    print(",".join(["Gemm_split_k_perfect", in_dtype, out_dtype] + [str(x) for x in [
        M, N, K, cost * 1e3
    ]]))


def run_conv2d_nchw_implicit_gemm_perfect(
        N, C, H, W, K, R, S,
        in_dtype="float16", out_dtype="float32",
        target="llvm", verify=True, dump=False):
    (
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size
    )    = (
        [64, 32, 128],
        [16, 16, 16],
        [16, 16, 16]
    )

    stride = 1
    padding = 1
    dilation = 1

    P = (H + 2 * padding - (R - 1) * dilation - 1) // stride + 1
    Q = (W + 2 * padding - (S - 1) * dilation - 1) // stride + 1

    epilogues = []

    (
        Output,
        (A, B),
        schedule_func,
        Params,
        Vars
    ) = kernel_conv2d_nchw_implicit_gemm_perfect(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A_dtype=in_dtype,
        B_dtype=in_dtype,
        C_dtype=out_dtype,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    sch = tvm.te.create_schedule(Output.op)
    for func in schedule_func:
        func(sch)

    ctx = tvm.context(target)
    import numpy as np
    if dump:
        print(tvm.lower(
            sch, [A, B, Output, *Params, *Vars],
            simple_mode=True
        ))

    gemm_func = tvm.build(
        sch, [A, B, Output, *Params, *Vars], target=target)

    params = [N, C, H, W, K, R, S]
    vars_ = [
        ceil(N*P*Q, threadblock_problem_size[0]),
        ceil(K, threadblock_problem_size[1]),
        ceil(C*R*S, threadblock_problem_size[2])]

    A_np = np.random.uniform(-1, 1, [N, C, H, W])
    B_np = np.random.uniform(-1, 1, [K, C, R, S])
    Output_np = np.zeros(
        [N, K, P, Q],
        dtype=Output.dtype)
    A_tvm = tvm.nd.array(A_np.astype(A.dtype), ctx)
    B_tvm = tvm.nd.array(B_np.astype(B.dtype), ctx)
    Output_tvm = tvm.nd.array(Output_np, ctx)

    gemm_func(A_tvm, B_tvm, Output_tvm,
                *params, *vars_)

    if verify:
        import torch
        if torch.cuda.is_available():
            A_torch = torch.tensor(A_np).to("cuda")
            B_torch = torch.tensor(B_np).to("cuda")
        else:
            A_torch = torch.tensor(A_np)
            B_torch = torch.tensor(B_np)
        Output_torch = torch.nn.functional.conv2d(
            A_torch.type(torch.float16),
            B_torch.type(torch.float16),
            padding=padding,
            stride=stride,
            dilation=dilation)
        _Output_tvm = Output_tvm.asnumpy()
        from tvm import testing
        testing.assert_allclose(
            Output_torch.cpu().numpy(), _Output_tvm,
            rtol=1e-0, atol=1e-0)

    timed_func = gemm_func.time_evaluator(
        gemm_func.entry_name, ctx, number=20, min_repeat_ms=500)
    cost = timed_func(A_tvm, B_tvm, Output_tvm, *params, *vars_).mean
    print(",".join(["Conv2d_nchw_implicit_gemm_perfect", in_dtype, out_dtype] + [str(x) for x in [
        N, C, H, W, K, R, S, cost * 1e3
    ]]))


def run_conv2d_nhwc_implicit_gemm_perfect(
        N, C, H, W, K, R, S,
        in_dtype="float16", out_dtype="float32",
        target="llvm", verify=True, dump=False):
    (
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size
    )    = (
        [64, 32, 128],
        [16, 16, 16],
        [16, 16, 16]
    )

    stride = 1
    padding = 1
    dilation = 1

    P = (H + 2 * padding - (R - 1) * dilation - 1) // stride + 1
    Q = (W + 2 * padding - (S - 1) * dilation - 1) // stride + 1

    epilogues = []

    (
        Output,
        (A, B),
        schedule_func,
        Params,
        Vars
    ) = kernel_conv2d_nhwc_implicit_gemm_perfect(
        threadblock_problem_size,
        warp_problem_size,
        tensorize_problem_size,
        epilogues,
        A_dtype=in_dtype,
        B_dtype=in_dtype,
        C_dtype=out_dtype,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    sch = tvm.te.create_schedule(Output.op)
    for func in schedule_func:
        func(sch)

    ctx = tvm.context(target)
    import numpy as np
    if dump:
        print(tvm.lower(
            sch, [A, B, Output, *Params, *Vars],
            simple_mode=True
        ))

    gemm_func = tvm.build(
        sch, [A, B, Output, *Params, *Vars], target=target)

    params = [N, C, H, W, K, R, S]
    vars_ = [
        ceil(N*P*Q, threadblock_problem_size[0]),
        ceil(K, threadblock_problem_size[1]),
        ceil(C*R*S, threadblock_problem_size[2])]

    A_np = np.random.uniform(-1, 1, [N, H, W, C])
    B_np = np.random.uniform(-1, 1, [K, C, R, S])
    Output_np = np.zeros(
        [N, P, Q, K],
        dtype=Output.dtype)
    A_tvm = tvm.nd.array(A_np.astype(A.dtype), ctx)
    B_tvm = tvm.nd.array(B_np.astype(B.dtype), ctx)
    Output_tvm = tvm.nd.array(Output_np, ctx)

    gemm_func(A_tvm, B_tvm, Output_tvm,
                *params, *vars_)

    if verify:
        import torch
        if torch.cuda.is_available():
            A_torch = torch.tensor(A_np).to("cuda")
            B_torch = torch.tensor(B_np).to("cuda")
        else:
            A_torch = torch.tensor(A_np)
            B_torch = torch.tensor(B_np)
        Output_torch = torch.nn.functional.conv2d(
            A_torch.type(torch.float16).permute(0, 3, 1, 2),
            B_torch.type(torch.float16),
            padding=padding,
            stride=stride,
            dilation=dilation).permute(0, 2, 3, 1)
        _Output_tvm = Output_tvm.asnumpy()
        from tvm import testing
        testing.assert_allclose(
            Output_torch.cpu().numpy(), _Output_tvm,
            rtol=1e-0, atol=1e-0)

    timed_func = gemm_func.time_evaluator(
        gemm_func.entry_name, ctx, number=20, min_repeat_ms=500)
    cost = timed_func(A_tvm, B_tvm, Output_tvm, *params, *vars_).mean
    print(",".join(["Conv2d_nhwc_implicit_gemm_perfect", in_dtype, out_dtype] + [str(x) for x in [
        N, C, H, W, K, R, S, cost * 1e3
    ]]))


if __name__ == "__main__":
    print("========= Gemm =========")
    print("type,input,output,M,N,K,time(ms)")
    run_gemm_split_K(1024, 1024, 1024, target="cuda", verify=True, dump=False)
    run_gemm_split_K_perfect(1024, 1024, 1024, target="cuda", verify=True)
    run_gemm(1024, 1024, 1024, target="cuda", verify=True)
    run_gemm_perfect(1024, 1024, 1024, target="cuda", verify=True)
    print("========= Conv2d =========")
    print("type,input,output,N,C,H,W,K,R,S,time(ms)")
    run_conv2d_nchw_implicit_gemm_perfect(
        1, 512, 7, 7, 512, 3, 3,
        in_dtype="float16", out_dtype="float16",
        target="cuda", verify=False, dump=False
    )
    run_conv2d_nchw_implicit_gemm_perfect(
        1, 64, 56, 56, 64, 3, 3,
        in_dtype="float16", out_dtype="float16",
        target="cuda", verify=False, dump=False
    )
    run_conv2d_nhwc_implicit_gemm_perfect(
        1, 512, 7, 7, 512, 3, 3,
        in_dtype="float16", out_dtype="float16",
        target="cuda", verify=False, dump=False
    )
    run_conv2d_nhwc_implicit_gemm_perfect(
        1, 64, 56, 56, 64, 3, 3,
        in_dtype="float16", out_dtype="float16",
        target="cuda", verify=False, dump=False
    )