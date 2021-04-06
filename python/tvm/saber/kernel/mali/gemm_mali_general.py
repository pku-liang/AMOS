import tvm
from ...threadblock import (
    threadblock_gemm_mali_general)
from ...utils import index


def kernel_gemm_general_perfect(
    threadblock_problem_size,
    warp_problem_size,
    instruction_problem_size,
    epilogues,
    A_dtype="float32",
    B_dtype="float32",
    C_dtype="float32"
):
    M = index("M")
    N = index("N")
    K = index("K")

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

    Params = [M, N, K]
    A = tvm.te.placeholder([M, K], dtype=A_dtype, name="A")
    B = tvm.te.placeholder([N, K], dtype=B_dtype, name="B")
    (
        Output,
        schedule_func,
        parse_func
    ) = threadblock_gemm_mali_general(
        [M1, N1, K1],
        threadblock_problem_size,
        warp_problem_size,
        instruction_problem_size,
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

    def schedule_kernel(sch, ctx=None):
        ctx = {} if ctx is None else ctx
        Last = Gemm
        m, n = sch[Last].op.axis
        m1, mi = sch[Last].pred_split(m, factor=M2*M3*M4, nparts=M1)
        n1, ni = sch[Last].pred_split(n, factor=N2*N3*N4, nparts=N1)
        m2, mi = sch[Last].split(mi, factor=M3*M4)
        m3, m4 = sch[Last].split(mi, factor=M4)
        n2, ni = sch[Last].split(ni, factor=N3*N4)
        n3, n4 = sch[Last].split(ni, factor=N4)
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
        
        new_ctx = {}
        new_ctx.update(ctx)
        new_ctx["Output"] = {
            "tensor": Last,
            "axis": [m1, n1, m2, m3, n2, n3, m4, n4]
        }
        return new_ctx

    return (
        Gemm,
        [A, B],
        [schedule_kernel] + schedule_func,
        Params,
        (M1, N1, K1)
    )
