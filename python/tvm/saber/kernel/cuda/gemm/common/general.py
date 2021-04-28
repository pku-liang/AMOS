import tvm
from tvm import te
from ...threadblock_implementation_route import (
    get_gemm_implementation_cuda
)
from ...utils import index


def kernel_gemm_general_perfect_common_common(arch, code, tag="double_buffer"):
    def _kernel_gemm_general_perfect(
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

        def block(x): return te.thread_axis(f"blockIdx.{x}")
        def thread(x): return te.thread_axis(f"threadIdx.{x}")
        def vthread(): return te.thread_axis("vthread")

        Params = [M, N, K]
        A = te.placeholder([K, M], dtype=A_dtype, name="A")
        B = te.placeholder([K, N], dtype=B_dtype, name="B")
        (
            Output,
            schedule_func,
            parse_func
        ) = get_gemm_implementation_cuda("general", arch, code, tag)(
            [M1, N1, K1],
            threadblock_problem_size,
            warp_problem_size,
            instruction_problem_size,
            epilogues,
            A, B,
            C_dtype=C_dtype
        )

        Gemm = te.compute(
            [M, N],
            lambda m, n:
                parse_func(m, n),
            name="Gemm"
        )

        def schedule_kernel(sch):
            Last = Gemm

            m, n = sch[Last].op.axis
            m1, mi = sch[Last].pred_split(m, factor=M2*M3*M4, nparts=M1)
            n1, ni = sch[Last].pred_split(n, factor=N2*N3*N4, nparts=N1)
            m2, mi = sch[Last].split(mi, factor=M3*M4)
            m3, m4 = sch[Last].split(mi, factor=M4)
            n2, ni = sch[Last].split(ni, factor=N3*N4)
            n3, n4 = sch[Last].split(ni, factor=N4)

            sch[Last].reorder(m1, n1, m2, n2, m3, n3, m4, n4)
            sch[Last].bind(m1, block("y"))
            sch[Last].bind(n1, block("x"))
            sch[Last].bind(m2, thread("z"))
            sch[Last].bind(n2, thread("y"))
            sch[Last].bind(m3, vthread())
            sch[Last].bind(n3, vthread())
            warp = sch[Last].fuse(m4, n4)
            sch[Last].bind(warp, thread("x"))

            # unroll, vec = sch[Last].split(n4, factor=C_vec_L)
            # sch[Last].unroll(unroll)
            # sch[Last].vectorize(vec)

        return (
            Gemm,
            [A, B],
            [schedule_kernel] + schedule_func,
            Params,
            (M1, N1, K1)
        )

    return _kernel_gemm_general_perfect
