import tvm
from ...threadblock_implementation_route import (
    get_gemm_implementation_cuda
)
from ...utils import index


def kernel_gemm_tensorcore_split_K_common_common(arch, code, tag="single_buffer"):
    def _kernel_gemm_tensorcore_split_K(
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
        ) = get_gemm_implementation_cuda("tensorcore", arch, code, tag)(
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

        def schedule_kernel(sch, ctx=None):
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
    return _kernel_gemm_tensorcore_split_K


def kernel_gemm_tensorcore_common_common(arch, code, tag="single_buffer"):
    def _kernel_gemm_tensorcore(
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
        ) = get_gemm_implementation_cuda("tensorcore", arch, code, tag)(
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

        def schedule_kernel(sch, ctx=None):
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
    
    return _kernel_gemm_tensorcore


def kernel_gemm_tensorcore_split_K_perfect_common_common(arch, code, tag="single_buffer"):
    def _kernel_gemm_tensorcore_split_K_perfect(
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
        ) = get_gemm_implementation_cuda("tensorcore", arch, code, tag)(
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

        def schedule_kernel(sch, ctx=None):
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

    return _kernel_gemm_tensorcore_split_K_perfect


def kernel_gemm_tensorcore_perfect_common_common(arch, code, tag="single_buffer"):
    def _kernel_gemm_tensorcore_perfect(
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
        ) = get_gemm_implementation_cuda("tensorcore", arch, code, tag)(
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

        def schedule_kernel(sch, ctx=None):
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

    return _kernel_gemm_tensorcore_perfect
