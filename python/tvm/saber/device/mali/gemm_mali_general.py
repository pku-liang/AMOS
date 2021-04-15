import tvm
from tvm.ir import transform
from ..base import Operator
from ...utils import ceil
from ...kernel import (
    get_gemm_implementation_mali
)
from ..measure import MeasureOptions, evaluate_function, evaluate_schedule


class GemmGeneral(Operator):
    def __init__(self, in_dtype="float32", out_dtype="float32",
                    threadblock_problem_size=[32, 32, 32],
                    warp_problem_size=[32, 32, 8],
                    instruction_problem_size=[4, 4, 8],
                    epilogues=[],
                    split_K=1,
                    arch="bifrost",
                    code="g71",
                    tag="single_buffer"):
        super(GemmGeneral, self).__init__()
        self.target = "opencl"
        self.target_host = "llvm -mtriple=aarch64-linux-android"
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.threadblock_problem_size = threadblock_problem_size
        self.warp_problem_size = warp_problem_size
        self.instruction_problem_size = instruction_problem_size
        self.epilogues = []
        self.split_K = split_K
        if self.split_K > 1:
            raise RuntimeError("Not support split_K > 1")
        else:
            self.get_context = lambda *_: get_gemm_implementation_mali("general", arch, code, tag)(
                                self.threadblock_problem_size,
                                self.warp_problem_size,
                                self.instruction_problem_size,
                                self.epilogues,
                                A_dtype=self.in_dtype,
                                B_dtype=self.in_dtype,
                                C_dtype=self.out_dtype
                            )


    def compile(self, dump=False):
        (
            Output,
            (A, B),
            schedule_func,
            Params,
            Vars
        ) = self.get_context()
        sch = tvm.te.create_schedule(Output.op)
        ctx = {}
        for func in schedule_func:
            ctx = func(sch, ctx=ctx)

        if dump:
            print(tvm.lower(
                sch, [A, B, Output, *Params, *Vars],
                simple_mode=True
            ))

        gemm_func = tvm.build(
            sch, [A, B, Output, *Params, *Vars],
            target=self.target, target_host=self.target_host
        )

        return gemm_func

    def expose_compile_context(self):
        (
            Output,
            (A, B),
            schedule_func,
            Params,
            Vars
        ) = self.get_context()
        def _sch_impl():
            sch = tvm.te.create_schedule(Output.op)
            ctx = {}
            for func in schedule_func:
                ctx = func(sch, ctx=ctx)
            return sch
        return _sch_impl, (A, B, Output), (*Params, *Vars)

    def evaluate(self, func, M, N, K, measure_opt=MeasureOptions(
            target="opencl",
            target_host="llvm -mtriple=aarch64-linux-android",
            timeout=40, number=200,
            min_repeat_ms=80,
            build_func="ndk",
            key="android",
            host="0.0.0.0",
            port=9190,
            cooldown_interval=5), new_process=False):
        A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
        B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        Output = tvm.te.placeholder([M, N], dtype=self.out_dtype)
        # Output = tvm.te.placeholder(
        #     # [
        #     #     (M + self.threadblock_problem_size[0] - 1) // self.threadblock_problem_size[0],
        #     #     (N + self.threadblock_problem_size[1] - 1) // self.threadblock_problem_size[1],
        #     #     self.threadblock_problem_size[0] // self.warp_problem_size[0],
        #     #     self.threadblock_problem_size[1] // self.warp_problem_size[1],
        #     #     self.warp_problem_size[0] // self.instruction_problem_size[0],
        #     #     self.warp_problem_size[1] // self.instruction_problem_size[1],
        #     #     self.instruction_problem_size[0],
        #     #     self.instruction_problem_size[1]
        #     # ],
        #     [
        #         (M + self.threadblock_problem_size[0] - 1) // self.threadblock_problem_size[0],
        #         self.threadblock_problem_size[0] // self.warp_problem_size[0],
        #         self.warp_problem_size[0] // self.instruction_problem_size[0],
        #         self.instruction_problem_size[0],
        #         (N + self.threadblock_problem_size[1] - 1) // self.threadblock_problem_size[1],
        #         self.threadblock_problem_size[1] // self.warp_problem_size[1],
        #         self.warp_problem_size[1] // self.instruction_problem_size[1],
        #         self.instruction_problem_size[1]
        #     ],
        #     dtype=self.out_dtype)
        args = [A, B, Output]
        var_values = [
            M, N, K,
            ceil(M, self.threadblock_problem_size[0]),
            ceil(N, self.threadblock_problem_size[1]),
            ceil(K, self.threadblock_problem_size[2])
        ]
        return evaluate_function(
            func, args, var_values, measure_opt, new_process=new_process
        )

    def expose_evaluate_context(self, M, N, K):
        A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
        B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        Output = tvm.te.placeholder([M, N], dtype=self.out_dtype)
        args = [A, B, Output]
        var_values = [
            M, N, K,
            ceil(M, self.threadblock_problem_size[0]),
            ceil(N, self.threadblock_problem_size[1]),
            ceil(K, self.threadblock_problem_size[2])
        ]
        return args, var_values

    def calculate(self, func, A, B, C):
        M, K = A.shape
        N, _ = B.shape
        var_values = [
            M, N, K,
            ceil(M, self.threadblock_problem_size[0]),
            ceil(N, self.threadblock_problem_size[1]),
            ceil(K, self.threadblock_problem_size[2])
        ]
        func(A, B, C, *var_values)

    def try_with(self, M, N, K, measure_opt=MeasureOptions(
            target="opencl",
            target_host="llvm -mtriple=aarch64-linux-android",
            timeout=40, number=100,
            min_repeat_ms=80,
            build_func="ndk",
            key="android",
            host="0.0.0.0",
            port=9190,
            cooldown_interval=5), new_process=False, dump=False):
        (
            Output,
            (A, B),
            schedule_func,
            Params,
            Vars
        ) = self.get_context()
        sch = tvm.te.create_schedule(Output.op)
        args = [A, B, Output]
        for func in schedule_func:
            func(sch)
        A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
        B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        Output = tvm.te.placeholder([M, N], dtype=self.out_dtype)
        # Output = tvm.te.placeholder(
        #     # [
        #     #     (M + self.threadblock_problem_size[0] - 1) // self.threadblock_problem_size[0],
        #     #     (N + self.threadblock_problem_size[1] - 1) // self.threadblock_problem_size[1],
        #     #     self.threadblock_problem_size[0] // self.warp_problem_size[0],
        #     #     self.threadblock_problem_size[1] // self.warp_problem_size[1],
        #     #     self.warp_problem_size[0] // self.instruction_problem_size[0],
        #     #     self.warp_problem_size[1] // self.instruction_problem_size[1],
        #     #     self.instruction_problem_size[0],
        #     #     self.instruction_problem_size[1]
        #     # ],
        #     [
        #         (M + self.threadblock_problem_size[0] - 1) // self.threadblock_problem_size[0],
        #         self.threadblock_problem_size[0] // self.warp_problem_size[0],
        #         self.warp_problem_size[0] // self.instruction_problem_size[0],
        #         self.instruction_problem_size[0],
        #         (N + self.threadblock_problem_size[1] - 1) // self.threadblock_problem_size[1],
        #         self.threadblock_problem_size[1] // self.warp_problem_size[1],
        #         self.warp_problem_size[1] // self.instruction_problem_size[1],
        #         self.instruction_problem_size[1]
        #     ],
        #     dtype=self.out_dtype)
        arg_values = [A, B, Output]
        var_values = [
            M, N, K,
            ceil(M, self.threadblock_problem_size[0]),
            ceil(N, self.threadblock_problem_size[1]),
            ceil(K, self.threadblock_problem_size[2])
        ]

        if dump:
            print(tvm.lower(sch, [*args, *Vars], simple_mode=True))
        return evaluate_schedule(
            sch, args, list(Params) + list(Vars), arg_values, var_values, measure_opt, new_process=new_process)
