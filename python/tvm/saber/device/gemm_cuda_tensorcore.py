import tvm
from .base import Operator
from ..utils import ceil
from ..kernel import (
    kernel_gemm_cuda_tensorcore_perfect,
    kernel_gemm_cuda_tensorcore_split_K_perfect
)
from .measure import MeasureOptions, evaluate_function, evaluate_schedule


class GemmTensorCore(Operator):
    def __init__(self, in_dtype="float16", out_dtype="float32",
                    threadblock_problem_size=[128, 128, 64],
                    warp_problem_size=[64, 64, 32],
                    tensorcore_problem_size=[16, 16, 16],
                    epilogues=[],
                    split_K=1):
        super(GemmTensorCore, self).__init__()
        self.target = "cuda"
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.threadblock_problem_size = threadblock_problem_size
        self.warp_problem_size = warp_problem_size
        self.tensorcore_problem_size = tensorcore_problem_size
        self.epilogues = []
        self.split_K = split_K
        if self.split_K > 1:
            self.get_context = lambda *_: kernel_gemm_cuda_tensorcore_split_K_perfect(
                                self.threadblock_problem_size,
                                self.warp_problem_size,
                                self.tensorcore_problem_size,
                                self.epilogues,
                                A_dtype=self.in_dtype,
                                B_dtype=self.in_dtype,
                                C_dtype=self.out_dtype,
                                split_K=self.split_K
                            )
        else:
            self.get_context = lambda *_: kernel_gemm_cuda_tensorcore_perfect(
                                self.threadblock_problem_size,
                                self.warp_problem_size,
                                self.tensorcore_problem_size,
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
        for func in schedule_func:
            func(sch)

        if dump:
            print(tvm.lower(
                sch, [A, B, Output, *Params, *Vars],
                simple_mode=True
            ))

        gemm_func = tvm.build(
            sch, [A, B, Output, *Params, *Vars], target=self.target)

        return gemm_func

    def evaluate(self, func, M, N, K, measure_opt=MeasureOptions(
            target="cuda", min_repeat_ms=500), new_process=False):
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
        return evaluate_function(
            func, args, var_values, measure_opt, new_process=new_process
        )

    def try_with(self, M, N, K, measure_opt=MeasureOptions(
            target="cuda", min_repeat_ms=500), new_process=False):
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
        arg_values = [A, B, Output]
        var_values = [
            M, N, K,
            ceil(M, self.threadblock_problem_size[0]),
            ceil(N, self.threadblock_problem_size[1]),
            ceil(K, self.threadblock_problem_size[2])
        ]
        return evaluate_schedule(
            sch, args, list(Params) + list(Vars), arg_values, var_values, measure_opt, new_process=new_process)
