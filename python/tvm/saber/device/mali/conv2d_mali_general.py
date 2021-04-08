import tvm
from ..base import Operator
from ...utils import ceil
from ...kernel import (
    kernel_conv2d_nchw_implicit_gemm_mali_general_perfect
)
from ..measure import MeasureOptions, evaluate_function, evaluate_schedule


class Conv2dGeneral(Operator):
    def __init__(self, in_dtype="float32", out_dtype="float32",
                    threadblock_problem_size=[32, 32, 32],
                    warp_problem_size=[8, 8, 8],
                    instruction_problem_size=[4, 4, 4],
                    epilogues=[],
                    layout="nchw",
                    stride=1,
                    padding=0,
                    dilation=1,
                    split_K=None):
        super(Conv2dGeneral, self).__init__()
        self.target = "opencl"
        self.target_host = "llvm -mtriple=aarch64-linux-android"
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.threadblock_problem_size = threadblock_problem_size
        self.warp_problem_size = warp_problem_size
        self.instruction_problem_size = instruction_problem_size
        self.epilogues = []
        self.layout = "nchw"
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.split_K = split_K
        if self.layout == "nchw":
            self.get_context = lambda *_: kernel_conv2d_nchw_implicit_gemm_mali_general_perfect(
                                self.threadblock_problem_size,
                                self.warp_problem_size,
                                self.instruction_problem_size,
                                self.epilogues,
                                A_dtype=self.in_dtype,
                                B_dtype=self.in_dtype,
                                C_dtype=self.out_dtype,
                                stride=stride,
                                padding=padding,
                                dilation=dilation
                            )
        else:
            raise RuntimeError("Layout not supported: " + str(layout))


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

        conv2d_func = tvm.build(
            sch, [A, B, Output, *Params, *Vars], target=self.target, target_host=self.target_host)

        return conv2d_func

    def evaluate(self, func, N, C, H, W, K, R, S, measure_opt=MeasureOptions(
            target="opencl",
            target_host="llvm -mtriple=aarch64-linux-android",
            timeout=10, number=400,
            min_repeat_ms=800,
            build_func="ndk",
            key="android",
            host="0.0.0.0",
            port=9190,
            cooldown_interval=5), new_process=False):
        P = (H + 2 * self.padding - (R - 1) * self.dilation - 1) // self.stride + 1
        Q = (W + 2 * self.padding - (S - 1) * self.dilation - 1) // self.stride + 1
        if self.layout == "nchw":
            A = tvm.te.placeholder([N, C, H, W], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, C, R, S], dtype=self.in_dtype)
            Output = tvm.te.placeholder([N, K, P, Q], dtype=self.out_dtype)
            # Output = tvm.te.placeholder(
            #     [
            #         (N*P*Q + self.threadblock_problem_size[0] - 1) // self.threadblock_problem_size[0],
            #         (K + self.threadblock_problem_size[1] - 1) // self.threadblock_problem_size[1],
            #         self.threadblock_problem_size[0] // self.warp_problem_size[0],
            #         self.warp_problem_size[0] // self.instruction_problem_size[0],
            #         self.threadblock_problem_size[1] // self.warp_problem_size[1],
            #         self.warp_problem_size[1] // self.instruction_problem_size[1],
            #         self.instruction_problem_size[0],
            #         self.instruction_problem_size[1]
            #     ],
            #     dtype=self.out_dtype)
        else:
            raise RuntimeError("Not support layout: " + str(self.layout))
        args = [A, B, Output]
        var_values = [
            N, C, H, W, K, R, S,
            ceil(N*P*Q, self.threadblock_problem_size[0]),
            ceil(K, self.threadblock_problem_size[1]),
            ceil(C*R*S, self.threadblock_problem_size[2])
        ]
        return evaluate_function(
            func, args, var_values, measure_opt, new_process=new_process
        )

    def calculate(self, func, A, B, C):
        N, C, H, W = A.shape
        K, _, R, S = B.shape
        var_values = [
            N, C, H, W, K, R, S,
            ceil(N*P*Q, self.threadblock_problem_size[0]),
            ceil(K, self.threadblock_problem_size[1]),
            ceil(C*R*S, self.threadblock_problem_size[2])
        ]
        func(A, B, C, *var_values)

    def try_with(self, N, C, H, W, K, R, S, measure_opt=MeasureOptions(
            target="opencl",
            target_host="llvm -mtriple=aarch64-linux-android",
            timeout=10, number=10,
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
        P = (H + 2 * self.padding - (R - 1) * self.dilation - 1) // self.stride + 1
        Q = (W + 2 * self.padding - (S - 1) * self.dilation - 1) // self.stride + 1
        if self.layout == "nchw":
            A = tvm.te.placeholder([N, C, H, W], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, C, R, S], dtype=self.in_dtype)
            Output = tvm.te.placeholder([N, K, P, Q], dtype=self.out_dtype)
            # Output = tvm.te.placeholder(
            #     [
            #         (N*P*Q + self.threadblock_problem_size[0] - 1) // self.threadblock_problem_size[0],
            #         (K + self.threadblock_problem_size[1] - 1) // self.threadblock_problem_size[1],
            #         self.threadblock_problem_size[0] // self.warp_problem_size[0],
            #         self.warp_problem_size[0] // self.instruction_problem_size[0],
            #         self.threadblock_problem_size[1] // self.warp_problem_size[1],
            #         self.warp_problem_size[1] // self.instruction_problem_size[1],
            #         self.instruction_problem_size[0],
            #         self.instruction_problem_size[1]
            #     ],
            #     dtype=self.out_dtype)
        else:
            raise RuntimeError("Not support layout:" + str(self.layout))
        arg_values = [A, B, Output]
        var_values = [
            N, C, H, W, K, R, S,
            ceil(N*P*Q, self.threadblock_problem_size[0]),
            ceil(K, self.threadblock_problem_size[1]),
            ceil(C*R*S, self.threadblock_problem_size[2])
        ]
        if dump:
            print(tvm.lower(sch, [*args, *Vars], simple_mode=True))
        return evaluate_schedule(
            sch, args, list(Params) + list(Vars), arg_values, var_values, measure_opt, new_process=new_process)
