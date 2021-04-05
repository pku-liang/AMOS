import tvm
from ..base import Operator
from ...utils import ceil
from ...kernel import (
    kernel_conv2d_nchw_implicit_gemm_cuda_tensorcore_perfect,
    kernel_conv2d_nhwc_implicit_gemm_cuda_tensorcore_perfect
)
from ..measure import MeasureOptions, evaluate_function, evaluate_schedule


class Conv2dTensorCore(Operator):
    def __init__(self, in_dtype="float16", out_dtype="float32",
                    threadblock_problem_size=[128, 128, 64],
                    warp_problem_size=[64, 64, 32],
                    tensorcore_problem_size=[16, 16, 16],
                    epilogues=[],
                    layout="nchw",
                    stride=1,
                    padding=0,
                    dilation=1,
                    split_K=None):
        super(Conv2dTensorCore, self).__init__()
        self.target = "cuda"
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.threadblock_problem_size = threadblock_problem_size
        self.warp_problem_size = warp_problem_size
        self.tensorcore_problem_size = tensorcore_problem_size
        self.epilogues = []
        self.layout = "nchw"
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.split_K = split_K
        if self.layout == "nchw":
            self.get_context = lambda *_: kernel_conv2d_nchw_implicit_gemm_cuda_tensorcore_perfect(
                                self.threadblock_problem_size,
                                self.warp_problem_size,
                                self.tensorcore_problem_size,
                                self.epilogues,
                                A_dtype=self.in_dtype,
                                B_dtype=self.in_dtype,
                                C_dtype=self.out_dtype,
                                stride=stride,
                                padding=padding,
                                dilation=dilation
                            )
        elif self.layout == "nhwc":
            self.get_context = lambda *_: kernel_conv2d_nhwc_implicit_gemm_cuda_tensorcore_perfect(
                                self.threadblock_problem_size,
                                self.warp_problem_size,
                                self.tensorcore_problem_size,
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
        for func in schedule_func:
            func(sch)

        if dump:
            print(tvm.lower(
                sch, [A, B, Output, *Params, *Vars],
                simple_mode=True
            ))

        conv2d_func = tvm.build(
            sch, [A, B, Output, *Params, *Vars], target=self.target)

        return conv2d_func

    def evaluate(self, func, N, C, H, W, K, R, S, measure_opt=MeasureOptions(
            target="cuda", min_repeat_ms=500), new_process=False):
        P = (H + 2 * self.padding - (R - 1) * self.dilation - 1) // self.stride + 1
        Q = (W + 2 * self.padding - (S - 1) * self.dilation - 1) // self.stride + 1
        if self.layout == "nchw":
            A = tvm.te.placeholder([N, C, H, W], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, C, R, S], dtype=self.in_dtype)
            Output = tvm.te.placeholder([N, K, P, Q], dtype=self.out_dtype)
        elif self.layout == "nhwc":
            A = tvm.te.placeholder([N, H, W, C], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, C, R, S], dtype=self.in_dtype)
            Output = tvm.te.placeholder([N, P, Q, K], dtype=self.out_dtype)
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

    def try_with(self, N, C, H, W, K, R, S, measure_opt=MeasureOptions(
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
        P = (H + 2 * self.padding - (R - 1) * self.dilation - 1) // self.stride + 1
        Q = (W + 2 * self.padding - (S - 1) * self.dilation - 1) // self.stride + 1
        if self.layout == "nchw":
            A = tvm.te.placeholder([N, C, H, W], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, C, R, S], dtype=self.in_dtype)
            Output = tvm.te.placeholder([N, K, P, Q], dtype=self.out_dtype)
        elif self.layout == "nhwc":
            A = tvm.te.placeholder([N, H, W, C], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, C, R, S], dtype=self.in_dtype)
            Output = tvm.te.placeholder([N, P, Q, K], dtype=self.out_dtype)
        arg_values = [A, B, Output]
        var_values = [
            N, C, H, W, K, R, S,
            ceil(N*P*Q, self.threadblock_problem_size[0]),
            ceil(K, self.threadblock_problem_size[1]),
            ceil(C*R*S, self.threadblock_problem_size[2])
        ]
        return evaluate_schedule(
            sch, args, list(Params) + list(Vars), arg_values, var_values, measure_opt, new_process=new_process)
