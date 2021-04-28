import tvm
from ..utils import ceil
from .measure import MeasureOptions, evaluate_function, evaluate_schedule


class Operator(object):
    pass


class GemmOperator(Operator):
    def __init__(self,
                 target,
                 get_gemm_implementation,
                 in_dtype, out_dtype,
                 threadblock_problem_size,
                 warp_problem_size,
                 instruction_problem_size,
                 epilogues,
                 split_K,
                 type_name,
                 arch,
                 code,
                 tag,
                 layout,
                 algorithm,
                 strategy):
        super(GemmOperator, self).__init__()
        self.target = target
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.threadblock_problem_size = threadblock_problem_size
        self.warp_problem_size = warp_problem_size
        self.instruction_problem_size = instruction_problem_size
        self.epilogues = []
        self.split_K = split_K
        self.layout = layout
        self.algorithm = algorithm
        self.strategy = strategy
        if self.split_K > 1:
            self.get_context = lambda *_: get_gemm_implementation(type_name, arch, code, tag, layout, algorithm, strategy)(
                self.threadblock_problem_size,
                self.warp_problem_size,
                self.instruction_problem_size,
                self.epilogues,
                A_dtype=self.in_dtype,
                B_dtype=self.in_dtype,
                C_dtype=self.out_dtype,
                split_K=split_K
            )
        else:
            self.get_context = lambda *_: get_gemm_implementation(type_name, arch, code, tag, layout, algorithm, strategy)(
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
        for func in schedule_func:
            func(sch)

        if dump:
            print(tvm.lower(
                sch, [A, B, Output, *Params, *Vars],
                simple_mode=True
            ))

        gemm_func = tvm.build(
            sch, [A, B, Output, *Params, *Vars],
            target=self.target
        )

        return gemm_func

    def evaluate(self, func, M, N, K, measure_opt=MeasureOptions(
            target="cuda", number=100,
            min_repeat_ms=500), new_process=False):
        if self.layout == "NN":
            A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, N], dtype=self.in_dtype)
        elif self.layout == "NT":
            A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
            B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        elif self.layout == "TN":
            A = tvm.te.placeholder([K, M], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, N], dtype=self.in_dtype)
        elif self.layout == "TT":
            A = tvm.te.placeholder([K, M], dtype=self.in_dtype)
            B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        else:
            raise RuntimeError("Unsupported layout:" + str(self.layout))
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

    def calculate(self, func, A, B, C):
        K, M = A.shape
        _, N = B.shape
        if self.layout == "NN":
            M, K = A.shape
            _, N = B.shape
        elif self.layout == "NT":
            M, K = A.shape
            N, _ = B.shape
        elif self.layout == "TN":
            K, M = A.shape
            _, N = B.shape
        elif self.layout == "TT":
            K, M = A.shape
            N, _ = B.shape
        else:
            raise RuntimeError("Unsupported layout:" + str(self.layout))
        var_values = [
            M, N, K,
            ceil(M, self.threadblock_problem_size[0]),
            ceil(N, self.threadblock_problem_size[1]),
            ceil(K, self.threadblock_problem_size[2])
        ]
        func(A, B, C, *var_values)

    def try_with(self, M, N, K, measure_opt=MeasureOptions(
            target="cuda", number=10,
            min_repeat_ms=80), new_process=False, dump=False):
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
        if self.layout == "NN":
            A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, N], dtype=self.in_dtype)
        elif self.layout == "NT":
            A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
            B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        elif self.layout == "TN":
            A = tvm.te.placeholder([K, M], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, N], dtype=self.in_dtype)
        elif self.layout == "TT":
            A = tvm.te.placeholder([K, M], dtype=self.in_dtype)
            B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        else:
            raise RuntimeError("Unsupported layout:" + str(self.layout))
        Output = tvm.te.placeholder([M, N], dtype=self.out_dtype)
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

    def expose_evaluate_context(self, M, N, K):
        if self.layout == "NN":
            A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, N], dtype=self.in_dtype)
        elif self.layout == "NT":
            A = tvm.te.placeholder([M, K], dtype=self.in_dtype)
            B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        elif self.layout == "TN":
            A = tvm.te.placeholder([K, M], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, N], dtype=self.in_dtype)
        elif self.layout == "TT":
            A = tvm.te.placeholder([K, M], dtype=self.in_dtype)
            B = tvm.te.placeholder([N, K], dtype=self.in_dtype)
        else:
            raise RuntimeError("Unsupported layout:" + str(self.layout))
        Output = tvm.te.placeholder([M, N], dtype=self.out_dtype)
        args = [A, B, Output]
        var_values = [
            M, N, K,
            ceil(M, self.threadblock_problem_size[0]),
            ceil(N, self.threadblock_problem_size[1]),
            ceil(K, self.threadblock_problem_size[2])
        ]
        return args, var_values


class Conv2dOperator(Operator):
    def __init__(self,
                 target,
                 get_conv2d_implementation,
                 in_dtype, out_dtype,
                 threadblock_problem_size,
                 warp_problem_size,
                 instruction_problem_size,
                 epilogues,
                 stride,
                 padding,
                 dilation,
                 split_K,
                 type_name,
                 arch,
                 code,
                 tag,
                 layout,
                 algorithm,
                 strategy):
        super(Conv2dOperator, self).__init__()
        self.target = target
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.threadblock_problem_size = threadblock_problem_size
        self.warp_problem_size = warp_problem_size
        self.instruction_problem_size = instruction_problem_size
        self.epilogues = []
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.split_K = split_K
        self.layout = layout
        self.algorithm = algorithm
        self.strategy = strategy
        if self.split_K > 1:
            self.get_context = lambda *_: get_conv2d_implementation(type_name, arch, code, tag, layout, algorithm, strategy)(
                self.threadblock_problem_size,
                self.warp_problem_size,
                self.instruction_problem_size,
                self.epilogues,
                A_dtype=self.in_dtype,
                B_dtype=self.in_dtype,
                C_dtype=self.out_dtype,
                split_K=split_K
            )
        else:
            self.get_context = lambda *_: get_conv2d_implementation(type_name, arch, code, tag, layout, algorithm, strategy)(
                self.threadblock_problem_size,
                self.warp_problem_size,
                self.instruction_problem_size,
                self.epilogues,
                A_dtype=self.in_dtype,
                B_dtype=self.in_dtype,
                C_dtype=self.out_dtype
            )

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

    def expose_evaluate_context(self, N, C, H, W, K, R, S, stride, padding, dilation):
        stride = (stride, stride) if isinstance(stride, int) else stride
        assert isinstance(stride, (list, tuple)) and len(stride) == 2
        padding = (padding, padding) if isinstance(padding, int) else padding
        assert isinstance(padding, (list, tuple)) and len(padding) == 2
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        assert isinstance(dilation, (list, tuple)) and len(dilation) == 2
        pR = (R - 1) * dilation[0] + 1
        pS = (S - 1) * dilation[1] + 1
        
        P = (H + 2 * padding[0] - pR) // stride[0] + 1
        Q = (W + 2 * padding[1] - pS) // stride[1] + 1

        mM = N * P * Q
        mN = K
        mK = C * R * S
        if self.layout == "nchw":
            A = tvm.te.placeholder([N, C, H, W], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, C, R, S], dtype=self.in_dtype)
            Output = tvm.te.placeholder([N, K, P, Q], dtype=self.out_dtype)
            args = [A, B, Output]
            var_values = [
                N, C, H, W, K, R, S,
                ceil(mM, self.threadblock_problem_size[0]),
                ceil(mN, self.threadblock_problem_size[1]),
                ceil(mK, self.threadblock_problem_size[2])
            ]
            return args, var_values
        elif self.layout == "nhwc":
            A = tvm.te.placeholder([N, H, W, C], dtype=self.in_dtype)
            B = tvm.te.placeholder([K, C, R, S], dtype=self.in_dtype)
            Output = tvm.te.placeholder([N, P, Q, K], dtype=self.out_dtype)
            args = [A, B, Output]
            var_values = [
                N, C, H, W, K, R, S,
                ceil(mM, self.threadblock_problem_size[0]),
                ceil(mN, self.threadblock_problem_size[1]),
                ceil(mK, self.threadblock_problem_size[2])
            ]
            return args, var_values
        else:
            raise RuntimeError("Unsupported Layout: " + str(self.layout))

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
        else:
            raise RuntimeError("Unsupported Layout: " + str(self.layout))
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