import tvm
import time
import numpy as np
from tvm import saber
from tvm.saber.utils import array_view


class Conv2dContext(object):
    def __init__(self, sch, inputs, outputs, params, param_func):
        self.sch = sch
        self.inputs = inputs
        self.outputs = outputs
        self.params = params
        self.param_func = param_func


class Conv2dCUDAOptimizeContext(object):
    def __init__(self):
        self.tb_tiles = [1, 1, 1]
        self.wp_tiles = [32, 16, 64]
        self.td_tiles = [1, 1, 4]


def conv2d_cuda_nhwc(image_prologues, kernel_prologues, epilogues,
                     R=3, S=3, pad=1, stride=1, dilation=1,
                     conv_opt_context=Conv2dCUDAOptimizeContext()):
    """
    Params:
    --------
    image_prologues: List of elementwise operator for image
    kernel_prologues: List of elementwise operator for kernel
    epilogues: List of elementwise operator for output
    """
    N = tvm.tir.Var("N", "int32")
    K = tvm.tir.Var("K", "int32")
    H = tvm.tir.Var("H", "int32")
    W = tvm.tir.Var("W", "int32")
    C = tvm.tir.Var("C", "int32")
    pH = H + 2 * pad
    pW = W + 2 * pad

    pR = (R - 1) * dilation + 1
    pS = (S - 1) * dilation + 1

    P = (pH - pR) // stride + 1
    Q = (pW - pS) // stride + 1
    A = tvm.te.placeholder([N, H, W, C], dtype="float32", name="A")
    # preprocess A
    A_prologue_tensors = []
    for pro in image_prologues:
        A = pro(A)
        A_prologue_tensors.append(A)
    # preprocess B
    B_prologue_tensors = []
    B = tvm.te.placeholder([K, C, R, S], dtype="float32", name="B")
    for pro in kernel_prologues:
        B = pro(B)
        B_prologue_tensors.append(B)
    padded = tvm.te.compute([N, pH, pW, C], lambda n, ph, pw, c:
                            tvm.tir.if_then_else(
        tvm.tir.all(ph >= pad, ph < pH - pad, pw >= pad, pw < pW - pad),
        A[n, ph - pad, pw - pad, c],
        tvm.tir.const(0, A.dtype)
    ), name="padded")

    def ceil(a, b):
        return (a + b - 1) // b

    mo = tvm.tir.Var("mo", "int32")
    no = tvm.tir.Var("no", "int32")
    ko = tvm.tir.Var("ko", "int32")

    tb_m, tb_n, tb_k = conv_opt_context.tb_tiles
    wp_m, wp_n, wp_k = conv_opt_context.wp_tiles
    td_m, td_n, td_k = conv_opt_context.td_tiles

    def get_M(a, b, c, d):
        return ((a * tb_m + b) * wp_m + c) * td_m + d

    def get_N(a, b, c, d):
        return ((a * tb_n + b) * wp_n + c) * td_n + d

    def get_K(a, b, c, d):
        return ((a * tb_k + b) * wp_k + c) * td_k + d

    def get_n(a, b, c, d):
        return get_N(a, b, c, d) // (P * Q)

    def get_k(a, b, c, d):
        return get_M(a, b, c, d)

    def get_p(a, b, c, d):
        return get_N(a, b, c, d) % (P * Q) // Q

    def get_q(a, b, c, d):
        return get_N(a, b, c, d) % Q

    def get_c(a, b, c, d):
        return get_K(a, b, c, d) // (R * S)

    def get_r(a, b, c, d):
        return get_K(a, b, c, d) % (R * S) // S

    def get_s(a, b, c, d):
        return get_K(a, b, c, d) % S

    # B_matrix = tvm.te.compute(
    #     [mo, ko, tb_m, tb_k, wp_m, td_m, wp_k, td_k],
    #     lambda imo, iko, imi, iki, imii, imiii, ikii, ikiii:
    #         tvm.tir.if_then_else(
    #             tvm.tir.all(
    #                 get_M(imo, imi, imii, imiii) < K,
    #                 get_K(iko, iki, ikii, ikiii) < C*R*S),
    #             B[get_k(imo, imi, imii, imiii),
    #             get_c(iko, iki, ikii, ikiii),
    #             get_r(iko, iki, ikii, ikiii),
    #             get_s(iko, iki, ikii, ikiii)],
    #             tvm.tir.const(0, B.dtype)
    #         ),
    #         name="B_matrix")

    # A_matrix = tvm.te.compute(
    #     [no, ko, tb_n, tb_k, wp_n, td_n, wp_k, td_k],
    #     lambda ino, iko, ini, iki, inii, iniii, ikii, ikiii:
    #         tvm.tir.if_then_else(
    #             tvm.tir.all(
    #                 get_N(ino, ini, inii, iniii) < N*P*Q,
    #                 get_K(iko, iki, ikii, ikiii) < C*R*S),
    #             padded[get_n(ino, ini, inii, iniii),
    #                 get_p(ino, ini, inii, iniii) * stride +
    #                 get_r(iko, iki, ikii, ikiii) * dilation,
    #                 get_q(ino, ini, inii, iniii) * stride +
    #                 get_s(iko, iki, ikii, ikiii) * dilation,
    #                 get_c(iko, iki, ikii, ikiii),
    #                 ],
    #             tvm.tir.const(0, padded.dtype)
    #         ),
    #     name="A_matrix"
    # )

    # rko = tvm.te.reduce_axis([0, ko], "rko")
    # rki = tvm.te.reduce_axis([0, tb_k], "rki")
    # rkii = tvm.te.reduce_axis([0, wp_k], "rkii")
    # rkiii = tvm.te.reduce_axis([0, td_k], "rkiii")
    # C_matrix = tvm.te.compute(
    #     [mo, no, tb_m, tb_n, wp_m, wp_n, td_m, td_n],
    #     lambda imo, ino, imi, ini, imii, inii, imiii, iniii:
    #         tvm.te.sum(
    #             A_matrix[ino, rko, ini, rki, inii, iniii, rkii, rkiii] *
    #             B_matrix[imo, rko, imi, rki, imii, imiii, rkii, rkiii],
    #             axis=[rko, rki, rkii, rkiii]
    #     ),
    #     name="C_matrix")

    B_matrix = tvm.te.compute(
        [mo, ko, tb_m, tb_k, wp_m, td_m, wp_k, td_k],
        lambda imo, iko, imi, iki, imii, imiii, ikii, ikiii:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_M(imo, imi, imii, imiii) < K,
                    get_K(iko, iki, (ikii + imii % wp_k) % wp_k, ikiii) < C*R*S),
                B[get_k(imo, imi, imii, imiii),
                get_c(iko, iki, (ikii + imii % wp_k) % wp_k, ikiii),
                get_r(iko, iki, (ikii + imii % wp_k) % wp_k, ikiii),
                get_s(iko, iki, (ikii + imii % wp_k) % wp_k, ikiii)],
                tvm.tir.const(0, B.dtype)
            ),
            name="B_matrix")

    A_matrix = tvm.te.compute(
        [no, ko, tb_n, tb_k, wp_n, td_n, wp_k, td_k],
        lambda ino, iko, ini, iki, inii, iniii, ikii, ikiii:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    get_N(ino, ini, inii, iniii) < N*P*Q,
                    get_K(iko, iki, (ikii + inii % wp_k) % wp_k, ikiii) < C*R*S),
                padded[get_n(ino, ini, inii, iniii),
                    get_p(ino, ini, inii, iniii) * stride +
                    get_r(iko, iki, (ikii + inii % wp_k) % wp_k, ikiii) * dilation,
                    get_q(ino, ini, inii, iniii) * stride +
                    get_s(iko, iki, (ikii + inii % wp_k) % wp_k, ikiii) * dilation,
                    get_c(iko, iki, (ikii + inii % wp_k) % wp_k, ikiii),
                    ],
                tvm.tir.const(0, padded.dtype)
            ),
        name="A_matrix"
    )

    rko = tvm.te.reduce_axis([0, ko], "rko")
    rki = tvm.te.reduce_axis([0, tb_k], "rki")
    rkii = tvm.te.reduce_axis([0, wp_k], "rkii")
    rkiii = tvm.te.reduce_axis([0, td_k], "rkiii")
    C_matrix = tvm.te.compute(
        [mo, no, tb_m, tb_n, wp_m, wp_n, td_m, td_n],
        lambda imo, ino, imi, ini, imii, inii, imiii, iniii:
            tvm.te.sum(
                A_matrix[ino, rko, ini, rki, inii, iniii, (rkii - inii % wp_k + wp_k) % wp_k, rkiii] *
                B_matrix[imo, rko, imi, rki, imii, imiii, (rkii - imii % wp_k + wp_k) % wp_k, rkiii],
                axis=[rko, rki, rkii, rkiii]
        ),
        name="C_matrix")

    epilogue_tensors = []
    Epilogue = C_matrix
    for epi in epilogues:
        Epilogue = epi(Epilogue)
        epilogue_tensors.append(Epilogue)

    Output = tvm.te.compute(
        [no, tb_n, wp_n, td_n, mo, tb_m, wp_m, td_m],
        lambda ino, ini, inii, iniii, imo, imi, imii, imiii, :
            Epilogue[imo, ino, imi, ini, imii, inii, imiii, iniii],
        name="Output")

    sch = tvm.te.create_schedule([Output.op])

    for pro in A_prologue_tensors:
        sch[pro].compute_inline()
    for pro in B_prologue_tensors:
        sch[pro].compute_inline()
    sch[padded].compute_inline()
    for epi in epilogue_tensors:
        sch[epi].compute_inline()

    ino, ini, inii, iniii, imo, imi, imii, imiii = sch[Output].op.axis
    sch[Output].reorder(ino, imo, ini, imi, inii, imii, iniii, imiii)
    block_x = tvm.te.thread_axis("blockIdx.x")
    block_y = tvm.te.thread_axis("blockIdx.y")
    block_z = tvm.te.thread_axis("blockIdx.z")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    thread_y = tvm.te.thread_axis("threadIdx.y")
    thread_z = tvm.te.thread_axis("threadIdx.z")
    sch[Output].bind(ino, block_y)
    sch[Output].bind(imo, block_x)
    sch[Output].bind(inii, thread_y)  # 2
    sch[Output].bind(imii, thread_x)  # 2
    # iniio, iniii = sch[Output].split(inii, nparts=32)
    # imiio, imiii = sch[Output].split(imii, nparts=8)
    # sch[Output].reorder(iniio, imiio, iniii, imiii)
    # fused = sch[Output].fuse(iniio, imiio)
    # sch[Output].bind(iniio, thread_x)  # 32
    sch[Output].unroll(ini)
    sch[Output].unroll(imi)

    sch[C_matrix].compute_at(sch[Output], imii)

    imo, ino, imi, ini, imii, inii, imiii, iniii = sch[C_matrix].op.axis
    rko, rki, rkii, rkiii = sch[C_matrix].op.reduce_axis
    sch[C_matrix].reorder(imo, ino, rko, imi, ini, rki,
                          rkii, imii, inii, imiii, iniii, rkiii)
    # sch[C_matrix].unroll(rko)
    sch[C_matrix].unroll(rki)
    sch[C_matrix].unroll(rkii)
    sch[C_matrix].unroll(imiii)
    sch[C_matrix].unroll(iniii)
    sch[A_matrix].set_scope("shared")
    sch[B_matrix].set_scope("shared")
    sch[C_matrix].set_scope("local")
    sch[A_matrix].compute_at(sch[C_matrix], rko)
    sch[B_matrix].compute_at(sch[C_matrix], rko)

    ino, iko, ini, iki, inii, iniii, ikii, ikiii = sch[A_matrix].op.axis
    fused = sch[A_matrix].fuse(ini, iki, inii, iniii, ikii)
    fused, tx = sch[A_matrix].split(fused, factor=wp_m)
    fused, ty = sch[A_matrix].split(fused, factor=wp_n)
    # fused, tz = sch[A_matrix].split(fused, factor=tb_n)
    sch[A_matrix].vectorize(ikiii)
    sch[A_matrix].bind(tx, thread_x)
    sch[A_matrix].bind(ty, thread_y)
    # sch[A_matrix].bind(tz, thread_z)

    imo, iko, imi, iki, imii, imiii, ikii, ikiii = sch[B_matrix].op.axis
    fused = sch[B_matrix].fuse(imi, iki, imii, imiii, ikii)
    fused, tx = sch[B_matrix].split(fused, factor=wp_m)
    fused, ty = sch[B_matrix].split(fused, factor=wp_n)
    # fused, tz = sch[B_matrix].split(fused, factor=tb_n)
    sch[B_matrix].vectorize(ikiii)
    sch[B_matrix].bind(tx, thread_x)
    sch[B_matrix].bind(ty, thread_y)
    # sch[B_matrix].bind(tz, thread_z)

    Vars = [N, K, H, W, C, mo, no, ko]

    def get_params(param_N, param_K, param_H, param_W, param_C):
        param_P = (param_H + 2 * pad - (R - 1) * dilation - 1) // stride + 1
        param_Q = (param_W + 2 * pad - (S - 1) * dilation - 1) // stride + 1
        param_mo = ceil(param_K, tb_m * wp_m * td_m)
        param_no = ceil(param_N * param_P * param_Q, tb_n * wp_n * td_n)
        param_ko = ceil(param_C * R * S, tb_k * wp_k * td_k)
        Params = [param_N, param_K, param_H, param_W,
                  param_C, param_mo, param_no, param_ko]
        return Params

    return Conv2dContext(
        sch,
        [A, B],
        [Output],
        Vars,
        get_params
    )


class Conv2d(object):
    def __init__(self, kernel_size, padding, stride, dilation,
                 layout="nhwc", target="cuda"):
        self.kernel_size = [kernel_size, kernel_size] if isinstance(
            kernel_size, int) else kernel_size
        assert isinstance(self.kernel_size, (list, tuple))
        assert len(self.kernel_size) == 2
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        assert isinstance(padding, int) and isinstance(
            stride, int) and isinstance(dilation, int)
        assert layout in ["nhwc"]
        self.target = target
        if target == "cuda":
            self.opt_context = Conv2dCUDAOptimizeContext()
            self.conv2d_func = conv2d_cuda_nhwc
        elif target == "llvm":
            self.opt_context = Conv2dCUDAOptimizeContext()
            self.conv2d_func = conv2d_cuda_nhwc
        else:
            raise RuntimeError("Target not supported: " + target)
        self.context = None
        self.runtime_func = None

    def compile(self, image_prologues, kernel_prologues, epilogues):
        self.context = self.conv2d_func(
            image_prologues, kernel_prologues, epilogues,
            self.kernel_size[0], self.kernel_size[1],
            self.padding, self.stride, self.dilation,
            self.opt_context)
        # print(tvm.lower(
        #     self.context.sch,
        #     self.context.inputs + self.context.outputs + self.context.params,
        #     simple_mode=True))
        self.runtime_func = tvm.build(
            self.context.sch,
            self.context.inputs + self.context.outputs + self.context.params,
            target=self.target
        )

    def run(self, ctx, A, B, Output, eval_time=False, number=20, min_repeat_ms=150):
        """
        Params:
        --------
        ctx: tvm.context
        A: tvm.nd.array
        B: tvm.nd.array
        Output: tvm.nd.array
        """
        N, H, W, C = [int(x) for x in A.shape]
        K, _, R, S = [int(x) for x in B.shape]
        Params = self.context.param_func(N, K, H, W, C)
        param_N, param_K, param_H, param_W, param_C, param_mo, param_no, param_ko = Params
        tb_m, tb_n, _ = self.opt_context.tb_tiles
        wp_m, wp_n, _ = self.opt_context.wp_tiles
        td_m, td_n, _ = self.opt_context.td_tiles
        Output_view = array_view(
            Output, [param_no, tb_n, wp_n, td_n, param_mo, tb_m, wp_m, td_m])
        if eval_time:
            time_evaluator = self.runtime_func.time_evaluator(
                self.runtime_func.entry_name, ctx, number=number, min_repeat_ms=min_repeat_ms)
            cost = time_evaluator(A, B, Output_view, *Params).mean
            return cost * 1e3
        else:
            self.runtime_func(A, B, Output_view, *Params)
            return 0

    def infer_shape(self, image_shape, filter_shape):
        N, H, W, C = [int(x) for x in image_shape]
        K, kC, R, S = [int(x) for x in filter_shape]
        assert C == kC
        assert R == self.kernel_size[0] and S == self.kernel_size[1]
        k_h = (R - 1) * self.dilation + 1
        k_w = (S - 1) * self.dilation + 1
        P = (H + 2 * self.padding - k_h) // self.stride + 1
        Q = (W + 2 * self.padding - k_w) // self.stride + 1
        return [N, P, Q, K]

    def test(self, N, H, W, C, K, number=20, min_repeat_ms=150, verify=False):
        output_shape = self.infer_shape([N, H, W, C], [K, C, *self.kernel_size])
        assert self.runtime_func is not None, "Please call compile first"
        assert self.context is not None
        Params = self.context.param_func(N, K, H, W, C)
        param_N, param_K, param_H, param_W, param_C, param_mo, param_no, param_ko = Params
        # print(param_no, param_mo, param_ko)
        tb_m, tb_n, _ = self.opt_context.tb_tiles
        wp_m, wp_n, _ = self.opt_context.wp_tiles
        td_m, td_n, _ = self.opt_context.td_tiles
        A, B = self.context.inputs
        Output, = self.context.outputs
        A_np = np.random.uniform(-1, 1, [N, H, W, C]).astype(A.dtype)
        B_np = np.random.uniform(-1, 1, [K, C, *self.kernel_size]).astype(B.dtype)
        # A_np = np.ones([N, H, W, C]).astype(A.dtype)
        # B_np = np.ones([K, C, *self.kernel_size]).astype(B.dtype)
        Output_np = np.zeros(
            [param_no, tb_n, wp_n, td_n, param_mo, tb_m, wp_m, td_m],
            dtype=Output.dtype)
        ctx = tvm.context(self.target)
        A_tvm = tvm.nd.array(A_np, ctx)
        B_tvm = tvm.nd.array(B_np, ctx)
        Output_tvm = tvm.nd.array(Output_np, ctx)
        cost = self.run(ctx, A_tvm, B_tvm, Output_tvm, True, number, min_repeat_ms)
        if verify:
            import torch
            A_torch = torch.tensor(A_np)
            B_torch = torch.tensor(B_np)
            A_torch = A_torch.permute(0, 3, 1, 2)
            Output_torch = torch.nn.functional.conv2d(A_torch, B_torch,
                stride=self.stride, padding=self.padding, dilation=self.dilation)
            Output_torch = Output_torch.permute(0, 2, 3, 1)
            # Output_tvm = tvm.nd.array(Output_np, ctx)
            # self.run(ctx, A_tvm, B_tvm, Output_tvm, False)
            Output_tvm = Output_tvm.asnumpy()
            Output_tvm = Output_tvm.reshape(
                param_no * tb_n * wp_n * td_n, param_mo * tb_m * wp_m * td_m)
            N, P, Q, K = Output_torch.shape
            Output_tvm = Output_tvm[:N*P*Q, :K]
            Output_tvm = Output_tvm.reshape([N, P, Q, K])
            from tvm import testing
            testing.assert_allclose(Output_torch.numpy(), Output_tvm, rtol=1e-3, atol=1e-3)
        return cost
