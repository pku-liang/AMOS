import tvm
from functools import reduce


def index(name):
    return tvm.tir.Var(name, "int32")

def allmul(lst):
    return reduce(lambda i, j: i * j: lst, 1)

def ceil(a, b):
    return (a + b - 1) // b


class Conv2dDirectOptContextCUDA(object):
    def __init__(self):
        # threadblock, [serial, thread, serial]
        self.N_tiles = [1, 1, 1]
        self.K_tiles = [1, 32, 1]
        self.P_tiles = [1, 2, 1]
        self.Q_tiles = [1, 2, 1]
        self.C_tiles = [1, 16, 4]
        self.R_tiles = [1, 1, 1]
        self.S_tiles = [1, 1, 1]
        
        self.A_shared_load_stages = 1
        self.B_shared_load_stages = 1
        self.A_shared_relayout = 0
        self.B_shared_relayout = 0
        self.A_crosswise = 0
        self.B_crosswise = 0
        self.A_use_fragment = 0
        self.B_use_fragment = 0
        self.use_parallel_reduce = 0


def conv2d_direct_nchw_cuda(
    R, S, stride, pad, dilation,
    elem_prologue_image, elem_prologue_filter,
    epilogue, in_dtype = "float32", out_dtype = "float32",
    ctx=Conv2dDirectOptContextCUDA()):
    N = index("N")
    K = index("K")
    H = index("H")
    W = index("W")
    C = index("C")
    pH = H + 2 * pad
    pW = W + 2 * pad
    pR = (R - 1) * dilation + 1
    pS = (S - 1) * dilation + 1
    P = (pH - pR) // stride + 1
    Q = (pW - pS) // stride + 1
    Image = tvm.te.placeholder([N, C, H, W], dtype=in_dtype, name="Image")
    Filter = tvm.te.placeholder([K, C, R, S], dtype=in_dtype, name="Filter")
    image_pro_tensors = []
    filter_pro_tensors = []
    A = Image
    B = Filter
    for pro in elem_prologue_image:
        A = pro(A)
        image_pro_tensors.append(A)
    for pro in elem_prologue_filter:
        B = pro(B)
        filter_pro_tensors.append(B)
    padded = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, ph, pw:
            tvm.tir.if_then_else(
                tvm.tir.all(ph >= pad, ph < pH - pad, pw >= pad, pw < pW - pad),
                A[n, c, ph - pad, pw - pad],
                tvm.tir.const(0, A.dtype)
            ),
        name="padded"
    )
    No = index("No")
    Ko = index("Ko")
    Qo = index("Qo")
    Po = index("Po")
    Co = index("Co")
    N_tiles = [No, *ctx.N_tiles]
    K_tiles = [Ko, *ctx.K_tiles]
    P_tiles = [Po, *ctx.P_tiles]
    Q_tiles = [Qo, *ctx.Q_tiles]
    C_tiles = [Co, *ctx.C_tiles]
    R_tiles = [ceil(R, allmul(ctx.R_tiles)), *ctx.R_tiles]
    S_tiles = [ceil(S, allmul(ctx.S_tiles)), *ctx.S_tiles]

    A_shared_shapes = [N_tiles, C_tiles, ]
    
    if ctx.A_shared_load_stages == 1:
        A_tb_frag = None
        A_shared = tvm.te.compute(

        )
    elif ctx.A_shared_load_stages == 2:
        pass
    else:
        A_tb_frag = None
        A_shared = None

    if ctx.B_shared_load_stages == 1:
        pass
    elif ctx.B_shared_load_stages == 2:
        pass




    C = tvm.te.compute(
        [*split_N(), *split_K(), *split_P(), *split_Q()],
        lambda no, , k, p, q:
            tvm.te.sum(
                padded[n, rc, p * stride + rr * dilation, q * stride + rs * dilation] *
                B[k, rc, rr, rs], axis=[rc, rr, rs]),
        name="C"
    )