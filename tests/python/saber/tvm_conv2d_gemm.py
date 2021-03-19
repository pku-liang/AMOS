import tvm
import time
import numpy as np


def conv2d(R=3, S=3, pad=1):
    N = tvm.tir.Var("N", "int32")
    K = tvm.tir.Var("K", "int32")
    H = tvm.tir.Var("H", "int32")
    W = tvm.tir.Var("W", "int32")
    C = tvm.tir.Var("C", "int32")
    pH = H + 2 * pad
    pW = W + 2 * pad
    # stride = tvm.tir.Var("stride", "int32")
    # dilation = tvm.tir.Var("dilation", "int32")
    stride = 1
    dilation = 1
    pR = (R - 1) * dilation + 1
    pS = (S - 1) * dilation + 1
    P = (pH - pR) // stride + 1
    Q = (pW - pS) // stride + 1
    A = tvm.te.placeholder([N, H, W, C], dtype="float32", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float32", name="B")
    rc = tvm.te.reduce_axis([0, C], "rc")
    rr = tvm.te.reduce_axis([0, R], "rr")
    rs = tvm.te.reduce_axis([0, S], "rs")
    padded = tvm.te.compute([N, C, pH, pW], lambda n, c, ph, pw:
                            tvm.tir.if_then_else(
        tvm.tir.all(ph >= pad, ph < pH - pad, pw >= pad, pw < pW - pad),
        A[n, c, ph - pad, pw - pad],
        tvm.tir.const(0, A.dtype)
    ), name="padded")

    def ceil(a, b):
        return (a + b - 1) // b

    tb_m = 2
    tb_n = 2
    tb_k = 1
    wp_m = 4
    wp_n = 2
    wp_k = 8
    td_m = 16
    td_n = 16
    td_k = 4
    Mm = K
    Mn = N * P * Q
    Mk = C * R * S
    mo = tvm.tir.Var("mo", "int32")
    no = tvm.tir.Var("no", "int32")
    ko = tvm.tir.Var("ko", "int32")

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

    B_matrix = tvm.te.compute(
        [mo, ko, tb_m, tb_k, wp_m, wp_k, td_m, td_k],
        lambda imo, iko, imi, iki, imii, ikii, imiii, ikiii:
            B[get_k(imo, imi, imii, imiii), get_c(iko, iki, ikii, ikiii),
                   get_r(iko, iki, ikii, ikiii), get_s(iko, iki, ikii, ikiii)],
            name="B_matrix")
    
    A_matrix = tvm.te.compute(
        [no, ko, tb_n, tb_k, wp_n, wp_k, td_n, td_k],
        lambda ino, iko, ini, iki, inii, ikii, iniii, ikiii:
            padded[get_n(ino, ini, inii, iniii), get_c(iko, iki, ikii, ikiii),
                get_p(ino, ini, inii, iniii) * stride + get_r(iko, iki, ikii, ikiii) * dilation,
                get_q(ino, ini, inii, iniii) * stride + get_s(iko, iki, ikii, ikiii) * dilation
            ],
        name="A_matrix"
    )

    rko = tvm.te.reduce_axis([0, ko], "rko")
    rki = tvm.te.reduce_axis([0, tb_k], "rki")
    rkii = tvm.te.reduce_axis([0, wp_k], "rkii")
    rkiii = tvm.te.reduce_axis([0, tb_k], "rkiii")
    C_matrix = tvm.te.compute(
        [mo, no, tb_m, tb_n, wp_m, wp_n, td_m, td_n],
        lambda imo, ino, imi, ini, imii, inii, imiii, iniii:
            tvm.te.sum(
                A_matrix[ino, rko, ini, rki, inii, rkii, iniii, rkiii] *
                B_matrix[imo, rko, imi, rki, imii, rkii, imiii, rkiii],
                axis=[rko, rki, rkii, rkiii]
            ),
        name="C_matrix")

    Output = tvm.te.compute(
        [no, tb_n, wp_n, td_n, mo, tb_m, wp_m, td_m],
        lambda ino, ini, inii, iniii, imo, imi, imii, imiii, :
            C_matrix[imo, ino, imi, ini, imii, inii, imiii, iniii],
        name="Output")

    sch = tvm.te.create_schedule([Output.op])

    sch[padded].compute_inline()
    ino, ini, inii, iniii, imo, imi, imii, imiii = sch[Output].op.axis
    sch[Output].reorder(ino, imo, ini, imi, inii, iniii, imii, imiii)
    block_x = tvm.te.thread_axis("blockIdx.x")
    block_y = tvm.te.thread_axis("blockIdx.y")
    block_z = tvm.te.thread_axis("blockIdx.z")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    thread_y = tvm.te.thread_axis("threadIdx.y")
    thread_z = tvm.te.thread_axis("threadIdx.z")
    sch[Output].bind(ino, block_y)
    sch[Output].bind(imo, block_x)
    sch[Output].bind(ini, thread_z) # 2
    sch[Output].bind(imi, thread_y) # 2
    fused = sch[Output].fuse(inii, iniii)
    sch[Output].bind(fused, thread_x) # 32

    sch[C_matrix].compute_at(sch[Output], imi)

    imo, ino, imi, ini, imii, inii, imiii, iniii = sch[C_matrix].op.axis
    rko, rki, rkii, rkiii = sch[C_matrix].op.reduce_axis
    sch[C_matrix].reorder(imo, ino, rko, imi, ini, rki, rkii, imii, inii, imiii, iniii, rkiii)
    sch[A_matrix].set_scope("shared")
    sch[B_matrix].set_scope("shared")
    sch[A_matrix].compute_at(sch[C_matrix], rko)
    sch[B_matrix].compute_at(sch[C_matrix], rko)

    ino, iko, ini, iki, inii, ikii, iniii, ikiii = sch[A_matrix].op.axis
    fused = sch[A_matrix].fuse(ini, iki, inii, ikii, iniii)
    fused, tx = sch[A_matrix].split(fused, factor=wp_n * td_n)
    fused, ty = sch[A_matrix].split(fused, factor=tb_m)
    fused, tz = sch[A_matrix].split(fused, factor=tb_n)
    sch[A_matrix].vectorize(ikiii)
    sch[A_matrix].bind(tx, thread_x)
    sch[A_matrix].bind(ty, thread_y)
    sch[A_matrix].bind(tz, thread_z)

    imo, iko, imi, iki, imii, ikii, imiii, ikiii = sch[B_matrix].op.axis
    fused = sch[B_matrix].fuse(imi, iki, imii, ikii, imiii)
    fused, tx = sch[B_matrix].split(fused, factor=wp_n * td_n)
    fused, ty = sch[B_matrix].split(fused, factor=tb_m)
    fused, tz = sch[B_matrix].split(fused, factor=tb_n)
    sch[B_matrix].vectorize(ikiii)
    sch[B_matrix].bind(tx, thread_x)
    sch[B_matrix].bind(ty, thread_y)
    sch[B_matrix].bind(tz, thread_z)

    Vars = [N, K, H, W, C, mo, no, ko]
    print(tvm.lower(sch, [A, B, Output] + Vars, simple_mode=True))
    param_N = 1
    param_K = 512
    param_H = 7
    param_W = 7
    param_C = 512
    param_P = (param_H + 2 * pad - (R - 1) * dilation - 1) // stride + 1
    param_Q = (param_W + 2 * pad - (S - 1) * dilation - 1) // stride + 1
    param_mo = ceil(param_K, tb_m * wp_m * td_m)
    param_no = ceil(param_N * param_P * param_Q, tb_n * wp_n * td_n)
    param_ko = ceil(param_C * R * S, tb_k * wp_k * td_k)
    Params = [param_N, param_K, param_H, param_W, param_C, param_mo, param_no, param_ko]

    target = "cuda"
    beg = time.time()
    func = tvm.build(sch, [A, B, Output] + Vars, target=target)
    end = time.time()
    print("Compile time is", (end - beg) * 1e3, "ms")
    np_A = np.random.uniform(-1, 1, [param_N, param_H, param_W, param_C]).astype(A.dtype)
    np_B = np.random.uniform(-1, 1, [param_K, param_C, R, S]).astype(B.dtype)
    np_C = np.zeros([param_no, tb_n, wp_n, td_n, param_mo, tb_m, wp_m, td_m], dtype=Output.dtype)
    ctx = tvm.context(target)
    tvm_A = tvm.nd.array(np_A, ctx)
    tvm_B = tvm.nd.array(np_B, ctx)
    tvm_C = tvm.nd.array(np_C, ctx)
    # func(tvm_A, tvm_B, tvm_C, *Params)

    timed_func = func.time_evaluator(func.entry_name, ctx, number=20)
    cost = timed_func(tvm_A, tvm_B, tvm_C, *Params).mean
    print("Execution time is", cost * 1e3, "ms")


if __name__ == "__main__":
    conv2d()
