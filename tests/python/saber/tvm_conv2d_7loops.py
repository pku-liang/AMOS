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
    A = tvm.te.placeholder([N, C, H, W], dtype="float32", name="A")
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
    Output = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
        tvm.te.sum(
            padded[
                n,
                rc,
                p * stride + rr * dilation,
                q * stride + rs * dilation]
            * B[k, rc, rr, rs], axis=[rc, rr, rs]), name="Output")

    
    sch = tvm.te.create_schedule(Output.op)

    sch[padded].compute_inline()
    # AA = sch.cache_read(A, "shared", [padded])
    # BB = sch.cache_read(B, "shared", [Output])
    LL = sch.cache_write(Output, "local")

    n, k, p, q = sch[Output].op.axis
    factors_n = [1, 1, 1, 1]
    factors_k = [8, 1, 64, 1]
    factors_p = [1, 1, 7, 1]
    factors_q = [7, 1, 1, 1]
    factors_rc = [8, 1, 64]
    factors_rr = [1, 1, 3]
    factors_rs = [1, 1, 3]
    def tile_axis(s, op, axis, factors):
        ret = []
        for f in reversed(factors[1:]):
            axis, inner = s[op].split(axis, factor=f)
            ret.append(inner)
        ret.append(axis)
        return list(reversed(ret))
    
    no, nv, nt, ni = tile_axis(sch, Output, n, factors_n)
    ko, kv, kt, ki = tile_axis(sch, Output, k, factors_k)
    po, pv, pt, pi = tile_axis(sch, Output, p, factors_p)
    qo, qv, qt, qi = tile_axis(sch, Output, q, factors_q)
    sch[Output].reorder(no, ko, po, qo, nv, kv, pv, qv, nt, kt, pt, qt, ni, ki, pi, qi)
    bz = sch[Output].fuse(no, ko)
    tz = sch[Output].fuse(nt, kt)
    block_x = tvm.te.thread_axis("blockIdx.x")
    block_y = tvm.te.thread_axis("blockIdx.y")
    block_z = tvm.te.thread_axis("blockIdx.z")
    thread_x = tvm.te.thread_axis("threadIdx.x")
    thread_y = tvm.te.thread_axis("threadIdx.y")
    thread_z = tvm.te.thread_axis("threadIdx.z")

    sch[Output].bind(bz, block_z)
    sch[Output].bind(po, block_y)
    sch[Output].bind(qo, block_x)
    sch[Output].bind(tz, thread_z)
    sch[Output].bind(pt, thread_y)
    sch[Output].bind(qt, thread_x)

    sch[LL].compute_at(sch[Output], qt)
    n, k, p, q = sch[LL].op.axis
    rc, rr, rs = sch[LL].op.reduce_axis
    rco, rcm, rci = tile_axis(sch, LL, rc, factors_rc)
    rro, rrm, rri = tile_axis(sch, LL, rr, factors_rr)
    rso, rsm, rsi = tile_axis(sch, LL, rs, factors_rs)
    sch[LL].reorder(rco, rro, rso, rcm, rrm, rsm, n, k, p, q, rci, rri, rsi)

    # sch[AA].compute_at(sch[LL], rso)
    # sch[BB].compute_at(sch[LL], rso)
    # for SS in [AA, BB]:
    #     n, c, h, w = sch[SS].op.axis
    #     fused = sch[SS].fuse(n, c, h, w)
    #     fused, vec = sch[SS].split(fused, factor=4)
    #     fused, tx = sch[SS].split(fused, factor=factors_q[2])
    #     fused, ty = sch[SS].split(fused, factor=factors_p[2])
    #     fused, tz = sch[SS].split(fused, factor=factors_n[2] * factors_k[2])
    #     sch[SS].bind(tx, thread_x)
    #     sch[SS].bind(ty, thread_y)
    #     sch[SS].bind(tz, thread_z)
    #     sch[SS].vectorize(vec)

    Vars = [N, K, H, W, C]
    param_N = 1
    param_K = 512
    param_H = 7
    param_W = 7
    param_C = 512
    # param_stride = 1
    # param_dilation = 1
    param_P = (param_H + 2 * pad - (R - 1) * dilation - 1) // stride + 1
    param_Q = (param_W + 2 * pad - (S - 1) * dilation - 1) // stride + 1
    Params = [param_N, param_K, param_H, param_W, param_C]
    print(tvm.lower(sch, [A, B, Output] + Vars, simple_mode=True))
    target = "cuda"
    beg = time.time()
    func = tvm.build(sch, [A, B, Output] + Vars, target=target)
    end = time.time()
    print("Compile time is", (end - beg) * 1e3, "ms")
    np_A = np.random.uniform(-1, 1, [param_N, param_C, param_H, param_W]).astype(A.dtype)
    np_B = np.random.uniform(-1, 1, [param_K, param_C, R, S]).astype(B.dtype)
    np_C = np.zeros([param_N, param_K, param_P, param_Q], dtype=Output.dtype)
    ctx = tvm.context(target)
    tvm_A = tvm.nd.array(np_A, ctx)
    tvm_B = tvm.nd.array(np_B, ctx)
    tvm_C = tvm.nd.array(np_C, ctx)
    # func(tvm_A, tvm_B, tvm_C, *Params)

    timed_func = func.time_evaluator(func.entry_name, ctx, number=20)
    cost = timed_func(tvm_A, tvm_B, tvm_C, *Params).mean
    print("Execution time is", cost * 1e3, "ms")


if __name__ == "__main__":
    conv2d();
