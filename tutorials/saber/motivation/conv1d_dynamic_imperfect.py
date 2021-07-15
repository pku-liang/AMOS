import tvm
from tvm import saber


def tile_axis(sch, op, axis, factors):
    ret = []
    for f in reversed(factors[1:]):
        axis, inner = sch[op].split(axis, factor=f)
        ret.append(inner)
    ret.append(axis)
    return list(reversed(ret))


def conv1d_cpu(L_value, K_value, CI_value, CO_value):
    L_factors = [-1, 4]
    CI_factors = [-1, 4]
    CO_factors = [-1, 4]
    L = tvm.tir.Var("L", "int32")
    K = tvm.tir.Var("K", "int32")
    CI = tvm.tir.Var("CI", "int32")
    CO = tvm.tir.Var("CO", "int32")
    A = tvm.te.placeholder([CI, L], name="A")
    W = tvm.te.placeholder([CO, CI, K], name="W")
    AA = tvm.te.compute([CI, L], lambda i, j: A[i, j], name="AA")
    WW = tvm.te.compute([CO, CI, K], lambda i, j, k: W[i, j, k], name="WW")
    P = (L - K) + 1
    P_value = (L_value - K_value) + 1
    rc = tvm.te.reduce_axis([0, CI], name="rc")
    rk = tvm.te.reduce_axis([0, K], name="rk")
    OO = tvm.te.compute(
        [CO, P],
        lambda co, p:
            tvm.te.sum(
                AA[rc, p + rk] * WW[co, rc, rk],
                axis=[rc, rk]
            ),
        name="OO"
    )
    Out = tvm.te.compute([CO, P], lambda i, j: OO[i, j], name="Out")


    sch = tvm.te.create_schedule(Out.op)
    c, p = sch[OO].op.axis
    co, ci = tile_axis(sch, OO, c, CO_factors)
    po, pi = tile_axis(sch, OO, p, L_factors)
    rc, rk = sch[OO].op.reduce_axis
    rco, rci = tile_axis(sch, OO, rc, CI_factors)
    sch[OO].reorder(co, po, rco, rk, ci, pi, rci)

    sch[AA].compute_at(sch[OO], rk)
    sch[WW].compute_at(sch[OO], rk)

    measure_opt = saber.MeasureOptions(
        target="llvm", build_func="default", target_host="llvm", timeout=10,
        verbose=1, number=100, repeat=1, min_repeat_ms=500,
        cooldown_interval=1, enable_cpu_cache_flush=1
    )

    cost = saber.measure_base.evaluate_schedule(
        sch, [A, W, Out], [L, K, CI, CO],
        [tvm.te.placeholder([CI_value, L_value]),
         tvm.te.placeholder([CO_value, CI_value, K_value]),
         tvm.te.placeholder([CO_value, P_value])],
        [L_value, K_value, CI_value, CO_value],
        measure_opt, new_process=True)

    print(tvm.lower(sch, [A, W, Out, L, K, CI, CO], simple_mode=True))
    print("CPU Cost is", cost, "ms")


def conv1d_gpu(L_value, K_value, CI_value, CO_value):
    L_factors = [-1, 4]
    CI_factors = [-1, 4]
    CO_factors = [-1, 4]
    L = tvm.tir.Var("L", "int32")
    K = tvm.tir.Var("K", "int32")
    CI = tvm.tir.Var("CI", "int32")
    CO = tvm.tir.Var("CO", "int32")
    A = tvm.te.placeholder([CI, L], name="A")
    W = tvm.te.placeholder([CO, CI, K], name="W")
    AA = tvm.te.compute([CI, L], lambda i, j: A[i, j], name="AA")
    WW = tvm.te.compute([CO, CI, K], lambda i, j, k: W[i, j, k], name="WW")
    P = (L - K) + 1
    P_value = (L_value - K_value) + 1
    rc = tvm.te.reduce_axis([0, CI], name="rc")
    rk = tvm.te.reduce_axis([0, K], name="rk")
    OO = tvm.te.compute(
        [CO, P],
        lambda co, p:
            tvm.te.sum(
                AA[rc, p + rk] * WW[co, rc, rk],
                axis=[rc, rk]
            ),
        name="OO"
    )
    Out = tvm.te.compute([CO, P], lambda i, j: OO[i, j], name="Out")


    sch = tvm.te.create_schedule(Out.op)
    c, p = sch[OO].op.axis
    co, ci = tile_axis(sch, OO, c, CO_factors)
    po, pi = tile_axis(sch, OO, p, L_factors)
    rc, rk = sch[OO].op.reduce_axis
    rco, rci = tile_axis(sch, OO, rc, CI_factors)
    sch[OO].reorder(co, po, rco, rk, ci, pi, rci)

    sch[OO].bind(co, tvm.te.thread_axis("blockIdx.y"))
    sch[OO].bind(po, tvm.te.thread_axis("blockIdx.x"))
    sch[OO].bind(ci, tvm.te.thread_axis("threadIdx.y"))
    sch[OO].bind(pi, tvm.te.thread_axis("threadIdx.x"))

    sch[AA].set_scope("local")
    sch[WW].set_scope("local")
    sch[AA].compute_at(sch[OO], rk)
    sch[WW].compute_at(sch[OO], rk)

    sch[Out].bind(sch[Out].op.axis[0], tvm.te.thread_axis("blockIdx.x"))
    sch[Out].bind(sch[Out].op.axis[1], tvm.te.thread_axis("threadIdx.x"))

    measure_opt = saber.MeasureOptions(
        target="cuda", build_func="default", target_host="llvm", timeout=10,
        verbose=1, number=100, repeat=1, min_repeat_ms=500,
        cooldown_interval=1, enable_cpu_cache_flush=1
    )

    cost = saber.measure_base.evaluate_schedule(
        sch, [A, W, Out], [L, K, CI, CO],
        [tvm.te.placeholder([CI_value, L_value]),
         tvm.te.placeholder([CO_value, CI_value, K_value]),
         tvm.te.placeholder([CO_value, P_value])],
        [L_value, K_value, CI_value, CO_value],
        measure_opt, new_process=True)

    print(tvm.lower(sch, [A, W, Out, L, K, CI, CO], simple_mode=True))
    print("GPU Cost is", cost, "ms")


if __name__ == "__main__":
    conv1d_cpu(128, 1, 256, 512)
    conv1d_gpu(128, 1, 256, 512)