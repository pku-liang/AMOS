from math import fabs
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
    P_factors = [-1, 4]
    CI_factors = [-1, 4]
    CO_factors = [-1, 4]
    L = tvm.tir.Var("L", "int32")
    K = tvm.tir.Var("K", "int32")
    CI = tvm.tir.Var("CI", "int32")
    CO = tvm.tir.Var("CO", "int32")
    PO = tvm.tir.Var("PO", "int32")
    CIO = tvm.tir.Var("CIO", "int32")
    COO = tvm.tir.Var("COO", "int32")
    A = tvm.te.placeholder([CI, L], name="A")
    W = tvm.te.placeholder([CO, CI, K], name="W")
    # AA = tvm.te.compute([CI, L], lambda i, j: A[i, j], name="AA")
    # WW = tvm.te.compute([CO, CI, K], lambda i, j, k: W[i, j, k], name="WW")
    AA = tvm.te.compute(
        [CIO, CI_factors[1], PO, P_factors[1], K],
        lambda cio, cii, po, pi, k:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    cio * CI_factors[1] + cii < CI,
                    po * P_factors[1] + pi + k < L
                ),
                A[cio * CI_factors[1] + cii, po * P_factors[1] + pi + k],
                0.0
            ),
            name="AA"
    )
    WW = tvm.te.compute(
        [COO, CO_factors[1], CIO, CI_factors[1], K],
        lambda coo, coi, cio, cii, k:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    cio * CI_factors[1] + cii < CI,
                    coo * CO_factors[1] + coi < CO
                ),
                W[coo * CO_factors[1] + coi, cio * CI_factors[1] + cii, k],
                0.0
            ),
            name="AA"
    )
    P = (L - K) + 1
    P_value = (L_value - K_value) + 1
    PO_value = (P_value + P_factors[1] - 1) // P_factors[1]
    CIO_value = (CI_value + CI_factors[1] - 1) // CI_factors[1]
    COO_value = (CO_value + CO_factors[1] - 1) // CO_factors[1]
    rco = tvm.te.reduce_axis([0, CIO], name="rc")
    rci = tvm.te.reduce_axis([0, CI_factors[1]], name="rc")
    rk = tvm.te.reduce_axis([0, K], name="rk")
    OO = tvm.te.compute(
        [COO, CO_factors[1], PO, P_factors[1]],
        lambda coo, coi, po, pi:
            tvm.te.sum(
                AA[rco, rci, po, pi, rk] * WW[coo, coi, rco, rci, rk],
                axis=[rco, rci, rk]
            ),
        name="OO"
    )
    Out = tvm.te.compute([CO, P], lambda i, j:
        OO[i//CO_factors[1], i%CO_factors[1], j//P_factors[1], j%P_factors[1]], name="Out")


    sch = tvm.te.create_schedule(Out.op)
    co, ci, po, pi = sch[OO].op.axis
    rco, rci, rk = sch[OO].op.reduce_axis
    sch[OO].reorder(co, po, rco, rk, ci, pi, rci)

    sch[AA].compute_at(sch[OO], rk)
    sch[WW].compute_at(sch[OO], rk)

    measure_opt = saber.MeasureOptions(
        target="c", build_func="default", target_host="llvm", timeout=10,
        verbose=1, number=100, repeat=1, min_repeat_ms=500,
        cooldown_interval=1, enable_cpu_cache_flush=1
    )

    cost = saber.measure_base.evaluate_schedule(
        sch, [A, W, Out], [L, K, CI, CO, PO, CIO, COO],
        [tvm.te.placeholder([CI_value, L_value]),
         tvm.te.placeholder([CO_value, CI_value, K_value]),
         tvm.te.placeholder([CO_value, P_value])],
        [L_value, K_value, CI_value, CO_value, PO_value, CIO_value, COO_value],
        measure_opt, new_process=True)

    print(tvm.lower(sch, [A, W, Out, L, K, CI, CO, PO, CIO, COO], simple_mode=True))
    print("CPU Cost is", cost, "ms")


def conv1d_gpu(L_value, K_value, CI_value, CO_value):
    P_factors = [-1, 4]
    CI_factors = [-1, 4]
    CO_factors = [-1, 4]
    L = tvm.tir.Var("L", "int32")
    K = tvm.tir.Var("K", "int32")
    CI = tvm.tir.Var("CI", "int32")
    CO = tvm.tir.Var("CO", "int32")
    PO = tvm.tir.Var("PO", "int32")
    CIO = tvm.tir.Var("CIO", "int32")
    COO = tvm.tir.Var("COO", "int32")
    A = tvm.te.placeholder([CI, L], name="A")
    W = tvm.te.placeholder([CO, CI, K], name="W")
    # AA = tvm.te.compute([CI, L], lambda i, j: A[i, j], name="AA")
    # WW = tvm.te.compute([CO, CI, K], lambda i, j, k: W[i, j, k], name="WW")
    AA = tvm.te.compute(
        [CIO, CI_factors[1], PO, P_factors[1], K],
        lambda cio, cii, po, pi, k:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    cio * CI_factors[1] + cii < CI,
                    po * P_factors[1] + pi + k < L
                ),
                A[cio * CI_factors[1] + cii, po * P_factors[1] + pi + k],
                0.0
            ),
            name="AA"
    )
    WW = tvm.te.compute(
        [COO, CO_factors[1], CIO, CI_factors[1], K],
        lambda coo, coi, cio, cii, k:
            tvm.tir.if_then_else(
                tvm.tir.all(
                    cio * CI_factors[1] + cii < CI,
                    coo * CO_factors[1] + coi < CO
                ),
                W[coo * CO_factors[1] + coi, cio * CI_factors[1] + cii, k],
                0.0
            ),
            name="AA"
    )
    P = (L - K) + 1
    P_value = (L_value - K_value) + 1
    PO_value = (P_value + P_factors[1] - 1) // P_factors[1]
    CIO_value = (CI_value + CI_factors[1] - 1) // CI_factors[1]
    COO_value = (CO_value + CO_factors[1] - 1) // CO_factors[1]
    rco = tvm.te.reduce_axis([0, CIO], name="rc")
    rci = tvm.te.reduce_axis([0, CI_factors[1]], name="rc")
    rk = tvm.te.reduce_axis([0, K], name="rk")
    OO = tvm.te.compute(
        [COO, CO_factors[1], PO, P_factors[1]],
        lambda coo, coi, po, pi:
            tvm.te.sum(
                AA[rco, rci, po, pi, rk] * WW[coo, coi, rco, rci, rk],
                axis=[rco, rci, rk]
            ),
        name="OO"
    )
    Out = tvm.te.compute([CO, P], lambda i, j:
        OO[i//CO_factors[1], i%CO_factors[1], j//P_factors[1], j%P_factors[1]], name="Out")


    sch = tvm.te.create_schedule(Out.op)
    co, ci, po, pi = sch[OO].op.axis
    rco, rci, rk = sch[OO].op.reduce_axis
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
        sch, [A, W, Out], [L, K, CI, CO, PO, CIO, COO],
        [tvm.te.placeholder([CI_value, L_value]),
         tvm.te.placeholder([CO_value, CI_value, K_value]),
         tvm.te.placeholder([CO_value, P_value])],
        [L_value, K_value, CI_value, CO_value, PO_value, CIO_value, COO_value],
        measure_opt, new_process=True)

    print(tvm.lower(sch, [A, W, Out, L, K, CI, CO, PO, CIO, COO], simple_mode=True))
    print("GPU Cost is", cost, "ms")


if __name__ == "__main__":
    conv1d_cpu(128, 1, 256, 512)
    conv1d_gpu(128, 1, 256, 512)