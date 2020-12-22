import tvm
import os
from tvm import auto_tensorize as at
from itertools import product

"""In this tutorial, we fix recipe, hand-craft match points,
    and fix transform decisions, to see how parameters affects performance
"""


def conv3d(N, C, D, H, W, K, KD, R, S, stride, padding, dilation):
    pD = D + 2 * padding
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, D, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, KD, R, S], dtype="float16", name="B")

    Pad = tvm.te.compute(
        [N, C, pD, pH, pW],
        lambda n, c, d, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                d >= padding, d - padding < D,
                h >= padding, h - padding < H,
                w >= padding, w - padding < W),
            A[n, c, d - padding, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype)
        ),
        name="Pad")

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rd = tvm.te.reduce_axis([0, KD], name="rd")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    assert dilation == 1 and stride == 1
    outD = (pD - dilation * (KD - 1) - 1) // stride + 1
    P = (pH - dilation * (R - 1) - 1) // stride + 1
    Q = (pW - dilation * (S - 1) - 1) // stride + 1

    Conv = tvm.te.compute(
        [N, K, outD, P, Q],
        lambda n, k, d, p, q:
            tvm.te.sum((Pad[n, rc, d+rd, p*stride+rr, q*stride+rs] * B[k, rc, rd, rr, rs]
                        ).astype("float16"), axis=[rc, rd, rr, rs]),
        name="Conv"
    )
    # bias = tvm.te.placeholder([K], dtype="float32", name="bias")
    # E = tvm.te.compute(
    #     [N, K, P, Q],
    #     lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bk],
    #     name="E"
    # )
    return [A, B, Conv]


def tensorize_tensorcore_fp16fp16(
    N, C, D, H, W, K, KD, R, S, stride, padding, dilation, layer
):
    recipe = at.WMMAFp16Fp16()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, Conv = conv3d(N, C, D, H, W, K, KD, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])

    # hand-craft the match results
    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[1]
    }
    elem_op_map = {}
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, d, p, q = target_dag.op_lst[1].axis
    rc, rd, rr, rs = target_dag.op_lst[1].reduce_axis
    axis_map = {
        ii: [n, n, n, p, p, q, q],
        jj: [k, k, k, k, k, k, k],
        kk: [rc, rr, rs, rc, rs, rc, rr]
    }
    match_result = at.IntrinMatchResult(
        recipe, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    # fix transform decisions
    gen = at.TransformGenerator(match_result)
    record = gen.get(policy="random")
    record.unfold_choice = ([1, 1, 1, 1, 1, 1, 1], record.unfold_choice[1])
    app = at.TransformApplier(match_result)
    new_state = app.apply(record)

    log_file = "conv3d-layer-%d-batch-%d-%s-%s.log" % (
        layer, N, compute_key, shape_key)

    # prepare schedulers
    schedule_gen = at.CUDAScheduleGenerator(
        match_result, new_state, log_file=log_file)
    if os.path.exists(log_file) and os.path.isfile(log_file):
        schedule_gen.load_from_file(log_file)
    sc_info = schedule_gen.get_schedule_compute_info()
    schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
    trials = 400
    measure_opt = at.MeasureOptions(
        target=recipe.target, timeout=10, number=200, min_repeat_ms=500)
    checker = at.CUDAProgramChecker()

    # use tuning to find params
    value, params = at.find_optimized_parameters(
        match_result, schedule_gen, schedule_app,
        measure_opt, checker, trials,  # policy="random",
        builder=at.pebble_local_builder_build,
        runner=at.pebble_local_runner_run)

    # load from file
    schedule_gen.clear("")
    schedule_gen.load_from_file(log_file)
    entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in file
    params, value = entry.record, 1 / entry.value
    print(value)
    print(params.to_json())

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)


def run(N, C, D, H, W, K, KD, R, S, stride, padding, dilation, layer):  
    tensorize_tensorcore_fp16fp16(
        N, C, D, H, W, K, KD, R, S, stride, padding, dilation, layer)


conv3d_shapes = [
   # C,  D,   H,   W,  K, KD, R, S, stride, padding, dilation,
    ( 1, 3, 448, 448, 64,  3, 7, 7,      1,       2,        1),
    (12, 4, 128, 128, 64,  3, 3, 3,      1,       0,        1)
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = 15
    for batch in batches:
        for i, shape in enumerate(conv3d_shapes[beg:beg+num]):
            (C, D, H, W, K, KD, R, S, stride, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print("N, C, D, H, W, K, KD, R, S, stride, padding, dilation")
            print(N, C, D, H, W, K, KD, R, S, stride, padding, dilation)
            try:
                run(
                    N, C, D, H, W, K, KD, R, S, stride, padding, dilation, i + beg + 1
                )
            except Exception as e:
                print("Fail to run\n", str(e))
