import tvm
import os
from tvm import auto_tensorize as at
from itertools import product

"""In this tutorial, we fix hw_abs_dag, hand-craft match points,
    and fix transform decisions, to see how parameters affects performance
"""


def depthwise_conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    assert(K % C == 0)
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, R, S], dtype="float16", name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                h >= padding, h - padding < H,
                w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype)
        ),
        name="Pad")
    
    B_reshaped = tvm.te.compute(
        [C, (K//C), R, S],
        lambda k_o, k_i, r, s: B[k_o * (K//C) + k_i, r, s]
    )

    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (pH - R) // stride + 1
    Q = (pW - S) // stride + 1

    Conv = tvm.te.compute(
        [N, C, K//C, P, Q],
        lambda n, k_o, k_i, p, q:
            tvm.te.sum((Pad[n, k_o, p*stride+rr, q*stride+rs] * B_reshaped[k_o, k_i, rr, rs]
                        ).astype("float16"), axis=[rr, rs]),
        name="Conv"
    )

    Conv_reshaped = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
            Conv[n, k//(K//C), k%(K//C), p, q],
        name="Reshaped"
    )

    # bias = tvm.te.placeholder([K], dtype="float32", name="bias")
    # E = tvm.te.compute(
    #     [N, K, P, Q],
    #     lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bk],
    #     name="E"
    # )
    return [A, B, Conv_reshaped]


def tensorize_tensorcore_fp16fp16(
    N, C, H, W, K, R, S, stride,
    padding, dilation, layer
):
    hw_abs_dag = at.WMMAFp16Fp16()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    A, B, Conv = depthwise_conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])

    # hand-craft the match results
    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[2]
    }
    elem_op_map = {}
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k_o, k_i, p, q = target_dag.op_lst[2].axis
    rr, rs = target_dag.op_lst[2].reduce_axis
    
    axis_map = {
        ii: [n, n, p, q],
        jj: [k_i, k_i, k_i, k_i],
        kk: [rr, rs, rs, rr]
    }
    match_result = at.IntrinMatchResult(
        hw_abs_dag, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    # fix transform decisions
    gen = at.MappingGenerator(match_result)
    record = gen.get(policy="random")
    record.unfold_choice = ([1, 1, 1, 1], record.unfold_choice[1])
    app = at.MappingApplier(match_result)
    new_state = app.apply(record)

    log_file = "MobileNetV2-layer-%d-batch-%d-%s-%s.log" % (
        layer, N, compute_key, shape_key)

    # prepare schedulers
    schedule_gen = at.CUDAScheduleGenerator(
        match_result, new_state, log_file=log_file)
    if os.path.exists(log_file) and os.path.isfile(log_file):
        schedule_gen.load_from_file(log_file)
    sc_info = schedule_gen.get_schedule_compute_info()
    schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
    trials = 1000
    measure_opt = at.MeasureOptions(
        target=hw_abs_dag.target, timeout=10, number=200, min_repeat_ms=500)
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
    return cost


def run(N, C, H, W, K, R, S, stride,
        padding, dilation, layer):
    return tensorize_tensorcore_fp16fp16(
        N, C, H, W, K, R, S, stride,
        padding, dilation, layer)


 # (N, C, H, W, K, _, R, S, _, stride, padding, dilation, _)
mobilenet_v2_shapes = [
    (1, 32, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1, 32),
    (1, 16, 112, 112, 16 * 6, 16, 3, 3, 1, 2, 1, 1, 16),
    (1, 24, 56, 56, 24 * 6, 24, 3, 3, 1, 2, 1, 1, 24),
    (1, 32, 28, 28, 32 * 6, 32, 3, 3, 1, 2, 1, 1, 32),
    (1, 64, 14, 14, 64 * 6, 64, 3, 3, 1, 1, 1, 1, 64),
    (1, 96, 14, 14, 96 * 6, 96, 3, 3, 1, 2, 1, 1, 96),
    (1, 160, 7, 7, 160 * 6, 160, 3, 3, 1, 1, 1, 1, 160),
]

if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = 7
    for batch in batches:
        costs = []
        for i, shape in enumerate(mobilenet_v2_shapes):
            (_, C, H, W, K, _, R, S, _, stride,
                padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            try:
                cost = run(
                    N, C, H, W, K, R, S, stride,
                    padding, dilation,
                    i
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
