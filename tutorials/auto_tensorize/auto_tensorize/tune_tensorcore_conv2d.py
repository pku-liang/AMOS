import tvm
import os
from tvm import auto_tensorize as at

"""In this tutorial, we fix recipe, hand-craft match points,
    and fix transform decisions, to see how parameters affects performance
"""


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float16", name="B")

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

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (pH - R) // stride + 1
    Q = (pW - S) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
            tvm.te.sum((Pad[n, rc, p+rr, q+rs] * B[k, rc, rr, rs]
                        ).astype("float16"), axis=[rc, rr, rs]),
        name="Conv"
    )
    # bias = tvm.te.placeholder([K], dtype="float32", name="bias")
    # E = tvm.te.compute(
    #     [N, K, P, Q],
    #     lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bk],
    #     name="E"
    # )
    return [A, B, Conv]


def tensorize_tensorcore_fp16fp16_nnn_16x16x16(
    N, C, H, W, K, R, S, stride, padding, dilation
):
    recipe = at.WMMAFp16Fp16()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])

    # hand-craft the match results
    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[1]
    }
    elem_op_map = {}
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[1].axis
    rc, rr, rs = target_dag.op_lst[1].reduce_axis
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

    # prepare schedulers
    schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)
    sc_info = schedule_gen.get_schedule_compute_info()
    schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
    trials = 2000
    measure_opt = at.MeasureOptions(target=recipe.target, timeout=20, number=200, min_repeat_ms=500)
    checker = at.CUDAProgramChecker()

    # use tuning to find params
    # value, params = at.find_optimized_parameters(
    #     match_result, schedule_gen, schedule_app,
    #     measure_opt, checker, trials, policy="random",
    #     builder=at.pebble_local_builder_build,
    #     runner=at.pebble_local_runner_run)
    # print(value)
    # print(params.to_json())

    # hand-craft params
    params = schedule_gen.get()
    print("Random params:")
    print(params.to_json())
    my_params = {'vectorize': (4, -1), 'spatial_factors': [([49, 1, 2, 2], (0, 1, 1)), ([2, 1, 8, 2], (1, 0, -1))], 'reduce_factors': [([36, 4, 1], (-1, -1))], 'last_factors': [([512, 7, 32], (-1, 0))], 'output_unroll_step': (64, 1), 'last_unroll_step': (512, 0)}
    params.from_json(my_params)
 
    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)


def tensorize_tensorcore_fp16fp16_ntn_16x16x16(
    N, C, H, W, K, R, S, stride, padding, dilation, layer
):
    recipe = at.WMMAFp16Fp16()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])

    # hand-craft the match results
    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[1]
    }
    elem_op_map = {}
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[1].axis
    rc, rr, rs = target_dag.op_lst[1].reduce_axis
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

    log_file = "Yolo-layer-%d.log" % layer

    # prepare schedulers
    schedule_gen = at.CUDAScheduleGenerator(match_result, new_state, log_file=log_file)
    if os.path.exists(log_file) and os.path.isfile(log_file):
        schedule_gen.load_from_file(log_file)
    sc_info = schedule_gen.get_schedule_compute_info()
    schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
    trials = 2000
    measure_opt = at.MeasureOptions(
        target=recipe.target, timeout=20, number=200, min_repeat_ms=500)
    checker = at.CUDAProgramChecker()

    # use tuning to find params
    value, params = at.find_optimized_parameters(
        match_result, schedule_gen, schedule_app,
        measure_opt, checker, trials, # policy="random",
        builder=at.pebble_local_builder_build,
        runner=at.pebble_local_runner_run)

    # load from file
    schedule_gen.clear("")
    schedule_gen.load_from_file(log_file)
    entry = schedule_gen.get_best_entry()
    params, value = entry.record, 1 / entry.value  # we store 1/time_cost in file
    print(value)
    print(params.to_json())

    # hand-craft params
    # params = schedule_gen.get()
    # my_params = {'vectorize': (2, 1), 'spatial_factors': [([49, 1, 2, 2], (0, 0, 0)), ([2, 1, 8, 2], (0, -1, 1))], 'reduce_factors': [([36, 2, 2], (1, 0))], 'last_factors': [([28, 448, 4], (-1, 1))], 'output_unroll_step': (64, 1), 'last_unroll_step': (64, 1)}
    # params.from_json(my_params)
    # print(params.to_json())
 
    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)


def run(N, C, H, W, K, R, S, stride, padding, dilation, layer):
    tensorize_tensorcore_fp16fp16_ntn_16x16x16(
        N, C, H, W, K, R, S, stride, padding, dilation, layer)


yolo_shapes_b1 = [
    # yolo
    (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    (1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1),  # conv5   4
    (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv7   6
    (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv9
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv10
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv11
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv12
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv13
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv14
    (1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1),  # conv15      8
    (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv17    10
    (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    # # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv19
    # # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv20
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
    # (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv24
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = 1
    for batch in batches:
        for i, shape in enumerate(yolo_shapes_b1[beg:beg+num]):
            _, C, H, W, K, _, R, S, _, stride, padding, dilation, _ = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            run(
                N, C, H, W, K, R, S, stride, padding, dilation, i + beg + 1
            )