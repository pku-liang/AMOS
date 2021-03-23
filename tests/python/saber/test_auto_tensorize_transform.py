import tvm
import os
import numpy as np
from tvm import auto_tensorize as at
from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[:len(prefix)] == prefix
    try:
        number = int(name[len(prefix):])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)
        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix):])


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="bfloat16", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="bfloat16", name="B")

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
            tvm.te.sum((Pad[n, rc, p*stride+rr, q*stride+rs] * B[k, rc, rr, rs]
                        ).astype("float32"), axis=[rc, rr, rs]),
        name="Conv"
    )
    # bias = tvm.te.placeholder([K], dtype="float32", name="bias")
    # E = tvm.te.compute(
    #     [N, K, P, Q],
    #     lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bk],
    #     name="E"
    # )
    return [A, B, Conv]


@register_test
def test1():
    recipe = at.WMMABf16Fp32()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)

    # N = tvm.tir.Var("N", "int32")
    N = 128
    K = tvm.tir.Var("K", "int32")
    H = tvm.tir.Var("H", "int32")
    W = tvm.tir.Var("W", "int32")
    C = tvm.tir.Var("C", "int32")
    R = S = 3
    # stride = tvm.tir.Var("stride", "int32")
    # padding = tvm.tir.Var("padding", "int32")
    # dilation = tvm.tir.Var("dilation", "int32")
    stride = 1
    padding = 1
    dilation = 1

    Vars = [K, H, W, C]

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

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    print(new_inputs)
    print(new_target_dag.tensors)
    A, B = new_inputs
    C, = new_target_dag.tensors
    
    # print(tvm.lower(
    #     sch, new_inputs + list(new_target_dag.tensors) + Vars, simple_mode=True))
    # func = tvm.build(sch, new_inputs + list(new_target_dag.tensors) + Vars, "cuda")

    log_file = "Yolo-layer-%d-batch-%d-%s-%s.log" % (
        0, N, compute_key, shape_key)

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
        runner=at.pebble_local_runner_run, verbose=True)

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="test case", type=int, default=1)
    parser.add_argument("--all", help="test all", action="store_true")

    args = parser.parse_args()
    if args.all:
        for k, v in TEST_CASES.items():
            print("############################################")
            print("test", k)
            v()
            print("Pass!")
    else:
        assert args.case in TEST_CASES, "Can't find case %s." % (
            str(args.case))
        case = TEST_CASES[args.case]
        case()
