import tvm
import os
import tvm.te as te
import tvm.auto_tensorize as at
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


N = 4
C = 1024
P = 14
Q = 14
K = 512
R = 3
S = 3
H = P + R // 2 * 2
W = Q + S // 2 * 2

input_dtype = "float16"
output_dtype = "float16"


def gemm_intrinsic_compute():
    A = te.placeholder([32, 32], dtype=input_dtype, name="AA")
    B = te.placeholder([32, 32], dtype=input_dtype, name="BB")
    k = te.reduce_axis([0, 32], name="kk")
    Out = te.compute(
        [32, 32], lambda i, j: te.sum((A[i, k] * B[k, j]).astype(output_dtype), axis=[k]), name="OO"
    )
    return Out


@register_test
def test1():
    A = te.placeholder([N, H, W, C], dtype=input_dtype, name="A")
    Weight = te.placeholder([R, S, C, K], dtype=input_dtype, name="W")
    rc = te.reduce_axis([0, C], name="rc")
    rr = te.reduce_axis([0, R], name="rr")
    rs = te.reduce_axis([0, S], name="rs")
    Out = te.compute(
        [N, P, Q, K],
        lambda b, p, q, k: te.sum(
            (A[b, p + rr, q + rs, rc] * Weight[rr, rs, rc, k]).astype(output_dtype),
            axis=[rc, rr, rs],
        ),
        name="Out",
    )

    intrin_t = gemm_intrinsic_compute()

    print("Target compute:")
    print(Out.op.body[0])

    print("Intrin compute:")
    print(intrin_t.op.body[0])

    # recipe = at.WMMAFp16Fp32()
    # main_capsule = recipe.get_capsule_compute_expression(
    #   'nnn', '16x16x16', recipe.main_capsule_name)

    print("Intrinsic match:")
    # print(at.intrinsic_match(Out, intrin_t, main_capsule[1][0].op))
    print(at.intrinsic_match(Out, intrin_t, intrin_t.op))


@register_test
def test2():
    A = te.placeholder([H, C], dtype=input_dtype)
    Weight = te.placeholder([C, W], dtype=input_dtype)
    rc = te.reduce_axis([0, C], name="rc")
    Out = te.compute(
        [H, W],
        lambda i, j: te.sum((A[i, rc] * Weight[rc, j]).astype(output_dtype), axis=[rc]),
        name="Out",
    )

    intrin_t = gemm_intrinsic_compute()

    print("Target compute:")
    print(Out.op.body[0])

    print("Intrin compute:")
    print(intrin_t.op.body[0])

    # recipe = at.WMMAFp16Fp32()
    # main_capsule = recipe.get_capsule_compute_expression(
    #   'nnn', '16x16x16', recipe.main_capsule_name)

    print("Intrinsic match:")
    # print(at.intrinsic_match(Out, intrin_t, main_capsule[1][0].op))
    print(at.intrinsic_match(Out, intrin_t, intrin_t.op))

    # {
    #   compute(OO, 0x56534d1f42f0): [{
    #     iter_var(i, range(min=0, ext=16)): iter_var(i, range(min=0, ext=32)),
    #     iter_var(j, range(min=0, ext=16)): iter_var(j, range(min=0, ext=32)),
    #     iter_var(rc, range(min=0, ext=1024)): iter_var(kk, range(min=0, ext=32))
    #   }]
    # }


@register_test
def test3():
    A = te.placeholder([N, H, W, C], dtype=input_dtype, name="A")
    Weight = te.placeholder([R, S, C, K], dtype=input_dtype, name="W")
    rc = te.reduce_axis([0, C], name="rc")
    rr = te.reduce_axis([0, R], name="rr")
    rs = te.reduce_axis([0, S], name="rs")
    Out = te.compute(
        [N, P, Q, K],
        lambda b, p, q, k: te.sum(
            (A[b, p + rr, q + rs, rc] * Weight[rr, rs, rc, k]).astype(output_dtype),
            axis=[rc, rr, rs],
        ),
        name="Out",
    )

    print("Target compute:")
    print(Out.op.body[0])

    recipe = at.WMMAFp16Fp16()
    main_capsule = recipe.get_capsule_compute_expression(
        "nnn", "16x16x16", recipe.main_capsule_name
    )
    intrin_Out = main_capsule[1][0]

    print("Intrin compute:")
    print(intrin_Out.op.body[0])

    print("Intrinsic match:")
    # print(at.intrinsic_match(Out, intrin_t, intrin_t.op))
    print(at.intrinsic_match(Out, intrin_Out, intrin_Out.op))


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
            tvm.te.sum((Pad[n, rc, p*stride+rr, q*stride+rs] * B[k, rc, rr, rs]
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


def get_match_result(target_dag, recipe, compute_key, shape_key):
    intrin_dag, main_tensors = recipe.get_effective_compute_dag(compute_key, shape_key)
    target_tensors = list(target_dag.tensors)
    intrin_tensors = list(intrin_dag.tensors)
    # TODO: (yicheng) remove such constraints, do a general DAG match
    assert len(target_tensors) == 1
    assert len(intrin_tensors) == 1
    assert len(main_tensors) == 1
    main_op = main_tensors[0].op
    print(main_op)
    print(target_tensors[0], target_tensors[0].op)
    print(intrin_tensors[0], intrin_tensors[0].op)
    raw_match = at.intrinsic_match(
        target_tensors[0],
        intrin_tensors[0],
        main_op)

    match_results = []
    for top, match_points in raw_match.items():
        main_op_map = {
            main_op: top
        }
        # TODO: (size): elem mapping seems not necessary
        elem_op_map = {}
        intrin_axis = main_op.axis
        intrin_reduce_axis = main_op.reduce_axis
        axis_map = {
            iiv : [] for iiv in list(intrin_axis) + list(intrin_reduce_axis)
        }
        for point in match_points:
            for tiv, iiv in point.items():
                axis_map[iiv].append(tiv)
        match_result = at.IntrinMatchResult(
            recipe, compute_key, shape_key,
            main_op_map, elem_op_map,
            axis_map, target_dag, intrin_dag
        )
        match_results.append(match_result)
    return match_results


@register_test
def test4(
):
    (N, C, H, W, K, R, S, stride,
    padding, dilation, layer) = (
        1, 1024, 14, 14, 1024, 3, 3, 1, 1, 1, 15
    )
    recipe = at.WMMAFp16Fp16()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, main_outputs = recipe.get_effective_compute_dag(compute_key, shape_key)

    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])

    # get the match result
    match_results = get_match_result(target_dag, recipe, compute_key, shape_key)

    print(match_results)
    
    match_result = match_results[0]
    # # hand-craft the match results
    # main_op_map = {
    #     intrin_dag.op_lst[0]: target_dag.op_lst[1]
    # }
    # elem_op_map = {}
    # ii, jj = intrin_dag.op_lst[0].axis
    # kk, = intrin_dag.op_lst[0].reduce_axis
    # n, k, p, q = target_dag.op_lst[1].axis
    # rc, rr, rs = target_dag.op_lst[1].reduce_axis
    # axis_map = {
    #     ii: [n, n, n, p, p, q, q],
    #     jj: [k, k, k, k, k, k, k],
    #     kk: [rc, rr, rs, rc, rs, rc, rr]
    # }
    # match_result = at.IntrinMatchResult(
    #     recipe, compute_key, shape_key,
    #     main_op_map, elem_op_map,
    #     axis_map, target_dag, intrin_dag
    # )

    # fix transform decisions
    gen = at.TransformGenerator(match_result)
    record = gen.get(policy="random")
    record.unfold_choice = ([1, 1, 1, 1, 1, 1, 1], record.unfold_choice[1])
    app = at.TransformApplier(match_result)
    new_state = app.apply(record)

    log_file = "Yolo-layer-%d-batch-%d-%s-%s.log" % (
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
        target=recipe.target, timeout=20, number=200, min_repeat_ms=500)
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


@register_test
def test5(
):
    (N, C, H, W, K, R, S, stride,
    padding, dilation, layer) = (
        1, 1024, 14, 14, 1024, 3, 3, 1, 1, 1, 15
    )

    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    # get the match result
    match_results = at.get_match_results(target_dag, target)

    for r in match_results:
        print(r.recipe, r.compute_key, r.shape_key)
    
    match_result = match_results[0]

    # fix transform decisions
    gen = at.TransformGenerator(match_result)
    record = gen.get(policy="random")
    record.unfold_choice = ([1, 1, 1, 1, 1, 1, 1], record.unfold_choice[1])
    app = at.TransformApplier(match_result)
    new_state = app.apply(record)

    log_file = "Yolo-layer-%d-batch-%d-%s-%s.log" % (
        layer, N, match_result.compute_key, match_result.shape_key)

    # prepare schedulers
    schedule_gen = at.CUDAScheduleGenerator(
        match_result, new_state, log_file=log_file)
    if os.path.exists(log_file) and os.path.isfile(log_file):
        schedule_gen.load_from_file(log_file)
    sc_info = schedule_gen.get_schedule_compute_info()
    schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
    trials = 400
    measure_opt = at.MeasureOptions(
        target=match_result.recipe.target, timeout=20, number=200, min_repeat_ms=500)
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


class AutoTensorizeResult(object):
    def __init__(self, sch_gen, sch_app, params, perf):
        self.sch_gen = sch_gen
        self.sch_app = sch_app
        self.params = params
        self.perf = perf

    def defined(self):
        return ((self.sch_gen is not None)
                and (self.sch_app is not None)
                and (self.params is not None)
                and (self.perf is not None))


def auto_tensorize(target_dag, target,
        log_file,
        measure_opt,
        trials=200,
        builder=at.pebble_local_builder_build,
        runner=at.pebble_local_runner_run,
        verbose=False):
    # refactor target
    measure_opt.target = target
    match_results = at.get_match_results(target_dag, target)
    for r in match_results:
        print(r.recipe, r.compute_key, r.shape_key)
    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target" % target, flush=True)
        return AutoTensorizeResult(None, None, None, None)
    # here is match intrin policy
    match_result = match_results[0]
    print(match_result.axis_map)

    gen = at.TransformGenerator(match_result)
    record = gen.get(policy="random")
    # here is transform policy
    record.unfold_choice = (
        [1 for _ in record.unfold_choice[0]], record.unfold_choice[1])
    app = at.TransformApplier(match_result)
    new_state = app.apply(record)

    if str(target) == "cuda":
        schedule_gen = at.CUDAScheduleGenerator(
            match_result, new_state, log_file=log_file)
        if os.path.exists(log_file) and os.path.isfile(log_file):
            schedule_gen.load_from_file(log_file)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
        checker = at.CUDAProgramChecker()
    else:
        raise RuntimeError("Do not support target: %s" % target)
    
    # use tuning to find params
    value, params = at.find_optimized_parameters(
        match_result, schedule_gen, schedule_app,
        measure_opt, checker, trials,  # policy="random",
        builder=builder,
        runner=runner,
        verbose=verbose)

    return AutoTensorizeResult(
        schedule_gen,
        schedule_app,
        params,
        value
    )


def get_schedule(sch_app, params):
    target_dag = sch_app.target_dag
    inputs = target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])

    args = inputs + list(target_dag.tensors)
    sch = sch_app.apply(sch, params)
    return sch, args


@register_test
def test6(
):
    (N, C, H, W, K, R, S, stride,
    padding, dilation, layer) = (
        1, 3, 14, 14, 64, 7, 7, 2, 3, 1, 15
    )

    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "Yolo-layer-%d-batch-%d.log" % (
        layer, N)

    trials = 10
    measure_opt = at.MeasureOptions(
        target=target, timeout=20, number=200, min_repeat_ms=500)

    result = auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, verbose=False)
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app

    # load from file
    schedule_gen.load_from_file(log_file, clear=True)
    entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in file
    params, value = entry.record, 1 / entry.value
    print(value)
    print(params.to_json())

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)

    sch, args = get_schedule(schedule_app, params)
    # print(tvm.lower(sch, args, simple_mode=True))


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
