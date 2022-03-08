import tvm
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


@register_test
def test1():
    print("##################################")
    print("Test 1")
    split_generator = at.SplitFactorGenerator(1024, 4)
    ret, d = split_generator.get()
    print("init:", ret)
    for i in range(20):
        ret, d = split_generator.get(hint=ret)
        print(ret, d)
    print("final:", ret)


@register_test
def test2():
    print("##################################")
    print("Test 2")
    split_generator = at.SplitFactorGenerator(1024, 4)
    ret, d = split_generator.get()
    print("init:", ret)
    for i in range(20):
        ret, d = split_generator.get(hint=ret, policy="q")
        print(ret, d)
    print("final:", ret)


@register_test
def test3():
    print("##################################")
    print("Test 3")
    generator = at.VectorizeLengthGenerator("cuda", "bfloat16")
    print(generator.lengths)
    ret, d = generator.get()
    print("init:", ret)
    for i in range(20):
        ret, d = generator.get(hint=ret, policy="q")
        print(ret, d)
    print("final:", ret)


@register_test
def test4():
    print("##################################")
    print("Test 4")
    N, C, H, W, K, R, S, stride, padding, dilation = 1, 256, 56, 56, 512, 3, 3, 1, 1, 1
    hw_abs_dag = at.WMMAFp16Fp16()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
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
        hw_abs_dag, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    # fix transform decisions
    gen = at.MappingGenerator(match_result)
    record = gen.get(policy="random")
    record.vmap_choice = ([1, 1, 1, 1, 1, 1, 1], record.vmap_choice[1])
    app = at.MappingApplier(match_result)
    new_state = app.apply(record)

    # prepare schedulers
    schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)

    # hand-craft params
    for i in range(100):
        params = schedule_gen.get_next()
        print("Random params:")
        print(params.to_json())
        schedule_gen.feedback(params, np.random.random())
        print(schedule_gen.score_table)
 

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
