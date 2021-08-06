import tvm
import math
import numpy as np
from tvm import testing
from tvm import auto_tensorize as at
from collections import OrderedDict
import itertools


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


def conv2d(N, C, H, W, K, R, S, stride, padding, with_bias=True, in_dtype=["float16", "float16"], out_dtype="float32"):
    H = H + 2 * padding
    W = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype[0], name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype=in_dtype[1], name="B")
    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (H - R) // stride + 1
    Q = (W - S) // stride + 1
    if in_dtype[0] == "uint8":
        Conv = tvm.te.compute(
            [N, K, P, Q],
            lambda n, k, p, q:
                tvm.te.sum((A[n, rc, p+rr, q+rs].astype(out_dtype) * B[k, rc, rr, rs].astype(out_dtype)
                            ), axis=[rc, rr, rs]),
            name="Conv"
        )
    else:
        Conv = tvm.te.compute(
            [N, K, P, Q],
            lambda n, k, p, q:
                tvm.te.sum((A[n, rc, p+rr, q+rs] * B[k, rc, rr, rs]
                            ).astype(out_dtype), axis=[rc, rr, rs]),
            name="Conv"
        )
    if not with_bias:
        return [A, B, Conv]
    bias = tvm.te.placeholder([N, K, P, Q], dtype=out_dtype, name="bias")
    E = tvm.te.compute(
        [N, K, P, Q],
        lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bn, bk, bp, bq],
        name="E"
    )
    return [A, B, bias, E]


def get_np_arrays(tensors):
    ret = []
    for t in tensors:
        np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(t.dtype)
        ret.append(np_ary)
    return ret


def get_tvm_arrays_from_np_arrays(arys, ctx):
    ret = []
    for ary in arys:
        tvm_ary = tvm.nd.array(ary, ctx)
        ret.append(tvm_ary)
    return ret


def get_tvm_arrays(tensors, ctx):
    ret = []
    for t in tensors:
        np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(t.dtype)
        tvm_ary = tvm.nd.array(np_ary, ctx)
        ret.append(tvm_ary)
    return ret


def get_factor_lst(value):
    assert isinstance(value, int)
    ret = []
    end = math.sqrt(value)
    for i in range(1, math.ceil(end)):
        if value % i == 0:
            ret.append(i)
            ret.append(value // i)
    if end - int(end) < 1e-10 and value % int(end) == 0:
        ret.append(int(end))

    return ret


def powerx_lst(x, left, right):
    ret = []
    beg = 1
    while beg < left:
        beg *= x
    while beg < right:
        ret.append(beg)
        beg = beg * x
    return ret


def any_factor_split(value, number, allow_non_divisible="off"):
    assert allow_non_divisible in ["off", "power2", "continuous"]
    ret = []
    assert isinstance(number, int)
    recursive_factor_split(value, [], number, ret, allow_non_divisible)
    return ret


def recursive_factor_split(left, cur, number, ret, policy):
    if number == 1:
        ret.append(cur + [left])
        return
    if policy == "power2":
        f_lst = get_factor_lst(left)
        f_lst.extend(powerx_lst(2, 1, left))
        f_lst = list(set(f_lst))
    elif policy == "continuous":
        f_lst = list(range(1, left + 1))
    else:
        f_lst = get_factor_lst(left)
        f_lst = sorted(f_lst)
    for f in f_lst:
        recursive_factor_split(left // f, cur + [f], number - 1, ret, policy)


def list_to_string(lst):
    return ":".join([str(x) for x in lst])


@register_test
def test1():
    print("##########################")
    print("Test 1")
    recipe = at.WMMAFp16Fp16()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    # c1
    A, B, E = conv2d(16, 64, 56, 56, 64, 3, 3, 1, 1, with_bias=False, out_dtype="float16")
    # c5
    # A, B, E = conv2d(16, 128, 28, 28, 128, 3, 3, 2, 1, with_bias=False, out_dtype="float16")
    # c9
    # A, B, E = conv2d(16, 256, 7, 7, 512, 3, 3, 2, 1, with_bias=False, out_dtype="float16")
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_np_arrays = get_np_arrays(inputs_ref)
    inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[0].axis
    rc, rr, rs = target_dag.op_lst[0].reduce_axis
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

    gen = at.TransformGenerator(match_result)

    possible_mappings = list(itertools.product([0, 1], repeat=7))

    # possible_mappings = [(0, 1, 0, 0, 0, 1, 0)]
    
    fout = open("mappings-layer-c1-3.log", "w")
    print("mapping,execution time", file=fout, flush=True)
    # ignore first [0, 0, ..., 0]
    list1 = any_factor_split(56, 2)
    list2 = any_factor_split(4, 3)
    list3 = any_factor_split(28, 2)
    # for f1 in [[-1, 8]]:
    #     print(f1)
    #     for f2 in [[1, 1, 4]]:
    #         print(f2)
    #         for f3 in [[-1, 4]]:
    #             print(f3)
    for mapping in possible_mappings[1:]:
        record = gen.get()
        record.unfold_choice = (mapping, record.unfold_choice[1])
        # print(record.to_json())
        app = at.TransformApplier(match_result)
        new_state = app.apply(record)

        schedule_gen = at.CUDAScheduleGeneratorV2(match_result, new_state)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.CUDAScheduleApplierV2(match_result, sc_info)
        params = schedule_gen.get()
        # my_params = {
        #     'vectorize': (1, 1),
        #     'spatial_factors': [([1, 1, 1], (0, 0)), ([1, 1, 1], (0, 0)), ([4, 1, 1], (-1, 1)), ([14, 1, 1], (-1, -1))],
        #     'reduce_factors': [([1, 1, 1], (0, 0)), ([3, 3, 4], (1, 1)), ([1, 1, 3], (0, -1))],
        #     'last_factors': [([-1, 32], (-1,))],
        #     'output_unroll_step': (64, -1),
        #     'last_unroll_step': (512, 1)}
        # my_params = {
        #     'inline': [1, 1], 
        #     'vectorize': [4, 1], 
        #     'spatial_factors': [[[7, 8, 1, 1], [-1, 1, 0]], [[1, 1, 4, 1], [0, 0, 0]], [[7, 4, 2, 1], [0, 1, -1]]], 
        #     'reduce_factors': [[[3, 2, 2], [1, 0]], [[3, 1, 1], [0, 0]]], 
        #     'last_factors': [[[3584, 2, 14], [-1, 0]]], 
        #     'output_unroll_step': [512, -1], 
        #     'last_unroll_step': [64, 1]}
        # my_params = {
        #     'inline': [1, 1], 
        #     'vectorize': [4, 1], 
        #     'spatial_factors': [[[*f1, 1, 1], [-1, 1, 0]], [[*f2, 1], [0, 0, 0]], [[*f3, 2, 1], [0, 1, -1]]], 
        #     'reduce_factors': [[[3, 2, 2], [1, 0]], [[3, 1, 1], [0, 0]]], 
        #     'last_factors': [[[3584, 2, 14], [-1, 0]]], 
        #     'output_unroll_step': [512, -1], 
        #     'last_unroll_step': [64, 1]}
        # my_params = {
        #     'inline': [0, -1], 
        #     'vectorize': [2, -1], 
        #     'spatial_factors': [[[49, 1, 1, 1], [0, 0, 0]], [[1, 2, 2, 2], [-1, -1, 1]], [[2, 1, 2, 4], [-1, -1, 0]]], 
        #     'reduce_factors': [[[4, 2, 1], [0, 1]], [[3, 1, 1], [0, 0]], [[1, 1, 3], [0, -1]]], 
        #     'last_factors': [[[112, 56, 8], [-1, 1]]], 
        #     'output_unroll_step': [64, -1], 
        #     'last_unroll_step': [512, 1]}
        my_params = {
            'inline': [0, -1], 
            'vectorize': [8, -1], 
            'spatial_factors': [[[7, 7, 1, 1], [1, 0, -1]], [[32, 1, 1, 1], [1, 0, 0]]], 
            'reduce_factors': [[[6, 6, 4], [0, 0]]], 
            'last_factors': [[[3136, 1, 4], [0, 0]]], 
            'output_unroll_step': [512, -1], 
            'last_unroll_step': [1500, 1]}
        params.from_json(my_params)
        print(params.to_json())
        
        new_target_dag = sc_info.target_dag
        new_inputs = new_target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])

        try:
            schedule_app.apply(sch, params)
            measure_opt = at.MeasureOptions(
                target="cuda", timeout=10, number=200, min_repeat_ms=500)
            cost = at.evaluate_schedule(sch, new_inputs + list(new_target_dag.tensors), measure_opt, new_process=True)
            
        except Exception as e:
            print(e)
            cost = 1e10
            continue
        
        print(cost)
        print(f"{list_to_string(mapping)},{cost}", file=fout, flush=True)
    print("Pass!\n")



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