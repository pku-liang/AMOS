import tvm
import numpy as np
from tvm import testing
from tvm import auto_tensorize as at


def conv2d(N, C, H, W, K, R, S, stride, padding):
    H = H + 2 * padding
    W = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float16", name="B")
    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (H - R) // stride + 1
    Q = (W - S) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
            tvm.te.sum((A[n, rc, p+rr, q+rs] * B[k, rc, rr, rs]
                        ).astype("float32"), axis=[rc, rr, rs]),
        name="Conv"
    )
    bias = tvm.te.placeholder([N, K, P, Q], dtype="float32", name="bias")
    E = tvm.te.compute(
        [N, K, P, Q],
        lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bn, bk, bp, bq],
        name="E"
    )
    return [A, B, bias, E]


def get_tvm_arrays(tensors, ctx):
    ret = []
    for t in tensors:
        np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(t.dtype)
        tvm_ary = tvm.nd.array(np_ary, ctx)
        ret.append(tvm_ary)
    return ret


def test1():
    print("##########################")
    print("Test 1")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"

    def cond(cur):
        return (
            cur in recipe.capsules and
            (cur in recipe.capsules and
             issubclass(recipe.capsules[cur], at.ComputeCapsule)))
    op_list, read_graph, feed_graph = recipe.serialize_dag(
        cond1=cond
    )
    outputs = []
    for x in op_list:
        if x not in feed_graph:
            outputs.append(x)
    ins, outs, cache = recipe.get_dag_compute_expression_with_inputs(
        compute_key, shape_key, outputs, read_graph)
    sch = tvm.te.create_schedule([x.op for x in outs])
    # print(tvm.lower(sch, ins + outs, simple_mode=True))
    main_intrin_op = cache[recipe.main_capsule_name][0].op
    ii, jj = main_intrin_op.axis
    kk, = main_intrin_op.reduce_axis

    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    Conv = E.op.input_tensors[0]
    n, k, p, q = Conv.op.axis
    rc, rr, rs = Conv.op.reduce_axis

    result = at.IntrinMatchResult(
        recipe, compute_key, shape_key,
        {0: 0}, {1: 1}, {ii: p, jj: k, kk: rc},
        at.compute_dag_from_tensors([E]),
        at.compute_dag_from_tensors(outs))
    print("Pass!\n")


def test2():
    print("##########################")
    print("Test 2")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_arrays = get_tvm_arrays(inputs_ref, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
        intrin_dag.op_lst[1]: target_dag.op_lst[1]
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
    state = at.TransformState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    request = at.TransformRequest(
        ".direct",
        {ii: n.var, jj: k.var, kk: rc.var},
        {n: ii.var, k: jj.var, rc: kk.var}, [n, k, rc], [p, q, rr, rs])
    new_state = at.transform_main_op(state, request)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    # print(tvm.lower(
    #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")
    outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
    func(*inputs_arrays, *outputs_arrays)
    for a, b in zip(outputs_arrays_ref, outputs_arrays):
        testing.assert_allclose(a.asnumpy(), b.asnumpy(), atol=1e-3)
    print("Pass!\n")


def test3():
    print("##########################")
    print("Test 3")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_arrays = get_tvm_arrays(inputs_ref, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
        intrin_dag.op_lst[1]: target_dag.op_lst[1]
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
    state = at.TransformState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    request = at.TransformRequest(
        ".direct",
        {ii: n.var, jj: k.var, kk: rc.var},
        {n: ii.var, k: jj.var, rc: kk.var}, [n, k, rc], [p, q, rr, rs])
    new_state = at.transform_main_op(state, request)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    # print(tvm.lower(
    #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")
    outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
    func(*inputs_arrays, *outputs_arrays)
    for a, b in zip(outputs_arrays_ref, outputs_arrays):
        testing.assert_allclose(a.asnumpy(), b.asnumpy(), atol=1e-3)
    print("Pass!\n")


def test4():
    print("##########################")
    print("Test 4")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_arrays = get_tvm_arrays(inputs_ref, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
        intrin_dag.op_lst[1]: target_dag.op_lst[1]
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
    state = at.TransformState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    no = tvm.tir.IterVar((0, 2), "no", 0)
    ko = tvm.tir.IterVar((0, 4), "ko", 0)
    rco = tvm.tir.IterVar((0, 8), "rco", 2)
    request = at.TransformRequest(
        ".split",
        {ii: n.var % 16, jj: k.var % 16, kk: rc.var %
            16, no: n.var // 16, ko: k.var // 16, rco: rc.var // 16},
        {n: ii.var + no.var * 16, k: jj.var + ko.var * 16,
            rc: kk.var + rco.var * 16},
        [n, k, rc], [p, q, rr, rs, no, ko, rco])
    new_state = at.transform_main_op(state, request)
    print("Compare new state and old state:")
    print("old axis map:", state.axis_map)
    print("new axis map:", new_state.axis_map)
    tmp = []
    for k, v in new_state.axis_map.items():
        tmp.append(v)
    for tri in zip(*tmp):
        print(tri)
    print("old main op map:", state.main_op_map)
    print("new main op map:", new_state.main_op_map)
    print(new_state, state)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    # print(tvm.lower(
    #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")
    outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
    func(*inputs_arrays, *outputs_arrays)
    for a, b in zip(outputs_arrays_ref, outputs_arrays):
        testing.assert_allclose(a.asnumpy(), b.asnumpy(), atol=1e-3, rtol=1e-2)
    print("Pass!\n")


def test5():
    print("##########################")
    print("Test 5")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_arrays = get_tvm_arrays(inputs_ref, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
        intrin_dag.op_lst[1]: target_dag.op_lst[1]
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
    state = at.TransformState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    no = tvm.tir.IterVar((0, 2), "no", 0)
    ko = tvm.tir.IterVar((0, 4), "ko", 0)
    rco = tvm.tir.IterVar((0, 8), "rco", 2)
    request = at.TransformRequest(
        ".fuse",
        {ii: (n.var * 14 + p.var) * 14 + q.var,
         jj: k.var,
         kk: (rc.var * 3 + rr.var) * 3 + rs.var},
        {n: ii.var // (14 * 14),
         p: ii.var % (14 * 14) // 14,
         q: ii.var % 14,
         k: jj.var,
         rc: kk.var // (3 * 3),
         rr: kk.var % (3 * 3) // 3,
         rs: kk.var % 3},
        [n, p, q, k, rc, rr, rs], [])
    new_state = at.transform_main_op(state, request)
    print("Compare new state and old state:")
    print("old axis map:", state.axis_map)
    print("new axis map:", new_state.axis_map)
    tmp = []
    for k, v in new_state.axis_map.items():
        tmp.append(v)
    for tri in zip(*tmp):
        print(tri)
    print("old main op map:", state.main_op_map)
    print("new main op map:", new_state.main_op_map)
    print(new_state, state)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    # print(tvm.lower(
    #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")
    outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
    func(*inputs_arrays, *outputs_arrays)
    for a, b in zip(outputs_arrays_ref, outputs_arrays):
        testing.assert_allclose(a.asnumpy(), b.asnumpy(), atol=1e-3, rtol=1e-2)
    print("Pass!\n")


def test6():
    print("##########################")
    print("Test 6")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_arrays = get_tvm_arrays(inputs_ref, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
        intrin_dag.op_lst[1]: target_dag.op_lst[1]
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
    state = at.TransformState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)

    request = at.TransformRequest(
        ".fuse",
        {ii: (n.var * 14 + p.var) * 14 + q.var,
         jj: k.var,
         kk: (rc.var * 3 + rr.var) * 3 + rs.var},
        {n: ii.var // (14 * 14),
         p: ii.var % (14 * 14) // 14,
         q: ii.var % 14,
         k: jj.var,
         rc: kk.var // (3 * 3),
         rr: kk.var % (3 * 3) // 3,
         rs: kk.var % 3},
        [n, p, q, k, rc, rr, rs], [])
    new_state = at.transform_main_op(state, request)

    target_dag = new_state.target_dag
    wi, wj = target_dag.op_lst[2].axis
    wk, = target_dag.op_lst[2].reduce_axis
    io = tvm.tir.IterVar((0, 13), "io", 0)
    jo = tvm.tir.IterVar((0, 4), "jo", 0)
    ko = tvm.tir.IterVar((0, 72), "ko", 2)
    request = at.TransformRequest(
        ".split",
        {ii: wi % 16, jj: wj % 16, kk: wk % 16,
         io: wi // 16, jo: wj // 16, ko: wk // 16},
        {wi: ii + io * 16, wj: jj + jo * 16,
         wk: kk + ko * 16},
        [wi, wj, wk], [io, jo, ko])
    new_state = at.transform_main_op(new_state, request)

    print("Compare new state and old state:")
    print("old axis map:", state.axis_map)
    print("new axis map:", new_state.axis_map)
    tmp = []
    for k, v in new_state.axis_map.items():
        tmp.append(v)
    for tri in zip(*tmp):
        print(tri)
    print("old main op map:", state.main_op_map)
    print("new main op map:", new_state.main_op_map)
    print(new_state, state)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    print(tvm.lower(
        sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")
    outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
    func(*inputs_arrays, *outputs_arrays)
    for a, b in zip(outputs_arrays_ref, outputs_arrays):
        testing.assert_allclose(a.asnumpy(), b.asnumpy(), atol=1e-3, rtol=1e-2)
    print("Pass!\n")


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
