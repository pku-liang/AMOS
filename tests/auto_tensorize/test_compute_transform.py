import tvm
from tvm import auto_tensorize as at


def gemm(M, N, K):
    A = tvm.te.placeholder([M, K], dtype="float16", name="A")
    B = tvm.te.placeholder([K, N], dtype="float16", name="B")
    tk = tvm.te.reduce_axis([0, K], name="tk")
    C = tvm.te.compute(
        [M, N],
        lambda ti, tj:
            tvm.te.sum((A[ti, tk] * B[tk, tj]).astype("float32"), axis=tk),
        name="C"
    )
    D = tvm.te.placeholder([M, N], dtype="float32", name="D")
    E = tvm.te.compute(
        [M, N],
        lambda si, sj: C[si, sj] + D[si, sj],
        name="E"
    )
    return [A, B, D, E]


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


def test2():
    print("##########################")
    print("Test 2")
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
    print(tvm.lower(sch, ins + outs, simple_mode=True))
    main_intrin_op = cache[recipe.main_capsule_name][0].op
    A, B, D, E = gemm(1024, 1024, 1024)
    C = E.op.input_tensors[0]
    i, j = C.op.axis
    k, = C.op.reduce_axis
    ii, jj = main_intrin_op.axis
    kk, = main_intrin_op.reduce_axis
    result = at.IntrinMatchResult(
        recipe, compute_key, shape_key, {0: 0}, {1: 1}, {ii: i, jj: j, kk: k},
        at.compute_dag_from_tensors([E]),
        at.compute_dag_from_tensors(outs))


def test3():
    print("##########################")
    print("Test 3")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"
    compute_dag = recipe.get_effective_compute_dag(compute_key, shape_key)

    inputs = compute_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in compute_dag.tensors])
    print(tvm.lower(sch, inputs + list(compute_dag.tensors), simple_mode=True))


def test4():
    print("##########################")
    print("Test 4")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, D, E = gemm(1024, 1024, 1024)
    target_dag = at.compute_dag_from_tensors([E])
    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
        intrin_dag.op_lst[1]: target_dag.op_lst[1]
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    i, j = target_dag.op_lst[0].axis
    k, = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [i], jj: [j], kk: [k]
    }
    state = at.TransformState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    request = at.TransformRequest(
        {ii: i.var, jj: j.var, kk: k.var},
        {i: ii.var, j: jj.var, k: kk.var}, [i, j, k], [])
    new_state = at.transform_main_op(state, request)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    print(tvm.lower(
        sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))


def test5():
    print("##########################")
    print("Test 5")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, D, E = gemm(1024, 1024, 1024)
    target_dag = at.compute_dag_from_tensors([E])
    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
        intrin_dag.op_lst[1]: target_dag.op_lst[1]
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    i, j = target_dag.op_lst[0].axis
    k, = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [i], jj: [j], kk: [k]
    }
    state = at.TransformState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    request = at.TransformRequest(
        {ii: i.var, jj: j.var, kk: k.var},
        {i: ii.var, j: jj.var, k: kk.var}, [i, j, k], [])
    new_state = at.transform_main_op(state, request)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    print(tvm.lower(
        sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))


def test6():
    print("##########################")
    print("Test 6")
    recipe = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, D, E = gemm(1024, 1024, 1024)
    target_dag = at.compute_dag_from_tensors([E])
    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
        intrin_dag.op_lst[1]: target_dag.op_lst[1]
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    i, j = target_dag.op_lst[0].axis
    k, = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [i], jj: [j], kk: [k]
    }
    state = at.TransformState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    io = tvm.tir.IterVar((0, 64), "io", 0)
    ii_ = tvm.tir.IterVar((0, 16), "ii", 0)
    jo = tvm.tir.IterVar((0, 64), "jo", 0)
    ji = tvm.tir.IterVar((0, 16), "ji", 0)
    ko = tvm.tir.IterVar((0, 64), "ko", 2)
    ki = tvm.tir.IterVar((0, 16), "ki", 2)
    request = at.TransformRequest(
        {ii: i.var % 16, jj: j.var % 16, kk: k.var % 16, io: i.var // 16, jo: j.var // 16, ko: k.var // 16},
        {i: ii.var + io.var * 16, j: jj.var + jo.var * 16, k: kk.var + ko.var * 16},
        [i, j, k], [io, jo, ko])
    new_state = at.transform_main_op(state, request)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    print(tvm.lower(
        sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
