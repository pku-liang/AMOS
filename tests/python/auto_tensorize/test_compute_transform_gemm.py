import tvm
import numpy as np
from tvm import testing
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
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"

    def cond(cur):
        return (
            cur in hw_abs_dag.hw_abs_dict and
            (cur in hw_abs_dag.hw_abs_dict and
             issubclass(hw_abs_dag.hw_abs_dict[cur], at.ComputeAbstraction)))
    op_list, read_graph, feed_graph = hw_abs_dag.serialize_dag(
        cond1=cond
    )
    outputs = []
    for x in op_list:
        if x not in feed_graph:
            outputs.append(x)
    ins, outs, cache = hw_abs_dag.get_dag_compute_expression_with_inputs(
        compute_key, shape_key, outputs, read_graph)
    print("Pass!\n")


def test2():
    print("##########################")
    print("Test 2")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"

    def cond(cur):
        return (
            cur in hw_abs_dag.hw_abs_dict and
            (cur in hw_abs_dag.hw_abs_dict and
             issubclass(hw_abs_dag.hw_abs_dict[cur], at.ComputeAbstraction)))
    op_list, read_graph, feed_graph = hw_abs_dag.serialize_dag(
        cond1=cond
    )
    outputs = []
    for x in op_list:
        if x not in feed_graph:
            outputs.append(x)
    ins, outs, cache = hw_abs_dag.get_dag_compute_expression_with_inputs(
        compute_key, shape_key, outputs, read_graph)
    sch = tvm.te.create_schedule([x.op for x in outs])
    # print(tvm.lower(sch, ins + outs, simple_mode=True))
    main_intrin_op = cache[hw_abs_dag.main_hw_abs_name][0].op
    A, B, D, E = gemm(1024, 1024, 1024)
    C = E.op.input_tensors[0]
    i, j = C.op.axis
    k, = C.op.reduce_axis
    ii, jj = main_intrin_op.axis
    kk, = main_intrin_op.reduce_axis
    result = at.IntrinMatchResult(
        hw_abs_dag, compute_key, shape_key, {0: 0}, {1: 1}, {ii: i, jj: j, kk: k},
        at.compute_dag_from_tensors([E]),
        at.compute_dag_from_tensors(outs))
    print("Pass!\n")


def test3():
    print("##########################")
    print("Test 3")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"
    compute_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)

    inputs = compute_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in compute_dag.tensors])
    # print(tvm.lower(sch, inputs + list(compute_dag.tensors), simple_mode=True))
    print("Pass!\n")


def test4():
    print("##########################")
    print("Test 4")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    A, B, D, E = gemm(128, 128, 128)
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
    i, j = target_dag.op_lst[0].axis
    k, = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [i], jj: [j], kk: [k]
    }
    state = at.MappingState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    request = at.MappingRequest(
        ".direct",
        {ii: i.var, jj: j.var, kk: k.var},
        {i: ii.var, j: jj.var, k: kk.var}, [i, j, k], [])
    new_state = at.mapping_main_op(state, request)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    # print(tvm.lower(
    #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")
    outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
    func(*inputs_arrays, *outputs_arrays)
    for a, b in zip(outputs_arrays_ref, outputs_arrays):
        testing.assert_allclose(a.asnumpy(), b.asnumpy())
    print("Pass!\n")


def test5():
    print("##########################")
    print("Test 5")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    A, B, D, E = gemm(128, 128, 128)
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
    i, j = target_dag.op_lst[0].axis
    k, = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [i], jj: [j], kk: [k]
    }
    state = at.MappingState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    request = at.MappingRequest(
        ".direct",
        {ii: i.var, jj: j.var, kk: k.var},
        {i: ii.var, j: jj.var, k: kk.var}, [i, j, k], [])
    new_state = at.mapping_main_op(state, request)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    # print(tvm.lower(
    #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")
    outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
    func(*inputs_arrays, *outputs_arrays)
    for a, b in zip(outputs_arrays_ref, outputs_arrays):
        testing.assert_allclose(a.asnumpy(), b.asnumpy())
    print("Pass!\n")


def test6():
    print("##########################")
    print("Test 6")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    A, B, D, E = gemm(128, 128, 128)
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
    i, j = target_dag.op_lst[0].axis
    k, = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [i], jj: [j], kk: [k]
    }
    state = at.MappingState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    io = tvm.tir.IterVar((0, 8), "io", 0)
    jo = tvm.tir.IterVar((0, 8), "jo", 0)
    ko = tvm.tir.IterVar((0, 8), "ko", 2)
    request = at.MappingRequest(
        ".split",
        {ii: i.var % 16, jj: j.var % 16, kk: k.var %
            16, io: i.var // 16, jo: j.var // 16, ko: k.var // 16},
        {i: ii.var + io.var * 16, j: jj.var + jo.var * 16,
            k: kk.var + ko.var * 16},
        [i, j, k], [io, jo, ko])
    new_state = at.mapping_main_op(state, request)
    print("Compare new state and old state:")
    print("old axis map:", state.axis_map)
    print("new axis map:", new_state.axis_map)
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
        testing.assert_allclose(a.asnumpy(), b.asnumpy())
    print("Pass!\n")


def test7():
    print("##########################")
    print("Test 7")
    hw_abs_dag = at.WMMAFp16Fp32Bias()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag, _ = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    A, B, D, E = gemm(128, 128, 128)
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
    i, j = target_dag.op_lst[0].axis
    k, = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [i], jj: [j], kk: [k]
    }
    state = at.MappingState(
        main_op_map, elem_op_map, axis_map, target_dag, intrin_dag)
    io = tvm.tir.IterVar((0, 8), "io", 0)
    jo = tvm.tir.IterVar((0, 8), "jo", 0)
    ko = tvm.tir.IterVar((0, 8), "ko", 2)
    request = at.MappingRequest(
        ".split",
        {ii: i.var % 16, jj: j.var % 16, kk: k.var %
            16, io: i.var // 16, jo: j.var // 16, ko: k.var // 16},
        {i: ii.var + io.var * 16, j: jj.var + jo.var * 16,
            k: kk.var + ko.var * 16},
        [i, j, k], [io, jo, ko])
    new_state = at.mapping_main_op(state, request)

    target_dag = new_state.target_dag
    wi, wj, hi, hj = target_dag.op_lst[2].axis
    wk, hk = target_dag.op_lst[2].reduce_axis
    request = at.MappingRequest(
        ".fuse",
        {ii: wi * 16 + hi, jj: wj * 16 + hj, kk: wk * 16 + hk},
        {wi: ii // 16, hi: ii % 16, wj: jj // 16, hj: jj %
            16, wk: kk // 16, hk: kk % 16},
        [wi, wj, hi, hj, wk, hk], [])
    new_state = at.mapping_main_op(new_state, request)

    new_target_dag = new_state.target_dag
    new_inputs = new_target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
    # print(tvm.lower(
    #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True))
    func = tvm.build(sch, new_inputs + list(new_target_dag.tensors), "llvm")
    outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
    func(*inputs_arrays, *outputs_arrays)
    for a, b in zip(outputs_arrays_ref, outputs_arrays):
        testing.assert_allclose(a.asnumpy(), b.asnumpy())
    print("Pass!\n")


if __name__ == "__main__":
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
