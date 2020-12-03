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


def test1():
    print("##########################")
    print("Test 1")
    recipe = at.WMMAFp16Fp32()
    compute_key = "ntn"
    shape_key = "16x16x16"
    intrin_dag = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
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
    for i in range(1):
        record = gen.get()
        record.unfold_choice = ([1, 0, 0, 0, 0, 0, 1], record.unfold_choice[1])
        print(record.to_json())
        app = at.TransformApplier(match_result)
        new_state = app.apply(record)

        # print("Compare new state and old state:")
        # print("new axis map:", new_state.axis_map)
        # tmp = []
        # for k, v in new_state.axis_map.items():
        #     tmp.append(v)
        # for tri in zip(*tmp):
        #     print(tri)
        # print("new main op map:", new_state.main_op_map)

        # new_target_dag = new_state.target_dag
        # print("org dag len:", len(new_target_dag.op_lst))
        # new_target_main_op = None
        # for k, v in new_state.main_op_map.items():
        #     new_target_main_op = v
        # assert new_target_main_op is not None

        # new_target_dag, _ = at.reconstruct_dag_as_intrin(
        #     new_target_dag, new_target_main_op, recipe, compute_key, shape_key)
        # print("new dag len:", len(new_target_dag.op_lst))

        # print("new dag load A op:",
        #       new_target_dag.op_lst[2].axis, new_target_dag.op_lst[2].body)
        # print("new dag load B op:",
        #       new_target_dag.op_lst[5].axis, new_target_dag.op_lst[5].body)
        # print("new dag main op:",
        #       new_target_dag.op_lst[6].axis, new_target_dag.op_lst[6].body)
        # print("new dag store op:",
        #       new_target_dag.op_lst[7].axis, new_target_dag.op_lst[7].body)

        schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
        params = schedule_gen.get()
        my_params = {
            'vectorize': (1, 1),
            'spatial_factors': [([1, 1, 1], (0, 0)), ([4, 1, 1], (-1, 1)), ([14, 1, 1], (-1, -1))],
            'reduce_factors': [([3, 3, 4], (1, 1)), ([1, 1, 3], (0, -1))],
            'last_factors': [([-1, 32], (-1,))],
            'output_unroll_step': (64, -1),
            'last_unroll_step': (512, 1)}
        params.from_json(my_params)
        print(params.to_json())
        
        new_target_dag = sc_info.target_dag
        new_inputs = new_target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
        # print("new dag len:", len(new_target_dag.op_lst))

        # print("new dag load A op:",
        #       new_target_dag.op_lst[2].axis, new_target_dag.op_lst[2].body)
        # print("new dag load B op:",
        #       new_target_dag.op_lst[5].axis, new_target_dag.op_lst[5].body)
        # print("new dag main op:",
        #       new_target_dag.op_lst[6].axis, new_target_dag.op_lst[6].body)
        # print("new dag store op:",
        #       new_target_dag.op_lst[7].axis, new_target_dag.op_lst[7].body)

        schedule_app.apply(sch, params)

        # print(tvm.lower(
        #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True), flush=True)
        func = tvm.build(sch, new_inputs +
                         list(new_target_dag.tensors), "cuda")
        # print(func.imported_modules[0].get_source())
        ctx = tvm.gpu()
        inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
        outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
        func(*inputs_arrays, *outputs_arrays)
        # for a, b in zip(outputs_arrays_ref, outputs_arrays):
        #     testing.assert_allclose(
        #         a.asnumpy(), b.asnumpy(), atol=1e-3, rtol=1e-2)

        gen.feedback(record, np.random.random())
    print("Pass!\n")


if __name__ == "__main__":
    test1()
