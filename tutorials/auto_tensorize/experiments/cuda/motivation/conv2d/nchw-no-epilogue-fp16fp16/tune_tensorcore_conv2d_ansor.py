import tvm
from tvm import auto_tensorize as at
from functools import reduce
from tvm import auto_scheduler
import numpy as np


def conv2d_implicit_gemm_nchw(N, C, H, W, K, R, S, stride, padding, m, n, k, compute_key, shape_key):
    dtype = "float16"
    out_dtype = "float16"
    OC = (C + k - 1) // k
    IC = C - OC * k
    A = tvm.te.placeholder([N, C, H, W], dtype=dtype)
    B = tvm.te.placeholder([K, C, R, S], dtype=dtype)
    pH = (H + 2 * padding - R) // stride + 1
    pW = (W + 2 * padding - S) // stride + 1
    GM = N * pH * pW
    GK = C * R * S
    GN = K
    TM = (GM + m - 1) // m
    TK = (GK + k - 1) // k
    TN = (GN + n - 1) // n

    def get_n(val):
        return val // (pH * pW)

    def get_c(val):
        return val // (R * S)


    def get_h(val):
        n = get_n(val)
        hw = val - n * (pH * pW)
        return hw // pW

    def get_w(val):
        n = get_n(val)
        hw = val - n * (pH * pW)
        h = hw // pW
        return hw - h * pW

    def get_r(val):
        c = get_c(val)
        rs = val - c * (R * S)
        return rs // S

    def get_s(val):
        c = get_c(val)
        rs = val - c * (R * S)
        r = rs // S
        return rs - r * S

    Pad = tvm.te.compute(
        [N, C, H + 2 * padding, W + 2 * padding],
        lambda i, c, h, w:
            tvm.tir.if_then_else(
                tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
                A[i, c, h - padding, w - padding],
                tvm.tir.const(0.0, dtype)
            ),
        name="Pad"
    )

    A1 = tvm.te.compute(
        [TM, TK, m, k],
        lambda i, j, ii, jj:
            tvm.tir.if_then_else(
                tvm.tir.all(i * m + ii < GM, j * k + jj < GK),
                Pad[get_n(i * m + ii),
                  get_c(j * k + jj),
                  get_h(i * m + ii) * stride + get_r(j * k + jj),
                  get_w(i * m + ii) * stride + get_s(j * k + jj)],
                tvm.tir.const(0.0, dtype)),
            name="A1")
    B1 = tvm.te.compute(
        [TN, TK, n, k],
        lambda i, j, ii, jj:
            tvm.tir.if_then_else(
                tvm.tir.all(i * n + ii < GN, j * k + jj < GK),
                B[i * n + ii,
                  get_c(j * k + jj),
                  get_r(j * k + jj),
                  get_s(j * k + jj)],
                tvm.tir.const(0.0, dtype)),
            name="B1")
    rk1 = tvm.te.reduce_axis([0, TK], name="rk1")
    rk2 = tvm.te.reduce_axis([0, k], name="rk2")
    C1 = tvm.te.compute(
        [TM, TN, m, n],
        lambda i, j, ii, jj:
            tvm.te.sum(
                (A1[i, rk1, ii, rk2] * B1[j, rk1, jj, rk2]).astype(out_dtype),
            axis=[rk1, rk2]))
    
    recipe = at.WMMAFp16Fp16()
    input_names, output_names, nodes, read_graph, feed_graph = \
        at.construct_dag(
            recipe, compute_key, shape_key, [A1, B1], [C1], [], [C1])
    output_tensors = reduce(
        lambda x, y: x + y, [nodes[x] for x in output_names], [])
    C1 = output_tensors[0]

    # def assemble(i, ph, pw):
    #     return i * pH * pW + ph * pW + pw
    # C2 = tvm.te.compute([N, K, pH, pW],
    #     lambda n, ok, ph, pw:
    #         C1[assemble(n, ph, pw)// m,
    #            ok // n,
    #            assemble(n, ph, pw) % m,
    #            ok % n],
    #         name="C2")
    return [A, B, C1]


def run(N, C, H, W, K, R, S, stride, padding, log_file):
    # Map<te::Operation, String> operation_role_,
    # String recipe_key_,
    # String compute_key_,
    # String shape_key_,
    # Map<te::Operation, IntImm> reserve_inner_axis_count_,
    # Array<IntImm> main_op_reserve_reduce_axis_,
    # Array<IntImm> main_op_reserve_reduce_axis_factor_
    m, n, k = (16, 16, 16)
    compute_key = "ntn"
    shape_key = "x".join([str(x) for x in [m, n, k]])
    A, B, C2 = conv2d_implicit_gemm_nchw(
        N, C, H, W, K, R, S, stride, padding, m, n, k, compute_key, shape_key)

    store = C2  # .op.input_tensors[0]
    Mma = store.op.input_tensors[0]
    load_A, load_B = Mma.op.input_tensors
    A1 = load_A.op.input_tensors[0]
    B1 = load_B.op.input_tensors[0]
    operation_role = {
        load_A.op: at.OperationRole.load_op,
        load_B.op: at.OperationRole.load_op,
        Mma.op: at.OperationRole.main_op,
        store.op: at.OperationRole.output_op
    }
    recipe_key = "wmma_fp16_fp16"
    capsule_map = {
        load_A.op: "load_a",
        load_B.op: "load_b",
        Mma.op: "mma",
        store.op: "store"
    }
    reserve_inner_axis_count = {
        load_A.op: 2,
        load_B.op: 2,
        Mma.op: 2,
        store.op: 2
    }
    main_op_reserve_reduce_axis = [
        1
    ]
    main_op_reserve_reduce_axis_factor = [
        16
    ]
    load_from_shared = {
        load_A.op: 1,
        load_B.op: 1
    }
    store_to_shared = {
        store.op: 0
    }
    recipe_stage = at.RecipeStage(
        operation_role,
        "cuda",
        recipe_key,
        compute_key,
        shape_key,
        capsule_map,
        reserve_inner_axis_count,
        main_op_reserve_reduce_axis,
        main_op_reserve_reduce_axis_factor,
        load_from_shared,
        store_to_shared,
        at.InstructionScope.warp
    )
    def task_func():
        return [A, B, C2]

    registered_func = auto_scheduler.register_workload(
        log_file[:-5], f=task_func)

    target = tvm.target.Target("cuda")

    # the last layer in resnet
    task = auto_scheduler.create_task(
        log_file[:-5], (), target, recipe=recipe_stage)

    # Inspect the computational graph
    print(task.compute_dag)

    print(task.compute_dag.init_state)

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    ######################################################################
    # Run the search
    # ^^^^^^^^^^^^^^
    # Now we get all inputs ready. Pretty simple, isn't it?
    # We can kick off the search and let the auto-scheduler do its magic.
    # After some measurement trials, it will return the best schedule it found.

    from tvm.auto_scheduler.cost_model import RandomModel, XGBModel
    from tvm.auto_scheduler.search_policy import SketchPolicy
    cost_model = RandomModel()
    search_policy = SketchPolicy(task, cost_model)
    sch, args = auto_scheduler.auto_schedule(task, search_policy=search_policy, tuning_options=tune_option)

    ######################################################################
    # We can lower the schedule to see the IR after auto-scheduling.
    # The auto-scheduler correctly performs optimizations including multi-level tiling,
    # cooperative fetching, unrolling and operator fusion.

    # print(tvm.lower(sch, args, simple_mode=True))

    inp, res = auto_scheduler.load_best(log_file, task.workload_key)
    sch, args = task.compute_dag.apply_steps_from_state(inp.state)
    print(tvm.lower(sch, args, simple_mode=True))

    # Print equivalent python schedule API. This can be used for debugging and
    # learning the behavior of the auto-scheduler.
    print("Equivalent python schedule:")
    print(task.compute_dag.print_python_code_from_state(inp.state))

    ######################################################################
    # Check correctness and evaluate performance
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # We build the binary and check its correctness and performance.

    func = tvm.build(sch, args, target)
    print(func.imported_modules[0].get_source())

    # check correctness
    data_np = np.random.uniform(size=[int(x) for x in args[0].shape]).astype(np.float16)
    weight_np = np.random.uniform(size=[int(x) for x in args[1].shape]).astype(np.float16)
    output_np = np.random.uniform(size=[int(x) for x in args[2].shape]).astype(np.float16)

    ctx = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
    gemm_tvm = tvm.nd.array(output_np, ctx=ctx)
    print(gemm_tvm.dtype)
    func(data_tvm, weight_tvm, gemm_tvm)

    # Check results
    # np.testing.assert_allclose(gemm_np, gemm_tvm.asnumpy(), rtol=1e-2)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
    cost = (np.median(evaluator(data_tvm, weight_tvm, gemm_tvm).results) * 1000)
    print(
        "Execution time of this operator: %.3f ms"
        % cost
    )
    print("GFLOPS=%f" % (N*C*H*W*K*R*S/stride/stride*2/cost*1e3/1e9))
    return cost


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
    results = []
    batches = [1]
    for batch in batches:
        results.append([])
        beg = 2
        end = beg + 15
        for i, shape in enumerate(yolo_shapes_b1[beg:end]):
            _, C, H, W, K, _, R, S, _, stride, padding, _, _ = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            try:
                res = run(
                    N, C, H, W, K, R, S, stride, padding,
                    "Conv2d_batch_" + str(batch) + "_layer" + str(i+beg) + ".json")
                results[-1].append(res)
            except Exception as e:
                results[-1].append("inf")
                print(type(e), e)
    for i, res_lst in enumerate(results):
        print("batch=", batches[i])
        for res in res_lst:
            print(res)
        print("\n")