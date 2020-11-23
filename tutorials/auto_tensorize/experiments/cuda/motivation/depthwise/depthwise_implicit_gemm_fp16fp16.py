import tvm
from tvm import auto_tensorize as at
from functools import reduce
from tvm import auto_scheduler
import numpy as np


def depthwise_implicit_gemm_nchw(N, C, H, W, K, R, S, stride, padding, m, n, k, compute_key, shape_key):
    dtype = "float16"
    out_dtype = "float16"
    A = tvm.te.placeholder([N, C, H, W], dtype=dtype)
    B = tvm.te.placeholder([K, R, S], dtype=dtype)
    pH = (H + 2 * padding - R) // stride + 1
    pW = (W + 2 * padding - S) // stride + 1
    GM = N * pH * pW
    GK = R * S
    GN = K
    TM = (GM + m - 1) // m
    TK = (GK + k - 1) // k
    TN = (GN + n - 1) // n

    def get_n(val):
        return val // (pH * pW)

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
        return val // S

    def get_s(val):
        r = val // S
        return val - r * S

    A1 = tvm.te.compute(
        [C, TM, TK, m, k],
        lambda c, i, j, ii, jj:
            tvm.tir.if_then_else(
                tvm.tir.all(i * m + ii < GM, j * k + jj < GK),
                A[get_n(i * m + ii),
                  c,
                  get_h(i * m + ii) * stride + get_r(j * k + jj),
                  get_w(i * m + ii) * stride + get_s(j * k + jj)],
                tvm.tir.const(0.0, dtype)),
            name="A1")
    B1 = tvm.te.compute(
        [TN, TK, n, k],
        lambda c, i, j, ii, jj:
            tvm.tir.if_then_else(
                tvm.tir.all(i * n + ii < GN, j * k + jj < GK),
                B[i * n + ii,
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
                (A1[(j * k + jj) // (K // C), i, rk1, ii, rk2] * B1[j, rk1, jj, rk2]).astype(out_dtype),
            axis=[rk1, rk2]))
    
    recipe = at.WMMAFp16Fp16()
    input_names, output_names, nodes, read_graph, feed_graph = \
        at.construct_dag(
            recipe, compute_key, shape_key, [A1, B1], [C1], [], [C1])
    output_tensors = reduce(
        lambda x, y: x + y, [nodes[x] for x in output_names], [])
    C1 = output_tensors[0]

    def assemble(i, ph, pw):
        return i * pH * pW + ph * pW + pw
    C2 = tvm.te.compute([N, K, pH, pW],
        lambda i, ok, ph, pw:
            C1[assemble(i, ph, pw)// m,
               ok // n,
               assemble(i, ph, pw) % m,
               ok % n],
            name="C2")
    return [A, B, C2]


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
    A, B, C2 = depthwise_implicit_gemm_nchw(
        N, C, H, W, K, R, S, stride, padding, m, n, k, compute_key, shape_key)

    store = C2.op.input_tensors[0]
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


mobilev2_shapes_b1 = [
    (1, 32, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1, 32),
    (1, 16, 112, 112, 16 * 6, 16, 3, 3, 1, 2, 1, 1, 16),
    (1, 24, 56, 56, 24 * 6, 24, 3, 3, 1, 2, 1, 1, 24),
    (1, 32, 28, 28, 32 * 6, 32, 3, 3, 1, 2, 1, 1, 32),
    (1, 64, 14, 14, 64 * 6, 64, 3, 3, 1, 1, 1, 1, 64),
    (1, 96, 14, 14, 96 * 6, 96, 3, 3, 1, 2, 1, 1, 96),
    (1, 160, 7, 7, 160 * 6, 160, 3, 3, 1, 1, 1, 1, 160),
]


if __name__ == "__main__":
    results = []
    batches = [1]
    for batch in batches:
        results.append([])
        beg = 0
        end = beg + 7
        for i, shape in enumerate(mobilev2_shapes_b1[beg:end]):
            _, C, H, W, K, _, R, S, _, stride, padding, _, _ = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            try:
                res = run(
                    N, C, H, W, K, R, S, stride, padding,
                    "Depthwise_batch_" + str(batch) + "_layer" + str(i+beg) + ".json")
                results[-1].append(res)
            except Exception as e:
                results[-1].append("inf")
                print(type(e), e)
    for i, res_lst in enumerate(results):
        print("batch=", batches[i])
        for res in res_lst:
            print(res)
        print("\n")