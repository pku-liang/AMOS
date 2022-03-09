import tvm
from tvm import auto_tensorize as at
from functools import reduce
from tvm import auto_scheduler
import numpy as np


def transformed_gemm(M, N, K, m, n, k, compute_key, shape_key):
    dtype = "float16"
    out_dtype = "float32"
    A = tvm.te.placeholder([M, K], dtype=dtype)
    B = tvm.te.placeholder([N, K], dtype=dtype)
    A1 = tvm.te.compute(
        [A.shape[0] // m, A.shape[1] // k, m, k],
        lambda i, j, ii, jj: A[i * m + ii, j * k + jj], name="A1")
    B1 = tvm.te.compute(
        [B.shape[0] // n, B.shape[1] // k, n, k],
        lambda i, j, ii, jj: B[i * n + ii, j * k + jj], name="B1")
    k1 = tvm.te.reduce_axis([0, A.shape[1] // k], name="k1")
    k2 = tvm.te.reduce_axis([0, k], name="k2")
    C1 = tvm.te.compute(
        [M // m, N // n, m, n],
        lambda i, j, ii, jj:
            tvm.te.sum((A1[i, k1, ii, k2] * B1[j, k1, jj, k2]).astype(out_dtype), axis=[k1, k2]))
    
    hw_abs_dag = at.WMMAFp16Fp32()
    input_names, output_names, nodes, read_graph, feed_graph = \
        at.construct_dag(
            hw_abs_dag, compute_key, shape_key, [A1, B1], [C1], [], [C1])
    output_tensors = reduce(
        lambda x, y: x + y, [nodes[x] for x in output_names], [])
    C1 = output_tensors[0]
    C2 = tvm.te.compute([M, N],
        lambda i, j: C1[i // m, j // n, i % m, j % n], name="C2")
    return [A, B, C2]


def main():
    # Map<te::Operation, String> operation_role_,
    # String hw_abs_dag_key_,
    # String compute_key_,
    # String shape_key_,
    # Map<te::Operation, IntImm> reserve_inner_axis_count_,
    # Array<IntImm> main_op_reserve_reduce_axis_,
    # Array<IntImm> main_op_reserve_reduce_axis_factor_
    M, N, K, m, n, k = (1024, 1024, 1024, 16, 16, 16)
    log_file = "test_gemm.json"
    compute_key = "ntn"
    shape_key = "x".join([str(x) for x in [m, n, k]])
    A, B, C2 = transformed_gemm(M, N, K, m, n, k, compute_key, shape_key)

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
    hw_abs_dag_key = "wmma_fp16_fp32"
    hw_abs_map = {
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
    hw_abs_dag_stage = at.HwAbsDAGStage(
        operation_role,
        "cuda",
        hw_abs_dag_key,
        compute_key,
        shape_key,
        hw_abs_map,
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
        "dynamic_transformed_gemm", f=task_func)

    target = tvm.target.Target("cuda")

    # the last layer in resnet
    task = auto_scheduler.create_task(
        "dynamic_transformed_gemm", (), target, hw_abs_dag=hw_abs_dag_stage)

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

    from tvm.auto_scheduler.cost_model import RandomModel
    from tvm.auto_scheduler.search_policy import SketchPolicy
    cost_model = RandomModel()
    search_policy = SketchPolicy(task, cost_model)
    # sch, args = auto_scheduler.auto_schedule(task, search_policy=search_policy, tuning_options=tune_option)

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
    data_np = np.random.uniform(size=(M, K)).astype(np.float16)
    weight_np = np.random.uniform(size=(N, K)).astype(np.float16)
    gemm_np = np.matmul(data_np, weight_np.T)

    ctx = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
    gemm_tvm = tvm.nd.empty(gemm_np.shape, ctx=ctx)
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
    print("GFLOPS=%f" % (M*N*K*2/cost*1e3/1e9))


if __name__ == "__main__":
    main()