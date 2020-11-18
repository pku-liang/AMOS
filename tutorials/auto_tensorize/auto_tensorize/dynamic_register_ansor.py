import tvm
import numpy as np
from tvm import auto_scheduler


@auto_scheduler.register_workload
def transformed_gemm(M, N, K, m, n, k):
    A = tvm.te.placeholder([M, K])
    B = tvm.te.placeholder([N, K])
    A1 = tvm.te.compute(
        [A.shape[0] // m, A.shape[1] // k, m, k],
        lambda i, j, ii, jj: A[i * m + ii, j * k + jj])
    B1 = tvm.te.compute(
        [B.shape[0] // n, B.shape[1] // k, n, k],
        lambda i, j, ii, jj: B[i * n + ii, j * k + jj])
    k1 = tvm.te.reduce_axis([0, A.shape[1] // k], name="k1")
    k2 = tvm.te.reduce_axis([0, k], name="k2")
    C1 = tvm.te.compute(
        [M // m, N // n, m, n],
        lambda i, j, ii, jj:
            tvm.te.sum(A1[i, k1, ii, k2] * B1[j, k1, jj, k2], axis=[k1, k2]))
    C2 = tvm.te.compute([M, N],
        lambda i, j: C1[i // m, j // n, i % m, j % n])
    return [A, B, C2]


@auto_scheduler.register_workload
def Gemm(M, N, K):
    A = tvm.te.placeholder([M, K])
    B = tvm.te.placeholder([N, K])
    k = tvm.te.reduce_axis([0, K])
    C = tvm.te.compute(
        [M, N],
        lambda i, j: tvm.te.sum(A[i, k] * B[j, k], axis=k))
    return [A, B, C]


def main():
    def gemm(M, N, K):
        A = tvm.te.placeholder([M, K])
        B = tvm.te.placeholder([N, K])
        k = tvm.te.reduce_axis([0, K])
        C = tvm.te.compute(
            [M, N],
            lambda i, j: tvm.te.sum(A[i, k] * B[j, k], axis=k))
        return [A, B, C]

    def transform_gemm(A, B, C, intrin_shape):
        m, n, k = intrin_shape
        A1 = tvm.te.compute(
            [A.shape[0] // m, A.shape[1] // k, m, k],
            lambda i, j, ii, jj: A[i * m + ii, j * k + jj])
        B1 = tvm.te.compute(
            [B.shape[0] // n, B.shape[1] // k, n, k],
            lambda i, j, ii, jj: B[i * n + ii, j * k + jj])
        k1 = tvm.te.reduce_axis([0, A.shape[1] // k], name="k1")
        k2 = tvm.te.reduce_axis([0, k], name="k2")
        C1 = tvm.te.compute(
            [C.shape[0] // m, C.shape[1] // n, m, n],
            lambda i, j, ii, jj:
                tvm.te.sum(A1[i, k1, ii, k2] * B1[j, k1, jj, k2], axis=[k1, k2]))
        C2 = tvm.te.compute(C.shape,
            lambda i, j: C1[i // m, j // n, i % m, j % n])
        
        def ret():
            return [A, B, C2]
        
        return ret
    
    M = 1024
    N = 32
    K = 1024
    log_file = "dynamic_register_gemm.json"
    args = (M, N, K)
    A, B, C = gemm(*args)
    task_func = transform_gemm(A, B, C, (4, 4, 4))
    registered_func = auto_scheduler.register_workload("dynamic_transformed_gemm", f=task_func)

    target = tvm.target.Target("cuda")

    # the last layer in resnet
    task = auto_scheduler.create_task("dynamic_transformed_gemm", (), target)

    # Inspect the computational graph
    print(task.compute_dag)

    ######################################################################
    # Next, we set parameters for the auto-scheduler. These parameters
    # mainly specify how we do the measurement during the search and auto-tuning.
    #
    # * :code:`measure_ctx` launches a different process for measurement. This
    #   provides an isolation. It can protect the master process from GPU crashes
    #   happended during measurement and avoid other runtime conflicts.
    # * :code:`min_repeat_ms` defines the minimum duration of one "repeat" in every measurement.
    #   This can warmup the GPU, which is necessary to get accurate measurement results.
    #   Typically, we recommend a value > 300 ms.
    # * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
    #   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
    #   good value for the search to converge. You can do more trials according to your time budget.
    # * In addition, we use :code:`RecordToFile` to dump measurement records into a file `conv2d.json`.
    #   The measurement records can be used to query the history best, resume the search,
    #   and do more analyses later.
    # * see :any:`auto_scheduler.TuningOptions`,
    #   :any:`auto_scheduler.LocalRPCMeasureContext` for more parameters.

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    ######################################################################
    # Run the search
    # ^^^^^^^^^^^^^^
    # Now we get all inputs ready. Pretty simple, isn't it?
    # We can kick off the search and let the auto-scheduler do its magic.
    # After some measurement trials, it will return the best schedule it found.

    sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)

    ######################################################################
    # We can lower the schedule to see the IR after auto-scheduling.
    # The auto-scheduler correctly performs optimizations including multi-level tiling,
    # cooperative fetching, unrolling and operator fusion.

    print(tvm.lower(sch, args, simple_mode=True))

    ######################################################################
    # Check correctness and evaluate performance
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # We build the binary and check its correctness and performance.

    func = tvm.build(sch, args, target)

    # check correctness
    data_np = np.random.uniform(size=(M, K)).astype(np.float32)
    weight_np = np.random.uniform(size=(N, K)).astype(np.float32)
    gemm_np = np.matmul(data_np, weight_np.T)

    ctx = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
    gemm_tvm = tvm.nd.empty(gemm_np.shape, ctx=ctx)
    func(data_tvm, weight_tvm, gemm_tvm)

    # Check results
    np.testing.assert_allclose(gemm_np, gemm_tvm.asnumpy(), rtol=1e-3)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
    print(
        "Execution time of this operator: %.3f ms"
        % (np.median(evaluator(data_tvm, weight_tvm, gemm_tvm).results) * 1000)
    )

    ######################################################################
    # Using the record file
    # ^^^^^^^^^^^^^^^^^^^^^
    # During the search, all measuremnt records are dumpped into the record
    # file "conv2d.json". The measurement records can be used to re-apply search results,
    # resume the search, and perform other analyses.

    ######################################################################
    # Here is an example where we load the best schedule from a file,
    # print the equivalent python schedule API, and build the binary again.

    # Load the measuremnt record for the best schedule
    inp, res = auto_scheduler.load_best(log_file, task.workload_key)

    # Print equivalent python schedule API. This can be used for debugging and
    # learning the behavior of the auto-scheduler.
    print("Equivalent python schedule:")
    print(task.compute_dag.print_python_code_from_state(inp.state))

    # Rebuild the binary. This shows how you can apply the best schedule from a
    # log file without reruning the search again.
    sch, args = task.compute_dag.apply_steps_from_state(inp.state)
    func = tvm.build(sch, args, target)

    ######################################################################
    # A more complicated example is to resume the search.
    # In this case, we need to create the search policy and cost model by ourselves
    # and resume the status of search policy and cost model with the log file.
    # In the example below we resume the status and do more 5 trials.


    # log_file = "transformed_gemm.json"
    # cost_model = auto_scheduler.XGBModel()
    # cost_model.update_from_file(log_file)
    # search_policy = auto_scheduler.SketchPolicy(
    #     task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    # )
    # tune_option = auto_scheduler.TuningOptions(
    #     num_measure_trials=5,
    #     runner=measure_ctx.runner,
    #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    # )
    # sch, args = auto_scheduler.auto_schedule(task, search_policy, tuning_options=tune_option)

    # kill the measurement process
    del measure_ctx


if __name__ == "__main__":
    main()