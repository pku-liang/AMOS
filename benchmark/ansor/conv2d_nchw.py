import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data", dtype="float16")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel", dtype="float16")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    return [data, kernel, conv]


def ansor_tune(N, H, W, CO, CI, KH, KW, strides, padding, target, layer):
    P = (H + 2 * padding - KH) // stride + 1
    Q = (W + 2 * padding - KW) // stride + 1
    strides = (strides, strides)
    padding = (padding, padding)
    task = auto_scheduler.create_task(
        conv2d_layer, (N, H, W, CO, CI, KH, KW, strides, padding), target)
    
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=1000,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile("conv2d_" + str(layer) + ".json")],
    )

    sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)

    inp, res = auto_scheduler.load_best("conv2d_" + str(layer) + ".json", task.workload_key)

    # Print equivalent python schedule API. This can be used for debugging and
    # learning the behavior of the auto-scheduler.
    print("Equivalent python schedule:")
    print(task.compute_dag.print_python_code_from_state(inp.state))

    # Rebuild the binary. This shows how you can apply the best schedule from a
    # log file without reruning the search again.
    sch, args = task.compute_dag.apply_steps_from_state(inp.state)
    func = tvm.build(sch, args, target)

    # check correctness
    data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float16)
    weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float16)
    # bias_np = np.random.uniform(size=(1, CO, 1, 1)).astype(np.float32)
    # conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
    # out_np = np.maximum(conv_np + bias_np, 0.0)
    conv_np = np.random.uniform(size=[N, CO, P, Q]).astype(np.float32)

    ctx = tvm.gpu()
    data_tvm = tvm.nd.array(data_np, ctx=ctx)
    weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
    # bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
    out_tvm = tvm.nd.empty(conv_np.shape, ctx=ctx)
    func(data_tvm, weight_tvm, out_tvm)

    # Check results
    # np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

    # Evaluate execution time
    evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
    cost = np.median(evaluator(data_tvm, weight_tvm, out_tvm).results) * 1000
    print(
    "Execution time of this operator: %.3f ms"
    % (cost))
    return cost


res18_shapes_b1 = [
    # resnet-18
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    #num = 15
    num = 12
    target = tvm.target.Target("cuda")
    for batch in batches:
        costs = []
        #for i, shape in enumerate(yolo_shapes_b1[beg:beg+num]):
        for i, shape in enumerate(res18_shapes_b1[beg:beg+num]):
            (_, C, H, W, K, _, R, S, _, stride,
                padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            # N, H, W, CO, CI, KH, KW, strides, padding, target, layer
            try:
                cost = ansor_tune(
                    N, H, W, K, C, R, S, stride,
                    padding, target, i + 1
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
