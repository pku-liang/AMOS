import tvm
import os
from tvm import auto_tensorize as at
import time
from tvm.te import schedule
import numpy as np
from tempfile import mkstemp
from tvm import rpc
from tvm.contrib import ndk
from traceback import print_exc


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="int8", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="int8", name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                h >= padding, h - padding < H,
                w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype)
        ),
        name="Pad")

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (pH - R) // stride + 1
    Q = (pW - S) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
            tvm.te.sum((Pad[n, rc, p*stride+rr, q*stride+rs] * B[k, rc, rr, rs]
                        ).astype("int8"), axis=[rc, rr, rs]),
        name="Conv"
    )
    # bias = tvm.te.placeholder([K], dtype="float32", name="bias")
    # E = tvm.te.compute(
    #     [N, K, P, Q],
    #     lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bk],
    #     name="E"
    # )
    return [A, B, Conv]


def tensorize_tensorcore_s8s8(
    N, C, H, W, K, R, S, stride,
    padding, dilation, layer
):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    
    target = "opencl"
    target_host = 'llvm -mtriple=aarch64-linux-android'

    log_file = "opencl-conv2d-int8-layer-%d-batch-%d.log" % (layer, N)

    trials = 1000
    measure_opt = at.MeasureOptions(
        target=target, target_host=target_host, timeout=40, number=10,
        min_repeat_ms=80, build_func="ndk", key="android",
        host="0.0.0.0", port=9190, cooldown_interval=5)

    print("Begin tensorize...")
    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, verbose=True,
        transform_dump=True, runner=at.pebble_rpc_runner_run, search_batch=8)
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app
    print("Tensorize done.")

    # load from file
    schedule_gen.load_from_file(log_file, clear=True)
    entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in file
    params, value = entry.record, 1 / entry.value
    print(value)
    print(params.to_json())

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)
    return cost


def run(N, C, H, W, K, R, S, stride,
        padding, dilation, layer):
    return tensorize_tensorcore_s8s8(
        N, C, H, W, K, R, S, stride,
        padding, dilation, layer)


small_conv_shapes_b1 = [
    # resnet-18
    (1, 32, 112, 112, 32, 32, 3, 3, 1, 1, 1, 1, 1),  # conv1  0
    (1, 16, 112, 112, 96, 16, 3, 3, 1, 2, 1, 1, 1),  # conv2   1
    (1, 24, 56, 56, 144, 24, 3, 3, 1, 2, 1, 1, 1),  # conv3   2
    (1, 32, 28, 28, 192, 32, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (1, 64, 14, 14, 384, 64, 3, 3, 1, 1, 1, 1, 1),  # conv5   4
    (1, 96, 14, 14, 576, 96, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 160, 7, 7, 960, 160, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = 1

    cmds = [
        "adb reverse tcp:9190 tcp:9190",
        "adb forward tcp:5001 tcp:5001",
        "adb shell am start -n org.apache.tvm.tvmrpc/org.apache.tvm.tvmrpc.MainActivity 1> /dev/null 2> /dev/null",
    ]
    os.system("; ".join(cmds))

    for batch in batches:
        costs = []
        for i, shape in enumerate(small_conv_shapes_b1[beg:beg+num]):
            (_, C, H, W, K, _, R, S, _, stride,
                padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            try:
                cost = run(
                    N, C, H, W, K, R, S, stride,
                    padding, dilation,
                    i + beg + 1
                )
                costs.append(cost)
            except Exception as e:
                print_exc()
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
