import tvm
import os
from tvm import auto_tensorize as at
from tvm.topi.bifrost.conv2d import _decl_winograd
from itertools import product
from traceback import print_exc
from contextlib import redirect_stderr

"""In this tutorial, we fix recipe, hand-craft match points,
    and fix transform decisions, to see how parameters affects performance
"""


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    A = tvm.te.placeholder([N, C, H, W], dtype="float32", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float32", name="B")

    with open("/dev/null", "w") as fp:
        with redirect_stderr(fp):
            Conv = _decl_winograd(tvm.autotvm.get_config(), A, B,
                                  stride, padding, dilation, "float32")
    return [A, B, Conv]


def tune_general_winograd(
    N, C, H, W, K, R, S, stride,
    padding, dilation, layer
):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])

    target = "opencl"
    target_host = 'llvm -mtriple=aarch64-linux-android'
    dev_key = "android"
    host = "0.0.0.0"
    port = 9190

    log_file = f"opencl-winograd-{layer}.log"

    # prepare schedulers
    schedule_gen = at.MaliGeneralScheduleGenerator(
        target_dag, log_file=log_file)
    if os.path.exists(log_file) and os.path.isfile(log_file):
        schedule_gen.load_from_file(log_file)
    schedule_app = at.MaliGeneralScheduleApplier(target_dag)
    trials = 0
    measure_opt = at.MeasureOptions(
        target=target, target_host=target_host,
        build_func="ndk", use_rpc=True, host=host, port=port,
        timeout=10, number=2, key=dev_key)
    checker = at.MaliProgramChecker(arch="g76")

    # use tuning to find params
    value, params = at.find_optimized_parameters(
        [], schedule_gen, schedule_app,
        measure_opt, checker, trials,  # policy="random",
        builder=at.pebble_local_builder_build,
        runner=at.pebble_rpc_runner_run)

    # load from file
    schedule_gen.clear("")
    schedule_gen.load_from_file(log_file)
    entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in file
    params, value = entry.record, 1 / entry.value
    print(value)
    print(params.to_json())

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)


def run(N, C, H, W, K, R, S, stride,
        padding, dilation, layer):
    tune_general_winograd(
        N, C, H, W, K, R, S, stride,
        padding, dilation, layer)


yolo_shapes_b1 = [
    # yolo
    (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
]


if __name__ == "__main__":
    cmds = [
        "adb reverse tcp:9190 tcp:9190",
        "adb forward tcp:5001 tcp:5001",
        "adb shell am start -n org.apache.tvm.tvmrpc/org.apache.tvm.tvmrpc.MainActivity 1> /dev/null 2> /dev/null",
    ]
    os.system("; ".join(cmds))

    batches = [2**i for i in range(1)]
    beg = 0
    num = 1
    for batch in batches:
        for i, shape in enumerate(yolo_shapes_b1[beg:beg+num]):
            (_, C, H, W, K, _, R, S, _, stride,
                padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            try:
                run(
                    N, C, H, W, K, R, S, stride,
                    padding, dilation,
                    i + beg + 1
                )
            except Exception as e:
                print_exc()
                print("Fail to run\n", str(e))
