import tvm
import os
from tvm import auto_tensorize as at


def variance(N, C, H, W):
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")

    Mean = tvm.te.placeholder([N, C, H, W], dtype="float16", name="Mean")

    Diff = tvm.te.compute([N, C, H, W], lambda n, c, h, w: (A[n, c, h, w] - Mean[n, c, h, w]) * (A[n, c, h, w] - Mean[n, c, h, w]), name="Diff")

    B = tvm.te.compute([C, N * H * W], lambda  i, j: Diff[j//(H*W), i, j%(H*W)//W, j%W], name="B")

    C_ = tvm.te.compute([N*H*W, 1], lambda i, j: tvm.tir.const(1.0/(N*H*W), "float16"), name="C")

    rk = tvm.te.reduce_axis([0, N*H*W], name="k")
    D = tvm.te.compute([C, 1], lambda i, j: tvm.te.sum((B[i, rk] * C_[rk, j]).astype("float16"), axis=rk), name="D")
    E = tvm.te.compute([C], lambda i: D[i, 0], name="E")
    return [A, B, E]


def tensorize_tensorcore_fp16fp16(
    N, C, H, W, layer
):
    A, B, E = variance(N, C, H, W)
    target_dag = at.compute_dag_from_tensors([E])
    target = "cuda"

    log_file = "mean-fp16-layer-%d.log" % (layer)

    trials = 1000
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)

    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, verbose=True, transform_dump=True)
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app

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


def run(N, C, H, W, layer):
    return tensorize_tensorcore_fp16fp16(
        N, C, H, W, layer)


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
    beg = 0
    num = 5
    costs = []
    for i, shape in enumerate(res18_shapes_b1[beg:beg+num]):
        (N, C, H, W, _, _, _, _, _, _, _, _, _) = shape
        print("\n\nProblem size:")
        print(N, C, H, W)
        # try:
        cost = run(
            N, C, H, W,
            i + beg + 1
        )
        costs.append(cost)
        # except Exception as e:
        #     print("Fail to run\n", str(e))
        #     costs.append(float("inf"))

    for cost in costs:
        print(cost)
