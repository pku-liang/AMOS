import tvm
import json
import numpy as np
import os
from tvm import auto_tensorize as at
from tvm.auto_tensorize import (
    CUDAParamsTenet,
    CUDAScheduleGeneratorTenet, CUDAScheduleApplierTenet)
import argparse


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float16", name="B")

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
                        ).astype("float16"), axis=[rc, rr, rs]),
        name="Conv"
    )
    # bias = tvm.te.placeholder([K], dtype="float32", name="bias")
    # E = tvm.te.compute(
    #     [N, K, P, Q],
    #     lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bk],
    #     name="E"
    # )
    return [A, B, Conv]


def tensorize_tensorcore_fp16fp16(
    N, C, H, W, K, R, S, stride,
    padding, dilation, layer
):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "conv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

    trials = 100
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)

    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, verbose=False, transform_dump=True)
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


def get_profile_data(
    N, C, H, W, K, R, S, stride,
    padding, dilation, layer
):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "conv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

    trials = 0
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)

    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, verbose=False, transform_dump=True)
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app

    records = []  # record, profile, model

    with open(log_file, "r") as fin:
        for i, line in enumerate(fin):
            obj = json.loads(line)
            robj = obj["record"]
            record = CUDAParamsTenet(
                robj["inline"],
                robj["vectorize"],
                robj["spatial_factors"],
                robj["reduce_factors"],
                robj["last_factors"],
                robj["output_unroll_step"],
                robj["last_unroll_step"])
            value = obj["value"]
            print("########################################\nNubmer:", i)
            cost = at.evaluate_params(schedule_app, record, measure_opt, dump=False) / 1e3
            records.append((record, 1/value, cost))

    with open("re-profile.log", "w") as fout:
        fout.write("number,profile(s),re-profile(s)\n")
        for i, (record, value, cost) in enumerate(records):
            print(f"{i},{value},{cost}", file=fout, flush=True)


def get_model_data(
    N, C, H, W, K, R, S, stride,
    padding, dilation, layer
):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "tenet cuda"

    log_file = "conv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

    trials = 0
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)

    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, verbose=False, transform_dump=True)
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app

    records = []  # record, profile, model

    with open(log_file, "r") as fin:
        for i, line in enumerate(fin):
            obj = json.loads(line)
            robj = obj["record"]
            record = CUDAParamsTenet(
                robj["inline"],
                robj["vectorize"],
                robj["spatial_factors"],
                robj["reduce_factors"],
                robj["last_factors"],
                robj["output_unroll_step"],
                robj["last_unroll_step"])
            value = obj["value"]
            print("########################################\nNubmer:", i)
            cost = at.evaluate_params(schedule_app, record, measure_opt, dump=True)
            records.append((record, 1/value, cost))

    with open("compare.log", "w") as fout:
        fout.write("number,profile(ms),model(G cycle)\n")
        for i, (record, value, cost) in enumerate(records):
            fout.write(f"{i},{value},{cost}\n")


def compare_pairwise():
    ground_truth = []
    model_predict = []
    num_data = 0
    with open("compare.log", "r") as fin:
        _ = fin.readline()
        for line in fin:
            _, g, m = line.split(",")
            ground_truth.append(float(g))
            model_predict.append(float(m))
            num_data += 1
    ground_matrix = [[0 for j in range(num_data)] for i in range(num_data)]
    model_matrix = [[0 for j in range(num_data)] for i in range(num_data)]
    for i in range(num_data):
        for j in range(num_data):
            if ground_truth[i] < ground_truth[j]:
                ground_matrix[i][j] = 1
            if model_predict[i] < model_predict[j]:
                model_matrix[i][j] = 1

    # top1 score
    top1 = 0
    for i in range(num_data):
        for j in range(num_data):
            if model_matrix[i][j] == ground_matrix[i][j]:
                top1 += 1

    print(f"Top1 accuracy: {top1/(num_data*num_data)}", flush=True)


def compare_pairwise_percentage():

    def helper(num):
        ground_truth = []
        model_predict = []
        num_data = 0
        with open("compare.log", "r") as fin:
            _ = fin.readline()
            for line in fin:
                _, g, m = line.split(",")
                ground_truth.append(float(g))
                model_predict.append(float(m))
                num_data += 1
                if num_data >= num:
                    break
        ground_matrix = [[0 for j in range(num_data)] for i in range(num_data)]
        model_matrix = [[0 for j in range(num_data)] for i in range(num_data)]
        for i in range(num_data):
            for j in range(num_data):
                if ground_truth[i] < ground_truth[j]:
                    ground_matrix[i][j] = 1
                if model_predict[i] < model_predict[j]:
                    model_matrix[i][j] = 1

        # top1 score
        top1 = 0
        for i in range(num_data):
            for j in range(num_data):
                if model_matrix[i][j] == ground_matrix[i][j]:
                    top1 += 1
        
        return top1/(num_data*num_data)

    resuts = []
    for i in range(1, 100):
        res = helper(i)
        resuts.append(res)

    with open("compare-pairwise-percentage.log", "w") as fout:
        for res in resuts:
            fout.write(str(res) + "\n")


def compare_recall_percentage():
    
    ground_truth = []
    model_predict = []
    num_data = 0
    with open("compare.log", "r") as fin:
        _ = fin.readline()
        for line in fin:
            _, g, m = line.split(",")
            ground_truth.append(float(g))
            model_predict.append(float(m))
            num_data += 1
        
    ground_truth = np.argsort(np.array(ground_truth))
    model_predict = np.argsort(np.array(model_predict))
        
    def helper(num):
        assert num > 0
        target = set(ground_truth[:num])
        answer = model_predict[:num]
        count = 0
        for ans in answer:
            if ans in target:
                count += 1
        return count / len(target)
        
    resuts = []
    for i in range(1, 100):
        res = helper(i)
        resuts.append(res)

    with open("compare-recall-percentage.log", "w") as fout:
        for res in resuts:
            fout.write(str(res) + "\n")


def run(N, C, H, W, K, R, S, stride,
        padding, dilation, layer):
    return tensorize_tensorcore_fp16fp16(
        N, C, H, W, K, R, S, stride,
        padding, dilation, layer)


def predict(N, C, H, W, K, R, S, stride,
        padding, dilation, layer):
    return get_model_data(
        N, C, H, W, K, R, S, stride,
        padding, dilation, layer)


def reprofile(N, C, H, W, K, R, S, stride,
        padding, dilation, layer):
    return get_profile_data(
        N, C, H, W, K, R, S, stride,
        padding, dilation, layer)  


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


conv3x3_shapes_b1 = [
    (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="profile")

    args = parser.parse_args()

    batches = [2**i for i in range(1)]
    beg = 6
    #num = 15
    num = 1
    for batch in batches:
        costs = []
        #for i, shape in enumerate(yolo_shapes_b1[beg:beg+num]):
        for i, shape in enumerate(res18_shapes_b1[beg:beg+num]):
            (_, C, H, W, K, _, R, S, _, stride,
                padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            # try:
            if args.mode == "profile":
                cost = run(
                    N, C, H, W, K, R, S, stride,
                    padding, dilation,
                    i + beg + 1
                )
            elif args.mode == "model":
                cost = predict(
                    N, C, H, W, K, R, S, stride,
                    padding, dilation,
                    i + beg + 1
                )
            elif args.mode == "reprofile":
                cost = reprofile(
                    N, C, H, W, K, R, S, stride,
                    padding, dilation,
                    i + beg + 1
                )
            elif args.mode == "compare":
                compare_pairwise()
                cost = 0
            elif args.mode == "compare-percentage":
                compare_pairwise_percentage()
                cost = 0
            elif args.mode == "compare-recall":
                compare_recall_percentage()
                cost = 0
            costs.append(cost)
            # except Exception as e:
            #     print("Fail to run\n", str(e))
            #     costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
