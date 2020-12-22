import tvm
import os
from tvm import auto_tensorize as at


def grouped_conv2d(N, C, H, W, K, R, S, stride, padding, dilation, groups):
    assert(K % groups == 0)
    assert(C % groups == 0)
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C//groups, R, S], dtype="float16", name="B")

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
    
    A_reshaped = tvm.te.compute(
        [N, groups, C//groups, pH, pW],
        lambda n, c_o, c_i, r, s: Pad[n, c_o * (C//groups) + c_i, r, s]
    )

    B_reshaped = tvm.te.compute(
        [groups, K//groups, C//groups, R, S],
        lambda k_o, k_i, c, r, s: B[k_o * (K//groups) + k_i, c, r, s]
    )

    rc = tvm.te.reduce_axis([0, C//groups], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (pH - R) // stride + 1
    Q = (pW - S) // stride + 1
    Conv = tvm.te.compute(
        [N, groups, K//groups, P, Q],
        lambda n, k_o, k_i, p, q:
            tvm.te.sum((A_reshaped[n, k_o, rc, p*stride+rr, q*stride+rs] * B_reshaped[k_o, k_i, rc, rr, rs]
                        ).astype("float16"), axis=[rc, rr, rs]),
        name="Conv"
    )

    Conv_reshaped = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q:
            Conv[n, k//(K//groups), k%(K//groups), p, q],
        name="Reshaped"
    )

    # bias = tvm.te.placeholder([K], dtype="float32", name="bias")
    # E = tvm.te.compute(
    #     [N, K, P, Q],
    #     lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bk],
    #     name="E"
    # )
    return [A, B, Conv_reshaped]


def tensorize_tensorcore_fp16fp16(
    N, C, H, W, K, R, S, stride,
    padding, dilation, groups, layer
):
    A, B, Conv = grouped_conv2d(N, C, H, W, K, R, S, stride, padding, dilation, groups)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "grouped-conv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

    trials = 1000
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)

    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, verbose=True)
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


def run(N, C, H, W, K, R, S, stride,
        padding, dilation, groups, layer):
    return tensorize_tensorcore_fp16fp16(
        N, C, H, W, K, R, S, stride,
        padding, dilation, groups, layer)


 # (N, C, H, W, K, _, R, S, _, stride, padding, dilation, groups, _)
 # group conv from AlexNet
alex_shapes = [
    (1, 96, 27, 27, 256, 96, 5, 5, 1, 1, 2, 1, 2, 1),
    (1, 384, 13, 13, 384, 384, 3, 3, 1, 1, 1, 1, 2, 1),
    (1, 384, 13, 13, 256, 384, 3, 3, 1, 1, 1, 1, 2, 1)
]

if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = 3
    for batch in batches:
        costs = []
        for i, shape in enumerate(alex_shapes):
            (_, C, H, W, K, _, R, S, _, stride,
                padding, dilation, groups, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding, groups)
            try:
                cost = run(
                    N, C, H, W, K, R, S, stride,
                    padding, dilation, groups,
                    i
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
