import tvm
import os
from tvm import auto_tensorize as at


def conv3d(N, C, D, H, W, K, KD, R, S, stride, padding, dilation):
    pD = D + 2 * padding
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, D, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, KD, R, S], dtype="float16", name="B")

    Pad = tvm.te.compute(
        [N, C, pD, pH, pW],
        lambda n, c, d, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                d >= padding, d - padding < D,
                h >= padding, h - padding < H,
                w >= padding, w - padding < W),
            A[n, c, d - padding, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype)
        ),
        name="Pad")

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rd = tvm.te.reduce_axis([0, KD], name="rd")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    assert dilation == 1 and stride == 1
    outD = (pD - dilation * (KD - 1) - 1) // stride + 1
    P = (pH - dilation * (R - 1) - 1) // stride + 1
    Q = (pW - dilation * (S - 1) - 1) // stride + 1

    Conv = tvm.te.compute(
        [N, K, outD, P, Q],
        lambda n, k, d, p, q:
            tvm.te.sum((Pad[n, rc, d+rd, p*stride+rr, q*stride+rs] * B[k, rc, rd, rr, rs]
                        ).astype("float16"), axis=[rc, rd, rr, rs]),
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
    N, C, D, H, W, K, KD, R, S, stride, padding, dilation, layer
):
    A, B, Conv = conv3d(N, C, D, H, W, K, KD, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "conv3d-fp16-layer-%d-batch-%d.log" % (layer, N)

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


def run(N, C, D, H, W, K, KD, R, S, stride, padding, dilation, layer):  
    return tensorize_tensorcore_fp16fp16(
        N, C, D, H, W, K, KD, R, S, stride, padding, dilation, layer)


conv3d_shapes = [
   # C,  D,   H,   W,  K, KD, R, S, stride, padding, dilation,
    ( 1, 3, 448, 448, 64,  3, 7, 7,      1,       2,        1),
    (12, 4, 128, 128, 64,  3, 3, 3,      1,       0,        1)
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = 2
    for batch in batches:
        costs = []
        for i, shape in enumerate(conv3d_shapes[beg:beg+num]):
            (C, D, H, W, K, KD, R, S, stride, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print("N, C, D, H, W, K, KD, R, S, stride, padding, dilation")
            print(N, C, D, H, W, K, KD, R, S, stride, padding, dilation)
            try:
                run(
                    N, C, D, H, W, K, KD, R, S, stride, padding, dilation, i + beg + 1
                )
            except Exception as e:
                print("Fail to run\n", str(e))
