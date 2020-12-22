import tvm
import os
from tvm import auto_tensorize as at


def conv1d(N, C, L, K, KL, stride, padding, dilation):
    assert dilation == 1 and stride == 1
    pL = L + 2 * padding
    A = tvm.te.placeholder([N, C, L], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, KL], dtype="float16", name="B")

    Pad = tvm.te.compute(
        [N, C, pL],
        lambda n, c, l: tvm.tir.if_then_else(
            tvm.tir.all(l >= padding, l - padding < L),
            A[n, c, l - padding],
            tvm.tir.const(0.0, A.dtype)
        ),
        name="Pad")

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, KL], name="rr")

    outL = (pL - KL) // stride + 1
    Conv = tvm.te.compute(
            [N, K, outL],
            lambda n, k, l:
                tvm.te.sum((Pad[n, rc, l*stride+rr] * B[k, rc, rr]).astype("float16"), axis=[rc, rr]),
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
    N, C, L, K, KL, stride, padding, dilation, layer
):
    A, B, Conv = conv1d(N, C, L, K, KL, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "conv1d-fp16-layer-%d-batch-%d.log" % (layer, N)

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


def run(N, C, L, K, KL, stride, padding, dilation, layer):
    return tensorize_tensorcore_fp16fp16(
       N, C, L, K, KL, stride, padding, dilation, layer)


conv1d_shapes = [
    # C,  L,  K, KL, stride, padding, dilation
    (16, 16, 32, 3,      1,        1,        1),
    (32, 32, 64, 5,      1,        0,        1)
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = 15
    for batch in batches:
        costs = []
        for i, shape in enumerate(conv1d_shapes[beg:beg+num]):
            (C, L, K, KL, stride, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print("N, C, L, K, KL, stride, padding, dilation")
            print(N, C, L, K, KL, stride, padding, dilation)
            try:
                cost = run(
                    N, C, L, K, KL, stride, padding, dilation,
                    i + beg + 1
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
