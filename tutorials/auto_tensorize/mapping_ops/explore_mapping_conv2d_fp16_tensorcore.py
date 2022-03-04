import tvm
import os
from tvm import auto_tensorize as at


def conv2d(N, C, H, W, K, R, S, stride, padding, dilation):
    kH = (R - 1) * dilation + 1
    kW = (S - 1) * dilation + 1
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype="float16", name="B")

    Pad = tvm.te.compute(
        [N, C, pH, pW],
        lambda n, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= padding, h - padding < H, w >= padding, w - padding < W),
            A[n, c, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype),
        ),
        name="Pad",
    )

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, kH], name="rr")
    rs = tvm.te.reduce_axis([0, kW], name="rs")

    P = (pH - kH) // stride + 1
    Q = (pW - kW) // stride + 1
    Conv = tvm.te.compute(
        [N, K, P, Q],
        lambda n, k, p, q: tvm.te.sum(
            (Pad[n, rc, p * stride + rr, q * stride + rs] * B[k, rc, rr, rs]).astype("float16"),
            axis=[rc, rr, rs],
        ),
        name="Conv",
    )
    return [A, B, Conv]


def tensorize_tensorcore_fp16fp16(N, C, H, W, K, R, S, stride, padding, dilation, layer):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "conv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

    trials = 0
    measure_opt = at.MeasureOptions(target=target, timeout=10, number=200, min_repeat_ms=500)

    result = at.auto_tensorize_v4(
        target_dag,
        target,
        log_file,
        measure_opt,
        trials=trials,
        search_group_size=20,
        transform_dump=False,
    )
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app

    # we store 1/time_cost in file
    params, value = result.params, result.perf
    print(value)
    print(params.to_json())

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)
    return cost


def run(N, C, H, W, K, R, S, stride, padding, dilation, layer):
    return tensorize_tensorcore_fp16fp16(N, C, H, W, K, R, S, stride, padding, dilation, layer)


if __name__ == "__main__":
    batch = 1
    layer_id = 0
    N, H, W, K, C, R, S, stride, padding, dilation = batch, 28, 28, 128, 128, 3, 3, 1, 1, 1
    print("Problem size:")
    print(N, C, H, W, K, R, S, stride, padding)
    cost = run(N, C, H, W, K, R, S, stride, padding, dilation, layer_id)
