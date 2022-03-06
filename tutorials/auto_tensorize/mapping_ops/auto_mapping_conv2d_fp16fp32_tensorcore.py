import tvm
import os
from tvm import auto_tensorize as at
import numpy as np


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


def tensorize_tensorcore_fp16fp32(N, C, H, W, K, R, S, stride, padding, dilation, layer):
    A, B, Conv = conv2d(N, C, H, W, K, R, S, stride, padding, dilation)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_file = "conv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

    # Set the trals in tuning
    # 1000 is a good choice
    # We use 20 just for a quick tutorial
    # If you have already tuned and gotten a log file
    # You can set it as 0 to bypass tuning
    trials = 20
    measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)

    result = at.auto_tensorize(
        target_dag,
        target,
        log_file,
        measure_opt,
        trials=trials,
        verbose=False,
        transform_dump=False,
        # you can choose a specific mapping by pointing out its id
        transform_policy="choose:0,1",
    )
    if not result.defined():
        print("Can't do tensorize.")
        return
    schedule_gen = result.sch_gen
    schedule_app = result.sch_app

    # load records from log file
    schedule_gen.load_from_file(log_file, clear=True)
    # get the best one
    entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in log file
    params, value = entry.record, 1 / entry.value
    print(value)
    print(params.to_json())

    # evalute the record
    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=False)
    print("Cost is %f ms" % cost)

    # retrieve schedule from the record
    target_dag = schedule_app.target_dag
    inputs = target_dag.get_inputs()
    args = inputs + list(target_dag.tensors)
    sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    sch = schedule_app.apply(sch, params)
    func = tvm.build(sch, args, target)

    # test correctness
    A, B = inputs
    (Conv,) = target_dag.tensors
    A_np = np.random.uniform(-10, 10, [int(x) for x in A.shape]).astype(A.dtype)
    B_np = np.random.uniform(-10, 10, [int(x) for x in B.shape]).astype(B.dtype)
    Conv_np = np.random.uniform(-10, 10, [int(x) for x in Conv.shape]).astype(Conv.dtype)

    # use scipy convolve2d api
    from tvm.topi.testing import conv2d_nchw_python

    Conv_golden = conv2d_nchw_python(A_np, B_np, stride, padding)

    ctx = tvm.context(target, 0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    Conv_tvm = tvm.nd.array(Conv_np, ctx)
    func(A_tvm, B_tvm, Conv_tvm)

    from tvm import testing

    testing.assert_allclose(Conv_golden, Conv_tvm.asnumpy(), atol=1e-2, rtol=1e-2)
    print("Correctness check passed!")
    return cost


def run(N, C, H, W, K, R, S, stride, padding, dilation, layer):
    return tensorize_tensorcore_fp16fp32(N, C, H, W, K, R, S, stride, padding, dilation, layer)


if __name__ == "__main__":
    batch = 1
    layer_id = 0
    N, H, W, K, C, R, S, stride, padding, dilation = batch, 28, 28, 128, 128, 3, 3, 1, 1, 1
    print("Problem size:")
    print(N, C, H, W, K, R, S, stride, padding)
    cost = run(N, C, H, W, K, R, S, stride, padding, dilation, layer_id)
