import tvm
import os
from tvm import auto_tensorize as at


def zero_pad1d(inputs, padding=0):
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    assert len(padding) == 2

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, in_len = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, in_len + padding[0] + padding[1]),
        lambda b, c, l: tvm.te.if_then_else(
                            tvm.te.all(l >= padding[0], l < in_len + padding[0]),
                            inputs[b, c, l - padding[0]],
                            padding_zero
                            )
        )


def conv1d(inputs, weight, stride=1, padding=0, dilation=1):
    batch_size, in_channel, in_len = inputs.shape
    out_channel, channel_per_group, k_len = weight.shape
    assert channel_per_group.value == in_channel.value

    stride = stride[0] if isinstance(stride, tuple) else stride
    padding = padding[0] if isinstance(padding, tuple) else padding
    assert isinstance(stride, (int, tvm.tir.IntImm)), "type(stride)={}".format(type(stride))
    assert isinstance(padding, (int, tvm.tir.IntImm)), "type(padding)={}".format(type(padding))
    assert isinstance(dilation, (int, tvm.tir.IntImm)), "type(dilation)={}".format(type(dilation))

    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1

    rc = tvm.te.reduce_axis((0, channel_per_group))
    rl = tvm.te.reduce_axis((0, k_len))

    padded = zero_pad1d(inputs, padding=padding)
    conved = tvm.te.compute(
        (batch_size, out_channel, out_len),
        lambda b, c, l: tvm.te.sum(
            (padded[b, rc, l * stride + rl * dilation] * 
            weight[c, rc, rl]), 
            axis=[rc, rl]
            )
    )
    return conved


def tensorize_tensorcore_fp16fp16(
    N, C, L, K, KL, stride, padding, dilation, layer
):
    A = tvm.te.placeholder([N, C, L], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, KL], dtype="float16", name="B")

    Conv = conv1d(A, B, stride=stride, padding=padding, dilation=dilation)
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


byte_net_shapes = [
  # (   C,   L,   K,  KL, stride,padding, dilation)
    ( 512, 892, 512,   3,       1,     2,        1),
    ( 512, 892,1024,   1,       1,     0,        1),
    (1024, 892, 512,   1,       1,     0,        1),
    ( 512, 892, 512,   3,       1,     4,        2),
    ( 512, 892, 512,   3,       1,     8,        4),
    ( 512, 892, 512,   3,       1,    16,        8),
    ( 512, 892, 512,   3,       1,    32,       16),
    (1024, 892, 250,   1,       1,     0,        1)
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = len(byte_net_shapes)
    for batch in batches:
        costs = []
        for i, shape in enumerate(byte_net_shapes[beg:beg+num]):
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
