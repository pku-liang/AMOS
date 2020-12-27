import tvm
import os
from tvm import auto_tensorize as at

def zero_pad2d(inputs, padding=0):
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert len(padding) == 4

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.te.if_then_else(
                            tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c, h - padding[0], w - padding[2]],
                            padding_zero
                            ),
        name='Padding'
        )

def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert (channel_per_group * groups).value == in_channel.value
    out_channel_per_group = out_channel // groups
    assert (out_channel_per_group * groups).value == out_channel.value

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")

    padded = zero_pad2d(inputs, padding=padding)
    output = tvm.te.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
            * weight[c, rc, rh, rw]),
            axis=[rc, rw, rh]
        )
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
        )
    return output

def concat_eight_vector_lastdim(cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7):
    batch, channel, h, w = cap0.shape
    eight = 8
 
    return tvm.te.compute([batch, channel, h, w, eight],
            lambda i, j, p, q, k:
                tvm.te.if_then_else(k == 0, cap0[i, j, p, q],
                    tvm.te.if_then_else(k == 1, cap1[i, j, p, q],
                        tvm.te.if_then_else(k == 2, cap2[i, j, p, q],
                            tvm.te.if_then_else(k == 3, cap3[i, j, p, q],
                                tvm.te.if_then_else(k == 4, cap4[i, j, p, q],
                                    tvm.te.if_then_else(k == 5, cap5[i, j, p, q], 
                                        tvm.te.if_then_else(k == 6, cap6[i, j, p, q],
                                            cap7[i, j, p, q]))))))),
            name="concat")

def capsule(H, W, N, C, outC, kernel_size, stride, padding, capsule_num):
    assert capsule_num == 8
    A_lst = []
    B_lst = []
    cap_lst = []
    for i in range(capsule_num):
        A_lst.append(tvm.te.placeholder([N, C, H, W], dtype="float16"))
        B_lst.append(tvm.te.placeholder([outC, C, kernel_size, kernel_size], dtype="float16"))
    for i in range(capsule_num):
        cap_lst.append(conv2d_nchw(A_lst[i], B_lst[i], bias=None, stride=stride, padding=padding))
    CapsuleConv = concat_eight_vector_lastdim(*cap_lst)
    return CapsuleConv


def tensorize_tensorcore_fp16fp16(
    H, W, N, C, outC, kernel_size, stride, padding, capsule_num, layer
):
    CapsuleConv = capsule(H, W, N, C, outC, kernel_size, stride, padding, capsule_num)
    target_dag = at.compute_dag_from_tensors([CapsuleConv])
    target = "cuda"

    log_file = "capsule-fp16-layer-%d-batch-%d.log" % (layer, N)

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


def run(H, W, N, C, outC, kernel_size, stride, padding, capsule_num, layer):
    return tensorize_tensorcore_fp16fp16(
        H, W, N, C, outC, kernel_size, stride, padding, capsule_num, layer)

capsule_config = [
    #H,  W,   C, outC, kernel_size, stride, padding, capsule_num
    (20, 20, 256,   32,          9,      2,       0,          8)
]

if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = len(capsule_config)
    for batch in batches:
        costs = []
        for i, shape in enumerate(capsule_config[beg:beg+num]):
            (H, W, C, outC, kernel_size, stride, padding, capsule_num) = shape
            assert capsule_num == 8
            N = batch
            print("\n\nProblem size:")
            print("H, W, N, C, outC, kernel_size, stride, padding, capsule_num,")
            print(H, W, N, C, outC, kernel_size, stride, padding, capsule_num)
            try:
                cost = run(
                    H, W, N, C, outC, kernel_size, stride, padding, capsule_num,
                    i + beg + 1
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
