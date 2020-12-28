import tvm
import os
from tvm import auto_tensorize as at

def zero_expand2d(inputs, stride=1):
    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    assert isinstance(stride, tuple), "type(stride)={}".format(type(stride))
    assert len(stride) == 2

    expand_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    out_height = (height - 1) * stride[0] + 1
    out_width = (width - 1) * stride[1] + 1
    return tvm.te.compute(
        (batch_size, in_channel, out_height, out_width),
        lambda b, c, h, w: tvm.te.if_then_else(
                            tvm.te.all(
                                h % stride[0] == 0,
                                w % stride[1] == 0
                                ),
                            inputs[b, c, h // stride[0], w // stride[1]],
                            expand_zero
                            )
        )


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


def conv_transpose2d_nchw(inputs, weight, stride=1, padding=0, output_padding=0, dilation=1):
    batch_size, input_channel, in_h, in_w = inputs.shape
    input_channel_w, output_channel, k_h, k_w = weight.shape
    assert input_channel.value == input_channel_w.value

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    output_padding = ((output_padding, output_padding) 
                        if isinstance(output_padding, (int, tvm.tir.IntImm)) else output_padding)
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(output_padding, tuple) and len(output_padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    kernel_h = (k_h - 1) * dilation[0] + 1
    kernel_w = (k_w - 1) * dilation[1] + 1
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    rc = tvm.te.reduce_axis((0, input_channel))
    rh = tvm.te.reduce_axis((0, k_h))
    rw = tvm.te.reduce_axis((0, k_w))

    expanded = zero_expand2d(inputs, stride=stride)
    padded = zero_pad2d(expanded, padding=(
                                    kernel_h - 1 - padding[0], 
                                    kernel_h - 1 - padding[0] + output_padding[0],
                                    kernel_w - 1 - padding[1],
                                    kernel_w - 1 - padding[1] + output_padding[1]))
    output = tvm.te.compute(
        (batch_size, output_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (padded[b, rc, h + rh * dilation[0], w + rw * dilation[1]] * 
            weight[rc, c, k_h - rh - 1, k_w - rw - 1]),
            axis=[rc, rw, rh])
    )
    return output



def tensorize_tensorcore_fp16fp16(
    H, W, N, C, K, kernel_size, stride, padding, layer
):
    A = tvm.te.placeholder([N, K, H, W], dtype="float16", name="A")
    B = tvm.te.placeholder([K, C, kernel_size, kernel_size], dtype="float16", name="B")
    Conv_transpose = conv_transpose2d_nchw(A, B, stride=stride, padding=padding)
    target_dag = at.compute_dag_from_tensors([Conv_transpose])
    target = "cuda"

    log_file = "transpose-conv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

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


def run(H, W, N, C, K, kernel_size, stride, padding, layer):
    return tensorize_tensorcore_fp16fp16(
        H, W, N, C, K, kernel_size, stride, padding, layer)


_ = None
#  (  N,   C,   H,   W,   K, kernel_size, stride, padding, dilation)
transpose2d_config = [
    ( _,   3, 112, 112,  64,           3,      7,       3,        1), # stem

    ( _,  64,  56,  56,  64,           3,      1,       1,        1), # layer1 x 4

    ( _,  64,  56,  56, 128,           1,      2,       0,        1), # layer2 downsample
    
    ( _,  64,  56,  56, 128,           3,      2,       1,        1), # layer2
    ( _, 128,  28,  28, 128,           3,      1,       1,        1), # layer2 x 3

    ( _, 128,  28,  28, 256,           1,      2,       0,        1), # layer3 downsample
    ( _, 128,  28,  28, 256,           3,      2,       1,        1), # layer3
    ( _, 256,  14,  14, 256,           3,      1,       1,        1), # layer3 x 3

    ( _, 256,  14,  14, 512,           1,      2,       0,        1), # layer4 downsample
    ( _, 256,  14,  14, 512,           3,      2,       1,        1), # layer4
    ( _, 256,   7,   7, 512,           3,      1,       1,        1), # layer4 x 3
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    num = len(transpose2d_config)
    for batch in batches:
        costs = []
        #for i, shape in enumerate(yolo_shapes_b1[beg:beg+num]):
        for i, shape in enumerate(transpose2d_config[beg:beg+num]):
            (  N,   C,   H,   W,   K, kernel_size, stride, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print(N,   C,   H,   W,   K, kernel_size, stride, padding, dilation)
            iH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            iW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
            try:
                cost = run(
                    iH, iW, N, C, K, kernel_size, stride, padding,
                    i + beg + 1
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
