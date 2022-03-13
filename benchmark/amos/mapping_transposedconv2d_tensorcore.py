import tvm
import os
from tvm import auto_tensorize as at
import argparse


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
            tvm.te.all(h % stride[0] == 0, w % stride[1] == 0),
            inputs[b, c, h // stride[0], w // stride[1]],
            expand_zero,
        ),
    )


def zero_pad2d(inputs, padding=0):
    padding = (
        (padding, padding, padding, padding)
        if isinstance(padding, (int, tvm.tir.IntImm))
        else padding
    )
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert len(padding) == 4

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.te.if_then_else(
            tvm.te.all(
                h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]
            ),
            inputs[b, c, h - padding[0], w - padding[2]],
            padding_zero,
        ),
        name="Padding",
    )


def transposed_conv2d(
    N, C, H, W, K, kernel_size, stride, padding, output_padding, dilation, in_dtype, out_dtype
):
    A = tvm.te.placeholder([N, K, H, W], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([K, C, kernel_size, kernel_size], dtype=in_dtype, name="B")
    batch_size, input_channel, in_h, in_w = A.shape
    input_channel_w, output_channel, k_h, k_w = B.shape

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    output_padding = (
        (output_padding, output_padding)
        if isinstance(output_padding, (int, tvm.tir.IntImm))
        else output_padding
    )
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(output_padding, tuple) and len(output_padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    kernel_h = (k_h - 1) * dilation[0] + 1
    kernel_w = (k_w - 1) * dilation[1] + 1
    out_h = (in_h - 1) * stride[0] - 2 * padding[0] + kernel_h + output_padding[0]
    out_w = (in_w - 1) * stride[1] - 2 * padding[1] + kernel_w + output_padding[1]
    rc = tvm.te.reduce_axis((0, input_channel), name="rc")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")

    expanded = zero_expand2d(A, stride=stride)
    padded = zero_pad2d(
        expanded,
        padding=(
            kernel_h - 1 - padding[0],
            kernel_h - 1 - padding[0] + output_padding[0],
            kernel_w - 1 - padding[1],
            kernel_w - 1 - padding[1] + output_padding[1],
        ),
    )
    output = tvm.te.compute(
        (batch_size, output_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (
                padded[b, rc, h + rh * dilation[0], w + rw * dilation[1]]
                * B[rc, c, k_h - rh - 1, k_w - rw - 1]
            ).astype(out_dtype),
            axis=[rc, rw, rh],
        ),
    )
    return [A, B, output]


def mapping_tensorcore(
    N,
    C,
    H,
    W,
    K,
    kernel_size,
    stride,
    padding,
    output_padding,
    dilation,
    layer,
    in_dtype,
    out_dtype,
    simple_mode=True,
    trials=-1,
    verbose=False,
    use_perf_model=False,
    perf_model_ratio=0.6,
):
    A, B, Conv = transposed_conv2d(
        N, C, H, W, K, kernel_size, stride, padding, output_padding, dilation, in_dtype, out_dtype
    )
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_dir = "transposed-conv2d-%s-%s-layer-%s" % (in_dtype, out_dtype, layer)
    log_file = "transposed-conv2d-%s-%s-layer-%s.log" % (in_dtype, out_dtype, layer)

    measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)

    if simple_mode:
        trials = 1000 if trials < 0 else trials
        result = at.auto_tensorize(
            target_dag,
            target,
            log_file,
            measure_opt,
            trials=trials,
            verbose=verbose,
            transform_strict=False,
        )
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
    else:
        trials = 4000 if trials < 0 else trials
        result = at.auto_tensorize_v4(
            target_dag,
            target,
            log_file,
            measure_opt,
            schedule_log_dir=log_dir,
            trials=trials,
            search_group_size=5,
            transform_dump=verbose,
            transform_strict=False,
            enable_perf_model=use_perf_model,
            perf_percentage=perf_model_ratio,
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

    cost = at.evaluate_params(schedule_app, params, measure_opt, dump=verbose)
    print("Cost of %s is %f ms" % (log_dir, cost))
    return cost


_ = None
#  (  N,   C,   H,   W,   K, kernel_size, stride, padding, dilation)
shapes_b1 = [
    (_, 3, 112, 112, 64, 3, 7, 3, 1),  # stem
    (_, 64, 56, 56, 64, 3, 1, 1, 1),  # layer1 x 4
    (_, 64, 56, 56, 128, 1, 2, 0, 1),  # layer2 downsample
    (_, 64, 56, 56, 128, 3, 2, 1, 1),  # layer2
    (_, 128, 28, 28, 128, 3, 1, 1, 1),  # layer2 x 3
    (_, 128, 28, 28, 256, 1, 2, 0, 1),  # layer3 downsample
    (_, 128, 28, 28, 256, 3, 2, 1, 1),  # layer3
    (_, 256, 14, 14, 256, 3, 1, 1, 1),  # layer3 x 3
    (_, 256, 14, 14, 512, 1, 2, 0, 1),  # layer4 downsample
    (_, 256, 14, 14, 512, 3, 2, 1, 1),  # layer4
    (_, 256, 7, 7, 512, 3, 1, 1, 1),  # layer4 x 3
]

supported_dtypes = set(
    [
        ("float16", "float16"),
        ("float16", "float32"),
        ("bfloat16", "float32"),
        ("float32", "float32"),
        ("float64", "float64"),
        ("int4", "int32"),
        ("int8", "int32"),
    ]
)

example_text = """
 example:
    python mapping_transposedconv2d_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_transposedconv2d_tensorcore.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_transposedconv2d_tensorcore.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_transposedconv2d_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 400 --simple_mode 0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--in_dtype",
        type=str,
        choices=["float16", "float32", "float64", "bfloat16", "int4", "int8"],
        default="float16",
    )
    parser.add_argument(
        "--out_dtype",
        type=str,
        choices=["float16", "float32", "float64", "int32"],
        default="float16",
    )
    parser.add_argument("--begin", type=int, choices=list(range(len(shapes_b1))), default=0)
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes_b1) + 1)), default=len(shapes_b1)
    )
    parser.add_argument("--simple_mode", type=int, default=1, choices=[0, 1])
    parser.add_argument("--trials", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_perf_model", action="store_true")
    parser.add_argument("--perf_model_ratio", type=float, default=0.6)

    args = parser.parse_args()
    assert 0 < args.perf_model_ratio <= 1.0
    if args.use_perf_model:
        assert args.simple_mode == 0, "Performance model is only supported without simple_mode"
    batches = [args.batch]
    beg = args.begin
    num = args.num
    print(args.simple_mode)
    assert (
        args.in_dtype,
        args.out_dtype,
    ) in supported_dtypes, (
        f"The desired dtype pair {(args.in_dtype, args.out_dtype)} is not supported by Tensor Core."
    )
    for batch in batches:
        costs = []
        for i, shape in enumerate(shapes_b1[beg : beg + num]):
            (N, C, H, W, K, kernel_size, stride, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, kernel_size, stride, padding, dilation)
            layer_name = f"({N},{C},{H},{W},{K},{kernel_size},{stride},{padding},{dilation})"
            try:
                cost = mapping_tensorcore(
                    N,
                    C,
                    H,
                    W,
                    K,
                    kernel_size,
                    stride,
                    padding,
                    0,
                    dilation,
                    layer_name,
                    args.in_dtype,
                    args.out_dtype,
                    simple_mode=args.simple_mode,
                    trials=args.trials,
                    verbose=args.verbose,
                    use_perf_model=args.use_perf_model,
                    perf_model_ratio=args.perf_model_ratio,
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
