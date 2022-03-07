import tvm
import os
from tvm import auto_tensorize as at
import argparse


def conv3d(
    N, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation, in_dtype, out_dtype
):
    pD = D + 2 * padding_d
    pH = H + 2 * padding
    pW = W + 2 * padding
    A = tvm.te.placeholder([N, C, D, H, W], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([K, C, KD, R, S], dtype=in_dtype, name="B")

    Pad = tvm.te.compute(
        [N, C, pD, pH, pW],
        lambda n, c, d, h, w: tvm.tir.if_then_else(
            tvm.tir.all(
                d >= padding_d,
                d - padding_d < D,
                h >= padding,
                h - padding < H,
                w >= padding,
                w - padding < W,
            ),
            A[n, c, d - padding_d, h - padding, w - padding],
            tvm.tir.const(0.0, A.dtype),
        ),
        name="Pad",
    )

    rc = tvm.te.reduce_axis([0, C], name="rc")
    rd = tvm.te.reduce_axis([0, KD], name="rd")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    outD = (pD - dilation * (KD - 1) - 1) // stride_d + 1
    P = (pH - dilation * (R - 1) - 1) // stride + 1
    Q = (pW - dilation * (S - 1) - 1) // stride + 1

    Conv = tvm.te.compute(
        [N, K, outD, P, Q],
        lambda n, k, d, p, q: tvm.te.sum(
            (
                Pad[
                    n, rc, d * stride_d + rd, p * stride + rr * dilation, q * stride + rs * dilation
                ]
                * B[k, rc, rd, rr, rs]
            ).astype(out_dtype),
            axis=[rc, rd, rr, rs],
        ),
        name="Conv",
    )
    return [A, B, Conv]


def mapping_tensorcore(
    N,
    C,
    D,
    H,
    W,
    K,
    KD,
    R,
    S,
    stride_d,
    stride,
    padding_d,
    padding,
    dilation,
    layer,
    in_dtype,
    out_dtype,
    simple_mode=True,
    trials=-1,
    verbose=False,
):
    A, B, Conv = conv3d(
        N,
        C,
        D,
        H,
        W,
        K,
        KD,
        R,
        S,
        stride_d,
        stride,
        padding_d,
        padding,
        dilation,
        in_dtype,
        out_dtype,
    )
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_dir = "conv3d-%s-%s-layer-%s" % (in_dtype, out_dtype, layer)
    log_file = "conv3d-%s-%s-layer-%s.log" % (in_dtype, out_dtype, layer)

    measure_opt = at.MeasureOptions(target=target, timeout=100, number=200, min_repeat_ms=500)

    if simple_mode:
        trials = 1000 if trials < 0 else trials
        result = at.auto_tensorize(
            target_dag, target, log_file, measure_opt, trials=trials, verbose=verbose
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
L = 8


#  (  N,   C,     L,   H,   W,   K,   D,   R,   S, stride_d, stride, padding_d, padding, dilation)
shapes_b1 = [
    (_, 3, L, 112, 112, 64, 1, 3, 3, 3, 7, 1, 3, 1),  # stem
    (_, 64, L, 56, 56, 64, 3, 3, 3, 1, 1, 1, 1, 1),  # layer1 x 4
    (_, 64, L, 56, 56, 128, 1, 1, 1, 2, 2, 0, 0, 1),  # layer2 downsample
    (_, 64, L, 56, 56, 128, 3, 3, 3, 2, 2, 1, 1, 1),  # layer2
    (_, 128, L // 2, 28, 28, 128, 3, 3, 3, 1, 1, 1, 1, 1),  # layer2 x 3
    (_, 128, L // 2, 28, 28, 256, 1, 1, 1, 2, 2, 0, 0, 1),  # layer3 downsample
    (_, 128, L // 2, 28, 28, 256, 3, 3, 3, 2, 2, 1, 1, 1),  # layer3
    (_, 256, L // 4, 14, 14, 256, 3, 3, 3, 1, 1, 1, 1, 1),  # layer3 x 3
    (_, 256, L // 4, 14, 14, 512, 1, 1, 1, 2, 2, 0, 0, 1),  # layer4 downsample
    (_, 256, L // 4, 14, 14, 512, 3, 3, 3, 2, 2, 1, 1, 1),  # layer4
    (_, 256, L // 8, 7, 7, 512, 3, 3, 3, 1, 1, 1, 1, 1),  # layer4 x 3
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
    python mapping_conv3d_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_conv3d_tensorcore.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_conv3d_tensorcore.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_conv3d_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 400 --simple_mode 0
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

    args = parser.parse_args()
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
            (_, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, D, H, W, K, KD, R, S, stride_d, stride, padding_d, padding, dilation)
            layer_name = f"({N},{C},{D},{H},{W},{K},{KD},{R},{S},{stride_d},{stride},{padding_d},{padding},{dilation})"
            try:
                cost = mapping_tensorcore(
                    N,
                    C,
                    D,
                    H,
                    W,
                    K,
                    KD,
                    R,
                    S,
                    stride_d,
                    stride,
                    padding_d,
                    padding,
                    dilation,
                    layer_name,
                    args.in_dtype,
                    args.out_dtype,
                    simple_mode=args.simple_mode,
                    trials=args.trials,
                    verbose=args.verbose,
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
