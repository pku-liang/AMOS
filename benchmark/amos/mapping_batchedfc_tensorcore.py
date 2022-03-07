import tvm
import os
from tvm import auto_tensorize as at
import argparse


def batched_fc(N, I, O, groups, in_dtype, out_dtype):
    channel_per_group = I // groups
    out_channel_per_group = O // groups

    A = tvm.te.placeholder([N, I], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([O, channel_per_group], dtype=in_dtype, name="B")

    A_reshaped = tvm.te.compute(
        [N, groups, channel_per_group], lambda n, c_o, c_i: A[n, c_o * channel_per_group + c_i]
    )

    B_reshaped = tvm.te.compute(
        [groups, out_channel_per_group, channel_per_group],
        lambda k_o, k_i, c: B[k_o * out_channel_per_group + k_i, c],
    )

    rc = tvm.te.reduce_axis([0, channel_per_group], name="rc")

    BFC = tvm.te.compute(
        [N, groups, out_channel_per_group],
        lambda n, k_o, k_i: tvm.te.sum(
            (A_reshaped[n, k_o, rc] * B_reshaped[k_o, k_i, rc]).astype(out_dtype),
            axis=[
                rc,
            ],
        ),
        name="BatchedFC",
    )

    return [A, B, BFC]


def mapping_tensorcore(
    N,
    I,
    O,
    groups,
    layer,
    in_dtype,
    out_dtype,
    simple_mode=True,
    trials=-1,
    verbose=False,
):
    A, B, Conv = batched_fc(N, I, O, groups, in_dtype, out_dtype)
    target_dag = at.compute_dag_from_tensors([Conv])
    target = "cuda"

    log_dir = "batched-fc-%s-%s-layer-%s" % (in_dtype, out_dtype, layer)
    log_file = "batched-fc-%s-%s-layer-%s.log" % (in_dtype, out_dtype, layer)

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


shapes_b1 = [
    # https://github.com/megvii-model/WeightNet/blob/master/shufflenet_v2.py
    # in_channels, out_channels, groups (ksize, stride, padding = 1, 1, 0)
    # shuffle_v2_cfg
    (24, 216, 24),
    (48, 576, 48),
    (56, 504, 56),
    (112, 1008, 112),
    (112, 1344, 112),
    (112, 3136, 112),
    (176, 4928, 176),
    (224, 2016, 224),
    (224, 12544, 224),
    (448, 50176, 448),
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
    python mapping_batchedfc_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_batchedfc_tensorcore.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_batchedfc_tensorcore.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_batchedfc_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 400 --simple_mode 0
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
            I, O, groups = shape
            N = batch
            print("\n\nProblem size:")
            print(N, I, O, groups)
            layer_name = f"({N},{I},{O},{groups})"
            try:
                cost = mapping_tensorcore(
                    N,
                    I,
                    O,
                    groups,
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
