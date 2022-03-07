import tvm
import os
from tvm import auto_tensorize as at
import argparse


def gemm(M, N, K, in_dtype, out_dtype):
    A = tvm.te.placeholder([M, K], dtype=in_dtype, name="A")
    B = tvm.te.placeholder([K, N], dtype=in_dtype, name="B")

    rk = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute(
        [M, N], lambda i, j: tvm.te.sum((A[i, rk] * B[rk, j]).astype(out_dtype), axis=rk), name="C"
    )
    return [A, B, C]


def mapping_tensorcore(
    M,
    N,
    K,
    layer,
    in_dtype,
    out_dtype,
    simple_mode=True,
    trials=-1,
    verbose=False,
):
    A, B, Gemm = gemm(M, N, K, in_dtype, out_dtype)
    target_dag = at.compute_dag_from_tensors([Gemm])
    target = "cuda"

    log_dir = "gemm-%s-%s-layer-%s" % (in_dtype, out_dtype, layer)
    log_file = "gemm-%s-%s-layer-%s.log" % (in_dtype, out_dtype, layer)

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


shapes = [(16, 512, 128), (1024, 16, 256), (256, 1024, 256), (512, 256, 16), (1024, 1024, 1024)]

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
    python mapping_gemm_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 20
    python mapping_gemm_tensorcore.py --in_dtype float16 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_gemm_tensorcore.py --in_dtype float32 --out_dtype float32 --begin 0 --num 1 --trials 20
    python mapping_gemm_tensorcore.py --in_dtype float16 --out_dtype float16 --begin 0 --num 1 --trials 400 --simple_mode 0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    parser.add_argument("--begin", type=int, choices=list(range(len(shapes))), default=0)
    parser.add_argument(
        "--num", type=int, choices=list(range(1, len(shapes) + 1)), default=len(shapes)
    )
    parser.add_argument("--simple_mode", type=int, default=1, choices=[0, 1])
    parser.add_argument("--trials", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    beg = args.begin
    num = args.num
    print(args.simple_mode)
    assert (
        args.in_dtype,
        args.out_dtype,
    ) in supported_dtypes, (
        f"The desired dtype pair {(args.in_dtype, args.out_dtype)} is not supported by Tensor Core."
    )
    costs = []
    for i, shape in enumerate(shapes[beg : beg + num]):
        (M, N, K) = shape
        print("\n\nProblem size:")
        print(M, N, K)
        layer_name = f"({M}, {N}, {K})"
        try:
            cost = mapping_tensorcore(
                M,
                N,
                K,
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
    for cost in costs:
        print(cost)
