import tvm
import os
from tvm import auto_tensorize as at


def gemm(M, N, K):
    A = tvm.te.placeholder([M, K], dtype="float16", name="A")
    B = tvm.te.placeholder([K, N], dtype="float16", name="B")

    rk = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute([M, N], lambda i, j: tvm.te.sum((A[i, rk] * B[rk, j]).astype("float16"), axis=rk), name="C")
    return [A, B, C]


def tensorize_tensorcore_fp16fp16(
    M, N, K, layer
):
    A, B, C = gemm(M, N, K)
    target_dag = at.compute_dag_from_tensors([C])
    target = "cuda"

    log_file = "gemm-fp16-layer-%d.log" % (layer)

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


def run(M, N, K, layer):
    return tensorize_tensorcore_fp16fp16(
        M, N, K, layer)


gemm_shapes = [
   (16, 512, 128),
   (1024, 16, 256),
   (256, 1024, 256),
   (512, 256, 16),
   (1024, 1024, 1024)
]


if __name__ == "__main__":
    beg = 0
    num = 5
    costs = []
    for i, shape in enumerate(gemm_shapes[beg:beg+num]):
        (M, N, K) = shape
        print("\n\nProblem size:")
        print(M, N, K)
        try:
            cost = run(
                M, N, K,
                i + beg + 1
            )
            costs.append(cost)
        except Exception as e:
            print("Fail to run\n", str(e))
            costs.append(float("inf"))

    for cost in costs:
        print(cost)
