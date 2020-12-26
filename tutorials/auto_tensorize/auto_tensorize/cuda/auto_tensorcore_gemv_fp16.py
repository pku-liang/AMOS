import tvm
import os
from tvm import auto_tensorize as at


def gemv(M, K):
    A = tvm.te.placeholder([M, K], dtype="float16", name="A")
    B = tvm.te.placeholder([K, 1], dtype="float16", name="B")

    rk = tvm.te.reduce_axis([0, K], name="k")
    C = tvm.te.compute([M, 1], lambda i, j: tvm.te.sum((A[i, rk] * B[rk, j]).astype("float16"), axis=rk), name="C")
    D = tvm.te.compute([M], lambda i: C[i, 0], name="D")
    return [A, B, D]


def tensorize_tensorcore_fp16fp16(
    M, K, layer
):
    A, B, D = gemv(M, K)
    target_dag = at.compute_dag_from_tensors([D])
    target = "cuda"

    log_file = "gemvfp16-layer-%d.log" % (layer)

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


def run(M, K, layer):
    return tensorize_tensorcore_fp16fp16(
        M, K, layer)


gemv_shapes = [
   (512, 16),
   (1024, 256),
   (256, 1024),
   (512, 256),
   (1024, 1024)
]


if __name__ == "__main__":
    beg = 0
    num = 5
    costs = []
    for i, shape in enumerate(gemv_shapes[beg:beg+num]):
        (M, K) = shape
        print("\n\nProblem size:")
        print(M, K)
        try:
            cost = run(
                M, K,
                i + beg + 1
            )
            costs.append(cost)
        except Exception as e:
            print("Fail to run\n", str(e))
            costs.append(float("inf"))

    for cost in costs:
        print(cost)
