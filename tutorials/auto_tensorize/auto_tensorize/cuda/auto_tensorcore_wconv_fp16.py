import tvm
import os
from tvm import auto_tensorize as at


# M.Conv2d(self.M*oup, oup*inp*ksize*ksize, 1, 1, 0, groups=self.G*oup, bias=False)
# ksize, stride, padding = 1, 1, 0
def grouped_pointwise_conv2d(N, I, O, groups):
    channel_per_group = I // groups
    out_channel_per_group = O // groups

    A = tvm.te.placeholder([N, I], dtype="float16", name="A")
    B = tvm.te.placeholder([O, channel_per_group], dtype="float16", name="B")
    
    A_reshaped = tvm.te.compute(
        [N, groups, channel_per_group],
        lambda n, c_o, c_i: A[n, c_o * channel_per_group + c_i]
    )

    B_reshaped = tvm.te.compute(
        [groups, out_channel_per_group, channel_per_group],
        lambda k_o, k_i, c: B[k_o * out_channel_per_group + k_i, c]
    )

    rc = tvm.te.reduce_axis([0, channel_per_group], name="rc")

    WConv = tvm.te.compute(
        [N, groups, out_channel_per_group],
        lambda n, k_o, k_i:
            tvm.te.sum((A_reshaped[n, k_o, rc] * B_reshaped[k_o, k_i, rc]
                        ).astype("float16"), axis=[rc, ]),
        name="WConv"
    )

    return [A, B, WConv]


def tensorize_tensorcore_fp16fp16(
    N, I, O, groups, layer
):
    A, B, WConv = grouped_pointwise_conv2d(N, I, O, groups)
    target_dag = at.compute_dag_from_tensors([WConv])
    target = "cuda"

    log_file = "wconv2d-fp16-layer-%d-batch-%d.log" % (layer, N)

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


def run(N, I, O, groups, layer):
    return tensorize_tensorcore_fp16fp16(
        N, I, O, groups, layer)


# https://github.com/megvii-model/WeightNet/blob/master/shufflenet_v2.py
# in_channels, out_channels, groups (ksize, stride, padding = 1, 1, 0)
shuffle_v2_cfg = [
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


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    #num = 15
    num = 10
    for batch in batches:
        costs = []
        for i, cfg in enumerate(shuffle_v2_cfg[beg:beg+num]):
            I, O, groups = cfg
            N = batch
            print("\n\nProblem size:")
            print(N, I, O, groups)
            try:
                cost = run(
                    N, I, O, groups,
                    i + beg + 1
                )
                costs.append(cost)
            except Exception as e:
                print("Fail to run\n", str(e))
                costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)
