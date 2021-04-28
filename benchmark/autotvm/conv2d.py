import os

import numpy as np

import tvm
from tvm import relay, autotvm
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_runtime as runtime
import numpy as np
import json
from collections import namedtuple
import conv2d


class ConvParams(object):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, strides=(1, 1), padding=(0, 0), bias=True, groups=1, use_fp16=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.use_fp16 = use_fp16

    def valid(self):
        return (
            self.in_channels is not None and
            self.out_channels is not None and
            self.kernel_size is not None and
            self.strides is not None and
            self.padding is not None and
            self.bias is not None and
            self.groups is not None and
            self.use_fp16 is not None
        )

    def to_tuple(self):
        return (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.strides,
            self.padding,
            self.bias,
            self.groups,
            self.use_fp16
        )

    def to_tuple_key(self):
        tmp = (
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.strides[0],
            self.padding[0],
            self.groups
        )
        ret = ",".join([str(x) for x in tmp])
        return ret

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.to_tuple() == other.to_tuple()
        else:
            return self.to_tuple() == other

    def __repr__(self):
        return "ConvParams" + str(self.to_tuple())


class ConvFullParams(object):
    def __init__(self, batch=None, H=None, W=None, in_channels=None, out_channels=None, kernel_size=None, strides=(1, 1), padding=(0, 0), bias=True, groups=1, use_fp16=False):
        self.batch = batch
        self.H = H
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.use_fp16 = use_fp16

    def valid(self):
        return (
            self.batch is not None and
            self.H is not None and
            self.W is not None and
            self.in_channels is not None and
            self.out_channels is not None and
            self.kernel_size is not None and
            self.strides is not None and
            self.padding is not None and
            self.bias is not None and
            self.groups is not None and
            self.use_fp16 is not None
        )

    def to_tuple(self):
        return (
            self.batch,
            self.H,
            self.W,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.strides,
            self.padding,
            self.bias,
            self.groups,
            self.use_fp16
        )

    def to_tuple_key(self):
        tmp = (
            self.batch,
            self.H,
            self.W,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.strides[0],
            self.padding[0],
            self.groups
        )
        ret = ",".join([str(x) for x in tmp])
        return ret

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.to_tuple() == other.to_tuple()
        else:
            return self.to_tuple() == other

    def __repr__(self):
        return "ConvFullParams" + str(self.to_tuple())


ConvShape = namedtuple("ConvShape", "batch, channels, height, width")


ConvShapeItem = namedtuple("ConvShapeItem", "count, shapes")


ConvShapePerf = namedtuple("ConvShapeItem", "shape, perf")


def get_conv_shapes(filename="conv_op_config_longtail.txt"):
    ret = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            conv_param = ConvParams(*obj["param"])
            count = obj["count"]
            shapes = obj["shapes"]
            shapes = [ConvShape(*x) for x in shapes]
            item = ConvShapeItem(count, shapes)
            ret.append((conv_param, item))

    return ret


def get_conv2d(N, C, H, W, K, R, S, stride, padding, dilation, groups, dtype="float32"):
    """Get the symbol definition and random weight of a network"""
    input_shape = (N, C, H, W)
    output_shape = (K, C, R, S)
    mod, params = relay.testing.conv2d.get_workload(
            N, C, H, W, K, R, S, stride, padding, dilation, groups, dtype
        )

    return mod, params, input_shape, output_shape


def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate(N, C, H, W, K, R, S, stride, padding, dilation, groups, dtype, target, log_file, tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_conv2d(N, C, H, W, K, R, S, stride, padding, dilation, groups, dtype)
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=target, params=params, ops=(relay.op.get("nn.conv2d"),)
    )

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=target, params=params)

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.GraphModule(lib["default"](ctx))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input("data", data_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )


def tune_conv2d(N, C, H, W, K, R, S, stride, padding, dilation, groups, dtype="float32"):
    target = tvm.target.cuda()

    #### TUNING OPTION ####
    network = "conv2d(" + ",".join(map(str, [N, C, H, W, K, R, S, stride, padding, dilation, groups, dtype])) + ")"
    log_file = "%s.log" % network

    tuning_option = {
        "log_filename": log_file,
        "tuner": "xgb",
        "n_trial": 2000,
        "early_stopping": 600,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=10),
            runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        ),
    }

    tune_and_evaluate(N, C, H, W, K, R, S, stride, padding, dilation, groups, dtype, target, log_file, tuning_option)


# tuning_option = {
#     "log_filename": log_file,
#     "tuner": "xgb",
#     "n_trial": 2000,
#     "early_stopping": 600,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(timeout=10),
#         runner=autotvm.RPCRunner(
#             "1080ti",  # change the device key to your key
#             "0.0.0.0",
#             9190,
#             number=20,
#             repeat=3,
#             timeout=4,
#             min_repeat_ms=150,
#         ),
#     ),
# }


if __name__ == "__main__":
    shapes = get_conv_shapes()
    for param, item in shapes[:10]:
        C, K, kernel_size, strides, padding, bias, groups, use_fp16 = param.to_tuple()
        dtype = "float32" if not use_fp16 else "float16"
        for ss in item.shapes:
            N, C, H, W = tuple(ss)
            tune_conv2d(N, C, H, W, K, *kernel_size, strides[0], padding[0], 1, groups, dtype)
