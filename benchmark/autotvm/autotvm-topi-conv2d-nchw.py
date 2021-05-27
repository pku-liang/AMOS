import os

import numpy as np

import tvm
from tvm import te
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime


class Initializer(object):
    """The base class of an initializer."""

    def __init__(self, **kwargs):
        self._kwargs = kwargs

    def __call__(self, desc, arr):
        """Initialize an array

        Parameters
        ----------
        desc : str
            Initialization pattern descriptor.

        arr : NDArray
            The array to be initialized.
        """
        if desc.endswith("weight"):
            self._init_weight(desc, arr)
        elif desc.endswith("bias"):
            self._init_bias(desc, arr)
        elif desc.endswith("gamma"):
            self._init_gamma(desc, arr)
        elif desc.endswith("beta"):
            self._init_beta(desc, arr)
        elif desc.endswith("mean"):
            self._init_mean(desc, arr)
        elif desc.endswith("var"):
            self._init_var(desc, arr)
        else:
            self._init_default(desc, arr)

    def _init_bias(self, _, arr):
        arr[:] = 0.0

    def _init_gamma(self, _, arr):
        arr[:] = 1.0

    def _init_beta(self, _, arr):
        arr[:] = 0.0

    def _init_mean(self, _, arr):
        arr[:] = 0.0

    def _init_var(self, _, arr):
        arr[:] = 1.0

    def _init_weight(self, name, arr):
        """Abstract method to Initialize weight."""
        raise NotImplementedError("Must override it")

    def _init_default(self, name, _):
        raise ValueError(
            "Unknown initialization pattern for %s. "
            "Default initialization is now limited to "
            '"weight", "bias", "gamma" (1.0), and "beta" (0.0).'
            "Please use mx.sym.Variable(init=mx.init.*) to set initialization pattern" % name
        )


class Xavier(Initializer):
    """ "Xavier" initialization for weights

    Parameters
    ----------
    rnd_type: str, optional
        Random generator type, can be ``'gaussian'`` or ``'uniform'``.

    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.

    magnitude: float, optional
        Scale of random number.
    """

    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        super(Xavier, self).__init__(
            rnd_type=rnd_type, factor_type=factor_type, magnitude=magnitude
        )
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)

    def _init_weight(self, name, arr):
        shape = arr.shape
        hw_scale = 1.0
        if len(shape) < 2:
            raise ValueError(
                "Xavier initializer cannot be applied to vector {0}. It requires at"
                " least 2D.".format(name)
            )
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        factor = 1.0
        if self.factor_type == "avg":
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == "in":
            factor = fan_in
        elif self.factor_type == "out":
            factor = fan_out
        else:
            raise ValueError("Incorrect factor type")
        # Hack for mobilenet, because there is less connectivity
        if "depthwise" in name:
            factor = hw_scale
        scale = np.sqrt(self.magnitude / factor)
        if self.rnd_type == "uniform":
            arr[:] = np.random.uniform(-scale, scale, size=arr.shape)
        else:
            raise ValueError("Unknown random type")


def conv2d_nchw(N, C, H, W, K, R, S, stride, padding, dilation, in_dtype=["float16", "float16"], out_dtype="float32"):
    data_shape = (N, C, H, W)
    weight_shape = (K, C, R, S)
    data = relay.var("data", shape=data_shape, dtype=in_dtype[0])
    weight = relay.var("weight", shape=weight_shape, dtype=in_dtype[1])
    conv2d = relay.nn.conv2d(
        data, weight,
        strides=(stride, stride),
        padding=(padding, padding),
        dilation=(dilation, dilation),
        groups=1,
        channels=K,
        kernel_size=(R, S),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_layout="NCHW",
        out_dtype=out_dtype)
    func = relay.Function([data, weight], conv2d)

    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    shape_dict = {
        v.name_hint : v.checked_type for v in mod["main"].params}
    np.random.seed(0)

    initializer = Xavier()
    params = {}
    for k, v in shape_dict.items():
        if k.startswith("data"):
            continue
        init_value = np.zeros(v.concrete_shape).astype(v.dtype)
        initializer(k, init_value)
        params[k] = tvm.nd.array(init_value, ctx=tvm.cpu(0))
    return mod, params, data_shape, weight_shape


###########################################
# Set Tuning Options
# ------------------
# Before tuning, we apply some configurations.

# You can skip the implementation of this function for this tutorial.
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


def tune_and_evaluate(N, C, H, W, K, R, S, stride, padding, dilation, in_dtypes, out_dtype, tuning_opt, target):
    # extract workloads from relay program
    print("Extract tasks...")
    log_file = tuning_opt["log_filename"]
    mod, params, input_shape, weight_shape = conv2d_nchw(N, C, H, W, K, R, S, stride, padding, dilation, in_dtypes, out_dtype)
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

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.GraphModule(lib["default"](ctx))
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(in_dtypes[0]))
        weight_tvm = tvm.nd.array((np.random.uniform(size=weight_shape)).astype(in_dtypes[1]))
        module.set_input("data", data_tvm)
        moduel.set_input("weight", weight_tvm)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )
    return np.mean(prof_res)


res18_shapes_b1 = [
    # resnet-18
    (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (1, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (1, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (1, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (1, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
]


if __name__ == "__main__":
    batches = [2**i for i in range(1)]
    beg = 0
    #num = 15
    num = 1
    in_dtypes = ["float32", "float32"]
    out_dtype = "float32"
    target = tvm.target.cuda()
    for batch in batches:
        costs = []
        #for i, shape in enumerate(yolo_shapes_b1[beg:beg+num]):
        for i, shape in enumerate(res18_shapes_b1[beg:beg+num]):
            (_, C, H, W, K, _, R, S, _, stride,
                padding, dilation, _) = shape
            N = batch
            print("\n\nProblem size:")
            print(N, C, H, W, K, R, S, stride, padding)
            log_file = "conv2d_nchw_topi_layer_" + str(i + 1) + ".log"
            tuning_option = {
                "log_filename": log_file,
                "tuner": "xgb",
                "n_trial": 1000,
                "early_stopping": 200,
                "measure_option": autotvm.measure_option(
                    builder=autotvm.LocalBuilder(timeout=10),
                    runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
                ),
            }
            # try:
            cost = tune_and_evaluate(
                N, C, H, W, K, R, S, stride,
                padding, dilation,
                in_dtypes,
                out_dtype,
                tuning_option,
                target
            )
            costs.append(cost)
            # except Exception as e:
            #     print("Fail to run\n", str(e))
            #     costs.append(float("inf"))
        print("\nBatch=", batch)
        for cost in costs:
            print(cost)

# target = tvm.target.cuda()

# tuning_option = {
#     "log_filename": log_file,
#     "tuner": "xgb",
#     "n_trial": 2000,
#     "early_stopping": 600,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(timeout=10),
#         runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
#     ),
# }


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
