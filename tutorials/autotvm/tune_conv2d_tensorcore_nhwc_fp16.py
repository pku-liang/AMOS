"""
Tuning High Performance Convolution on NVIDIA GPUs
=========================================================================
**Author**: `Lianmin Zheng <https://https://github.com/merrymercy>`_

This is an advanced tutorial for writing high performance tunable template for 
NVIDIA GPU. By running auto-tuner on this template, we can outperform the
vendor provided library CuDNN in many cases.
"""

######################################################################
# Install dependencies
# --------------------
# To use autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make tvm run faster in tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import logging
import sys
import numpy as np

import tvm
from tvm import topi
from tvm.topi import cuda

from tvm import autotvm

######################################################################
# Step 1:  Define the search space
# --------------------------------
# There are plenty of useful schedule primitives in tvm. You can also find 
# some tutorials that describe them in more details, such as 
# (1). :ref:`opt-conv-gpu`
# (2). `Optimizing DepthwiseConv on NVIDIA GPU <https://tvm.ai/2017/08/22/Optimize-Deep-Learning-GPU-Operators-with-TVM-A-Depthwise-Convolution-Example.html>`_
# 
# However, their implementations are manually tuned for some special input
# shapes. In this section, we build a large enough space to cover
# the techniques used in these tutorials. Then we rely on the efficient auto-tuner
# to search through this space and pick some good configurations.
# 
# If you are familiar with writing cuda schedule, you can find the following
# template is very general. Actually this template can be easily modified 
# to tune other operators such as depthwise convolution and gemm.
# In order to fully understand this template, you should be familiar with
# the schedule primitives and auto tuning API. You can refer to the above
# tutorials and :doc:`autotvm tutorial <tune_simple_template>`
#
# It is worth noting that the search space for a conv2d operator
# can be very large (at the level of 10^9 for some input shapes)
#

# to run this, first:
# export PATH=/usr/local/cuda-10.1/nvvm/libdevice:$PATH

@autotvm.template("conv2d_nhwc_tensorcore_test")
def conv2d_nhwc_tensorcore_test(N, H, W, CO, CI, KH, KW, stride, padding):
    data = tvm.te.placeholder((N, H, W, CI), name='data', dtype="float16")
    kernel = tvm.te.placeholder((KH, KW, CI, CO), name='kernel', dtype="float16")
    cfg = autotvm.get_config()

    conv = topi.cuda.nhwc_tensorcore_cuda(
        cfg, data, kernel, stride, padding, dilation=1, out_dtype='float16')
    s = tvm.te.create_schedule([conv.op])
    topi.cuda.schedule_nhwc_tensorcore_cuda(cfg, s, conv)

    return s, [data, kernel, conv]

######################################################################
# Step 2:  Search through the space
# ---------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBoostTuner` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this template

# logging config (for printing tuning log to screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


# the last layer in yolo
def run(name, N, H, W, CO, CI, KH, KW, stride, pad):
    N, H, W, CO, CI, KH, KW, strides, padding = N, H, W, CO, CI, KH, KW, (stride, stride), (pad, pad)
    task = autotvm.task.create("conv2d_nhwc_tensorcore_test",
                               args=(N, H, W, CO, CI, KH, KW, strides, padding),
                               target='cuda')
    print(task.config_space)
    logfile = "conv2d_" + name + ".log"

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.XGBTuner(task)
    tuner.tune(n_trial=1000,
               measure_option=measure_option,
               callbacks=[autotvm.callback.log_to_file(logfile)])

    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(logfile)
    best_config = dispatch_context.query(task.target, task.workload)
    print("\nBest config:")
    print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(logfile):
        with tvm.target.create("cuda"):
            s, arg_bufs = conv2d_nhwc_tensorcore_test(N, H, W, CO, CI, KH, KW, strides, padding)
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(N, H, W, CI)).astype(np.float16)
    w_np = np.random.uniform(size=(KH, KW, CI, CO)).astype(np.float16)
    c_np = np.random.uniform(size=(N, (H + 2 * pad - KH) // stride + 1, (W + 2 * pad - KW) // stride + 1, CO)).astype(np.float16)
    # c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.array(c_np, ctx=ctx)
    # c_tvm = tvm.nd.empty((N, CO, (H + 2 * pad - KH) // stride + 1, (W + 2 * pad - KW) // stride + 1), ctx=ctx)
    # func(a_tvm, w_tvm, c_tvm)

    # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
    # and the overhead of kernel launch. You can also use nvprof to validate the result.
    evaluator = func.time_evaluator(func.entry_name, ctx, number=400, min_repeat_ms=500)
    cost = evaluator(a_tvm, w_tvm, c_tvm).mean * 1e3
    # print('Time cost of this operator: %f' % cost)
    # with open("autotvm_conv_nhwc.txt", "a") as f:
    #     f.write("name, {}\n".format(cost))
    return cost


res18_shapes_b1 = [
    # resnet-18
    (16, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (16, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (16, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (16, 64, 56, 56, 128, 64, 3, 3, 1, 2, 1, 1, 1),  # conv4   3
    (16, 64, 56, 56, 128, 64, 1, 1, 1, 2, 0, 1, 1),  # conv5   4
    (16, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (16, 128, 28, 28, 256, 128, 3, 3, 1, 2, 1, 1, 1),  # conv7   6
    (16, 128, 28, 28, 256, 128, 1, 1, 1, 2, 0, 1, 1),  # conv8   7
    (16, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # conv9   8
    (16, 256, 14, 14, 512, 256, 3, 3, 1, 2, 1, 1, 1),  # conv10  9
    (16, 256, 14, 14, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # conv11  10
    (16, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv12  11
    # (256, 512, 7, 7, 512, 512, 1, 1, 1, 1, 0, 1, 1)
]


if __name__ == "__main__":
    costs = []
    for i, args in enumerate(res18_shapes_b1):
        name = "resnet-18-layer-" + str(i+1)
        N, CI, H, W, CO, _, KW, KH, _, stride, pad, _, _ = args
        N = (N + 15) // 16 * 16
        CI = (CI + 15) // 16 * 16
        CO = (CO + 15) // 16 * 16
        try:
            cost = run(name, N, H, W, CO, CI, KH, KW, stride, pad)
        except Exception as e:
            print(e, flush=True)
            cost = float("inf")
        costs.append(cost)
    print("The costs:")
    for cost in costs:
        print(cost)
