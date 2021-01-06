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

@autotvm.template("dense_tensorcore")
def dense_tensorcore(N, K, M):
    data = tvm.te.placeholder((N, K), name='data', dtype="float16")
    kernel = tvm.te.placeholder((M, K), name='kernel', dtype="float16")
    cfg = autotvm.get_config()

    dense = topi.cuda.dense_tensorcore_cuda(
        data, kernel, out_dtype='float16')
    # s = tvm.te.create_schedule([dense.op])
    s = topi.cuda.schedule_dense_tensorcore(cfg, dense)

    return s, [data, kernel, dense]

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
def run(name, N, K, M):
    task = autotvm.task.create("dense_tensorcore",
                               args=(N, K, M),
                               target='cuda')
    print(task.config_space)
    logfile = "conv3d_" + name + ".log"

    # Use local gpu, measure 10 times for every config to reduce variance
    # The timeout of compiling a program is 10 seconds, the timeout for running is 4 seconds
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(repeat=3, min_repeat_ms=100, timeout=4)
    )

    # Begin tuning, log records to file `conv2d.log`
    # During tuning we will also try many invalid configs, so you are expected to
    # see many error reports. As long as you can see non-zero GFLOPS, it is okay.
    tuner = autotvm.tuner.GATuner(task)
    tuner.tune(n_trial=100,
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
            s, arg_bufs = dense_tensorcore(N, K, M)
            func = tvm.build(s, arg_bufs)

    # check correctness
    # a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
    # w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
    # c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

    ctx = tvm.gpu()
    a_tvm = tvm.nd.empty([int(x) for x in arg_bufs[0].shape], dtype=arg_bufs[0].dtype, ctx=ctx)
    w_tvm = tvm.nd.empty([int(x) for x in arg_bufs[1].shape], dtype=arg_bufs[1].dtype, ctx=ctx)
    c_tvm = tvm.nd.empty([int(x) for x in arg_bufs[-1].shape], dtype=arg_bufs[-1].dtype, ctx=ctx)
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


gemm_shapes = [
   (16, 512, 128),
   (1024, 16, 256),
   (256, 1024, 256),
   (512, 256, 16),
   (1024, 1024, 1024)
]


if __name__ == "__main__":
    costs = []
    for i, args in enumerate(gemm_shapes):
        name = "gemm-" + str(i+1)
        M, N, K = args
        try:
            cost = run(name, N, K, M)
        except Exception as e:
            print(e, flush=True)
            cost = float("inf")
        costs.append(cost)
    print("The costs:")
    for cost in costs:
        print(cost)
