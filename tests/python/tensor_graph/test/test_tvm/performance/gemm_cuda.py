"""
Tuning High Performance GEMM on NVIDIA GPUs
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
# To make TVM run faster in tuning, it is recommended to use cython
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
import argparse
import numpy as np
import torch
import tvm
from tvm import te
from tvm import topi
from topi.testing import conv2d_nchw_python
from tvm.autotvm.task.space import SplitEntity

from tvm import autotvm
from tvm.contrib import cublas

######################################################################


@autotvm.template("test_tvm/performance/gemm")
def gemm(batch, in_dim, out_dim):
    data = tvm.te.placeholder([batch, in_dim], dtype="float32", name="data")
    weight = tvm.te.placeholder(
        [out_dim, in_dim], dtype="float32", name="data")

    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = te.reduce_axis((0, in_dim), name='k')
    matmul = tvm.te.compute((batch, out_dim),
                            lambda i, j: te.sum(data[i, k] *
                                                weight[j, k], axis=k),
                            name='gemm')

    s = tvm.te.create_schedule(matmul.op)
    cfg = autotvm.get_config()

    if batch < 32:
        C = matmul
        A, _ = C.op.input_tensors

        cfg.define_split('tile_k', in_dim, num_outputs=2)
        if cfg.is_fallback:
            cfg["tile_k"] = SplitEntity([-1, 64] if in_dim > 64 else [1, 64])

        _, kf = cfg['tile_k'].apply(s, C, C.op.reduce_axis[0])
        CF = s.rfactor(C, kf)

        if C.op in s.outputs:
            Out = C
        else:
            Out = s.outputs[0].output(0)
            s[C].compute_at(s[Out], s[Out].op.axis[1])
        s[Out].bind(s[Out].op.axis[0], te.thread_axis("blockIdx.y"))
        s[Out].bind(s[Out].op.axis[1], te.thread_axis("blockIdx.x"))

        tx = s[C].op.reduce_axis[0]
        thread_x = te.thread_axis("threadIdx.x")
        s[C].bind(tx, thread_x)
        s[CF].compute_at(s[C], tx)
        s[C].set_store_predicate(thread_x.var.equal(0))
        s[Out].set_store_predicate(thread_x.var.equal(0))
    else:
        C = matmul
        A, B = C.op.input_tensors
        k = C.op.reduce_axis[0]
        batch, out_dim = s[C].op.axis
        in_dim = k

        # create tuning space
        try:
            block_cand = [64, 128]
            vthread_cand = [2**x for x in range(1, 7)]
            n_thread_cand = [2**x for x in range(3, 7)]
            cfg.define_split('tile_x', batch, num_outputs=4,
                             filter=lambda x: (x.size[1] in vthread_cand and
                                               x.size[2] in n_thread_cand and
                                               (x.size[1] * x.size[2] * x.size[3]) in block_cand))
            cfg.define_split('tile_y', out_dim, num_outputs=4,
                             filter=lambda x: (x.size[1] in vthread_cand and
                                               x.size[2] in n_thread_cand and
                                               (x.size[1] * x.size[2] * x.size[3]) in block_cand))
            cfg.define_split('tile_k', in_dim, num_outputs=3,
                             filter=lambda x: x.size[0] > 2)
        except IndexError:
            cfg.define_split('tile_x', batch, num_outputs=4)
            cfg.define_split('tile_y', out_dim, num_outputs=4)
            cfg.define_split('tile_k', in_dim, num_outputs=3)

        if cfg.is_fallback:
            if batch > 1:
                cfg['tile_x'] = SplitEntity([-1, 2, 16, 2])
            else:
                cfg['tile_x'] = SplitEntity([1, 1, 1, 1])
            if out_dim > 1:
                cfg['tile_y'] = SplitEntity([-1, 2, 16, 2])
            else:
                cfg['tile_y'] = SplitEntity([1, 1, 1, 1])
            if in_dim > 8:
                cfg['tile_k'] = SplitEntity([-1, 8, 1])
            else:
                cfg['tile_k'] = SplitEntity([-1, 1, 1])

        # Explicit memory access
        AA = s.cache_read(A, "shared", [C])
        BB = s.cache_read(B, "shared", [C])
        AL = s.cache_read(AA, "local", [C])
        BL = s.cache_read(BB, "local", [C])
        CC = s.cache_write(C, "local")

        # Deal with op fusion
        if C.op not in s.outputs:
            print("Remind me if one day this is visited, zsz...", __file__)
            s[C].compute_inline()
            C = s.outputs[0].output(0)

        # Split and reorder computation
        bx, txz, tx, xi = cfg['tile_x'].apply(s, C, C.op.axis[0])
        by, tyz, ty, yi = cfg['tile_y'].apply(s, C, C.op.axis[1])
        s[C].reorder(by, bx, tyz, txz, ty, tx, yi, xi)
        s[CC].compute_at(s[C], tx)

        # Binding
        s[C].bind(by, te.thread_axis("blockIdx.y"))
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(tyz, te.thread_axis("vthread"))
        s[C].bind(txz, te.thread_axis("vthread"))
        s[C].bind(ty, te.thread_axis("threadIdx.y"))
        s[C].bind(tx, te.thread_axis("threadIdx.x"))

        # Split reduction
        yo, xo = CC.op.axis
        ko, kt, ki = cfg['tile_k'].apply(s, CC, k)
        s[CC].reorder(ko, kt, ki, yo, xo)
        s[AA].compute_at(s[CC], ko)
        s[BB].compute_at(s[CC], ko)
        s[CC].unroll(kt)
        s[AL].compute_at(s[CC], kt)
        s[BL].compute_at(s[CC], kt)

        # Schedule for A's shared memory load
        num_thread_x = cfg['tile_x'].size[2]
        ty, _ = s[AA].split(s[AA].op.axis[0], nparts=num_thread_x)
        _, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread_x * 4)
        tx, xi = s[AA].split(xi, nparts=num_thread_x)
        s[AA].bind(ty, te.thread_axis("threadIdx.y"))
        s[AA].bind(tx, te.thread_axis("threadIdx.x"))
        s[AA].double_buffer()

        # Schedule for B' shared memory load
        num_thread_y = cfg['tile_y'].size[2]
        ty, _ = s[BB].split(s[BB].op.axis[0], nparts=num_thread_y)
        _, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread_y * 4)
        tx, xi = s[BB].split(xi, nparts=num_thread_y)
        s[BB].bind(ty, te.thread_axis("threadIdx.y"))
        s[BB].bind(tx, te.thread_axis("threadIdx.x"))
        s[BB].double_buffer()

    return s, [data, weight, matmul]


######################################################################
# logging config (for printing tuning log to screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


def tvm_gemm(batch, in_dim, out_dim, test_only=False, number=100, dev=0):
    task_name = 'gemm_%d_%d_%d' % (batch, in_dim, out_dim)
    task = autotvm.task.create("test_tvm/performance/gemm",
                               args=(batch, in_dim, out_dim),
                               target='cuda')
    print(len(task.config_space))

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
    file_name = '%s.log' % (task_name)
    if not test_only:
        tuner.tune(n_trial=2000,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.log_to_file(file_name)])

    #########################################################################
    # Finally we can inspect the best config from log file, check correctness,
    # and measure running time.

    # inspect the best config
    dispatch_context = autotvm.apply_history_best(file_name)
    best_config = dispatch_context.query(task.target, task.workload)
    # print("\nBest config:")
    # print(best_config)

    # apply history best from log file
    with autotvm.apply_history_best(file_name):
        with tvm.target.create("cuda"):
            s, arg_bufs = gemm(batch, in_dim, out_dim)
            func = tvm.build(s, arg_bufs)

    # check correctness
    a_np = np.random.uniform(size=(batch, in_dim)).astype(np.float32)
    w_np = np.random.uniform(size=(out_dim, in_dim)).astype(np.float32)
    c_np = np.matmul(a_np, w_np.T)

    ctx = tvm.gpu(dev)
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    print('Time cost of tvm gemm (%d %d %d): %f ms' %
          (batch, in_dim, out_dim, evaluator(a_tvm, w_tvm, c_tvm).mean*1e3))


def cublas_gemm(N, K, M, number=100, dev=0):
    data = tvm.te.placeholder([N, K], dtype="float32", name="data")
    weight = tvm.te.placeholder(
        [M, K], dtype="float32", name="data")

    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    matmul = cublas.matmul(data, weight, transb=True, dtype="float32")

    s = tvm.te.create_schedule(matmul.op)

    func = tvm.build(s, [data, weight, matmul], target="cuda")

    ctx = tvm.gpu(dev)
    a_np = np.random.uniform(size=(N, K)).astype(np.float32)
    w_np = np.random.uniform(size=(M, K)).astype(np.float32)
    c_np = np.matmul(a_np, w_np.T)

    ctx = tvm.gpu(dev)
    a_tvm = tvm.nd.array(a_np, ctx=ctx)
    w_tvm = tvm.nd.array(w_np, ctx=ctx)
    c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
    func(a_tvm, w_tvm, c_tvm)

    tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
    print('Time cost of cublas gemm (%d %d %d): %f ms' %
          (N, K, M, evaluator(a_tvm, w_tvm, c_tvm).mean*1e3))


def pytorch_gemm(N, K, M, number=100, dev=0):
    A = torch.rand([N, K], dtype=torch.float32).cuda("cuda:" + str(dev))
    B = torch.rand([K, M], dtype=torch.float32).cuda("cuda:" + str(dev))

    # warm-up
    torch.mm(A, B)
    torch.cuda.synchronize()
    sum_time = 0.0
    for i in range(number):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        ans = torch.mm(A, B)
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        sum_time += start.elapsed_time(end)

    print('Time cost of pytorch gemm (%d %d %d): %f ms' % (N, K, M, sum_time / number))


gemm_shape = [
    # standard
    (1024, 1024, 1024),
    (512, 512, 512),
    (256, 256, 256),
    # special
    (32, 1024, 10),
    (64, 1024, 100),
    (32, 1024, 1024),
    (64, 512, 2048)
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", help="tune tvm", action="store_true")
    parser.add_argument(
        "--test", help="test tvm and pytorch", action="store_true")
    parser.add_argument("--number", help="number of runs",
                        type=int, default=100)
    parser.add_argument("--dev", help="which device to run",
                        type=int, default=0)

    args = parser.parse_args()
    if args.tune:
        for (batch, in_dim, out_dim) in gemm_shape:
            tvm_gemm(batch, in_dim, out_dim, False, args.number, args.dev)

    if args.test:
        for (batch, in_dim, out_dim) in gemm_shape:
            tvm_gemm(batch, in_dim, out_dim, True, args.number, args.dev)
            cublas_gemm(batch, in_dim, out_dim, args.number, args.dev)
            pytorch_gemm(batch, in_dim, out_dim, args.number, args.dev)
