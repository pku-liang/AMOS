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
import copy
import argparse
import numpy as np
import torch
import tvm
from tvm import te
from tvm import topi
from topi.testing import conv2d_nchw_python
from tvm.autotvm.task.space import SplitEntity

from tvm import autotvm

######################################################################

@autotvm.template("test_tvm/performance/conv2d_hwcn")
def conv2d_hwcn(N, K, C, H, W, R, S, stride, padding, dilation):
  dtype="float32"
  Input = tvm.te.placeholder([H, W, C, N], dtype=dtype, name="Input")
  Filter = tvm.te.placeholder([R, S, C, K], dtype=dtype, name="Filter")
  Output = topi.nn.conv2d_hwcn(Input, Filter, stride, padding, dilation, dtype)

  """Schedule conv2d_hwcn"""
  B = Output
  A = Input
  W = Filter
  sch = tvm.te.create_schedule(B.op)
  cfg = autotvm.get_config()
  PaddedInput = sch[B].op.input_tensors[0]
  sch[PaddedInput].compute_inline()
  AA = sch.cache_read(PaddedInput, "shared", [B])
  WW = sch.cache_read(W, "shared", [B])
  AL = sch.cache_read(AA, "local", [B])
  WL = sch.cache_read(WW, "local", [B])

  if B.op in sch.outputs:
      Out = B
      BL = sch.cache_write(Out, "local")
  else:
      Out = sch.outputs[0].output(0)
      sch[B].set_scope("local")
      BL = B

  hi, wi, fi, ni = sch[Out].op.axis

  # Create tuning space
  n_thread_cand = [1, 2, 4, 8, 16, 32]
  vthread_cand = [1, 2, 4, 8]

  cfg.define_split(
      'tile_fi',
      fi,
      num_outputs=4,
      filter=lambda x:
      (x.size[1] in vthread_cand and x.size[2] in n_thread_cand))
  cfg.define_split(
      'tile_ni',
      ni,
      num_outputs=4,
      filter=lambda x:
      (x.size[1] in vthread_cand and x.size[2] in n_thread_cand))

  if cfg.is_fallback:
      cfg['tile_fi'] = SplitEntity([-1, 2, 8, 4])
      cfg['tile_ni'] = SplitEntity([-1, 2, 8, 4])

  # Scheduling
  step = 8

  bz = sch[Out].fuse(hi, wi)
  by, tyz, ty, fi = cfg['tile_fi'].apply(sch, Out, fi)
  bx, txz, tx, ni = cfg['tile_ni'].apply(sch, Out, ni)
  sch[Out].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

  sch[Out].bind(bz, te.thread_axis('blockIdx.z'))
  sch[Out].bind(by, te.thread_axis('blockIdx.y'))
  sch[Out].bind(bx, te.thread_axis('blockIdx.x'))
  sch[Out].bind(tyz, te.thread_axis('vthread'))
  sch[Out].bind(txz, te.thread_axis('vthread'))
  sch[Out].bind(ty, te.thread_axis('threadIdx.y'))
  sch[Out].bind(tx, te.thread_axis('threadIdx.x'))

  # Schedule BL local write
  sch[BL].compute_at(sch[Out], tx)
  yi, xi, fi, ni = sch[BL].op.axis
  ry, rx, rc = sch[BL].op.reduce_axis
  rco, rci = sch[BL].split(rc, factor=step)
  sch[BL].reorder(rco, ry, rx, rci, fi, ni)
  fuse_index = sch[BL].fuse(ry, rx)
  fuse_index = sch[BL].fuse(fuse_index, rco)
  rx = fuse_index

  sch[AA].compute_at(sch[BL], rx)
  sch[WW].compute_at(sch[BL], rx)
  sch[AL].compute_at(sch[BL], rci)
  sch[WL].compute_at(sch[BL], rci)
  # Schedule for A's shared memory load
  yi, xi, ci, ni = sch[AA].op.axis
  ty, ci = sch[AA].split(ci, nparts=cfg['tile_fi'].size[2])
  tx, ni = sch[AA].split(ni, nparts=cfg['tile_ni'].size[2])
  _, ni = sch[AA].split(ni, factor=4)
  sch[AA].reorder(ty, tx, yi, xi, ci, ni)
  sch[AA].bind(ty, te.thread_axis('threadIdx.y'))
  sch[AA].bind(tx, te.thread_axis('threadIdx.x'))
  sch[AA].vectorize(ni)
  # Schedule for W's shared memory load
  yi, xi, ci, fi = sch[WW].op.axis
  ty, ci = sch[WW].split(ci, nparts=cfg['tile_fi'].size[2])
  tx, fi = sch[WW].split(fi, nparts=cfg['tile_ni'].size[2])
  _, fi = sch[WW].split(fi, factor=4)
  sch[WW].reorder(ty, tx, yi, xi, ci, fi)
  sch[WW].bind(ty, te.thread_axis('threadIdx.y'))
  sch[WW].bind(tx, te.thread_axis('threadIdx.x'))
  sch[WW].vectorize(fi)

  return sch, [Input, Filter, Output]


######################################################################
# logging config (for printing tuning log to screen)
logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))


res_shapes_b1 = [
  # resnet-50
  # batch, C, H, W, K, _, R, S, _, st, pad, dilation, group 
  (1, 3, 224, 224, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1
  (1, 64, 56, 56, 256, 64, 1, 1, 1, 1, 0, 1, 1),  # res2a_branch1
  (1, 64, 56, 56, 64, 64, 1, 1, 1, 1, 0, 1, 1),  # res2a_branch2a
  (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # res2a_branch2b
  # (1, 64, 56, 56, 256, 64, 1, 1, 1, 1, 0, 1, 1),  # res2a_branch2c
  (1, 256, 56, 56, 64, 256, 1, 1, 1, 1, 0, 1, 1),  # res2b_branch2a
  # (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # res2b_branch2b
  # (1, 64, 56, 56, 256, 64, 1, 1, 1, 1, 0, 1, 1),  # res2b_branch2c
  # (1, 256, 56, 56, 64, 256, 1, 1, 1, 1, 0, 1, 1),  # res2c_branch2a
  # (1, 64, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1, 1),  # res2c_branch2b
  # (1, 64, 56, 56, 256, 64, 1, 1, 1, 1, 0, 1, 1),  # res2c_branch2c
  (1, 256, 56, 56, 512, 256, 1, 1, 1, 2, 0, 1, 1),  # res3a_branch1
  (1, 256, 56, 56, 128, 256, 1, 1, 1, 2, 0, 1, 1),  # res3a_branch2a
  (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # res3a_branch2b
  (1, 128, 28, 28, 512, 128, 1, 1, 1, 1, 0, 1, 1),  # res3a_branch2c
  (1, 512, 28, 28, 128, 512, 1, 1, 1, 1, 0, 1, 1),  # res3b_branch2a
  # (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # res3b_branch2b
  # (1, 128, 28, 28, 512, 128, 1, 1, 1, 1, 0, 1, 1),  # res3b_branch2c
  # (1, 512, 28, 28, 128, 512, 1, 1, 1, 1, 0, 1, 1),  # res3c_branch2a
  # (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # res3c_branch2b
  # (1, 128, 28, 28, 512, 128, 1, 1, 1, 1, 0, 1, 1),  # res3c_branch2c
  # (1, 512, 28, 28, 128, 512, 1, 1, 1, 1, 0, 1, 1),  # res3d_branch2a
  # (1, 128, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1, 1),  # res3d_branch2b
  # (1, 128, 28, 28, 512, 128, 1, 1, 1, 1, 0, 1, 1),  # res3d_branch2c
  (1, 512, 28, 28, 1024, 512, 1, 1, 1, 2, 0, 1, 1),  # res4a_branch1
  (1, 512, 28, 28, 256, 512, 1, 1, 1, 2, 0, 1, 1),  # res4a_branch2a
  (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # res4a_branch2b
  (1, 256, 14, 14, 1024, 256, 1, 1, 1, 1, 0, 1, 1),  # res4a_branch2c
  (1, 1024, 14, 14, 256, 1024, 1, 1, 1, 1, 0, 1, 1),  # res4b_branch2a
  # (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # res4b_branch2b
  # (1, 256, 14, 14, 1024, 256, 1, 1, 1, 1, 0, 1, 1),  # res4b_branch2c
  # (1, 1024, 14, 14, 256, 1024, 1, 1, 1, 1, 0, 1, 1),  # res4c_branch2a
  # (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # res4c_branch2b
  # (1, 256, 14, 14, 1024, 256, 1, 1, 1, 1, 0, 1, 1),  # res4c_branch2c
  # (1, 1024, 14, 14, 256, 1024, 1, 1, 1, 1, 0, 1, 1),  # res4d_branch2a
  # (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # res4d_branch2b
  # (1, 256, 14, 14, 1024, 256, 1, 1, 1, 1, 0, 1, 1),  # res4d_branch2c
  # (1, 1024, 14, 14, 256, 1024, 1, 1, 1, 1, 0, 1, 1),  # res4e_branch2a
  # (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # res4e_branch2b
  # (1, 256, 14, 14, 1024, 256, 1, 1, 1, 1, 0, 1, 1),  # res4e_branch2c
  # (1, 1024, 14, 14, 256, 1024, 1, 1, 1, 1, 0, 1, 1),  # res4f_branch2a
  # (1, 256, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1, 1),  # res4f_branch2b
  # (1, 256, 14, 14, 1024, 256, 1, 1, 1, 1, 0, 1, 1),  # res4f_branch2c
  (1, 1024, 14, 14, 2048, 1024, 1, 1, 1, 2, 0, 1, 1),  # res5a_branch1
  (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 2, 0, 1, 1),  # res5a_branch2a
  (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # res5a_branch2b
  (1, 512, 7, 7, 2048, 512, 1, 1, 1, 1, 0, 1, 1),  # res5a_branch2c
  (1, 2048, 7, 7, 512, 2048, 1, 1, 1, 1, 0, 1, 1),  # res5b_branch2a
  # (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # res5b_branch2b
  # (1, 512, 7, 7, 2048, 512, 1, 1, 1, 1, 0, 1, 1),  # res5b_branch2c
  # (1, 2048, 7, 7, 512, 2048, 1, 1, 1, 1, 0, 1, 1),  # res5c_branch2a
  # (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # res5c_branch2b
  # (1, 512, 7, 7, 2048, 512, 1, 1, 1, 1, 0, 1, 1),  # res5c_branch2c
]

def copy_change_batch(batch, x):
    ret = copy.deepcopy(x[1:])
    ret = (batch,) + ret
    return ret

conv2d_shapes = [copy_change_batch(64, x) for x in res_shapes_b1]

def to_int(expr):
    try:
        res = int(expr)
    except Exception as e:
        raise RuntimeError("fail to convert to int: %s" % str(e))
    return res


def to_tuple(expr_tuple):
    return tuple([to_int(x) for x in expr_tuple])


def tvm_conv2d_hwcn(test_id, test_only=False, number=100, dev=0):
  task_name = 'conv2d_hwcn_%d' % (test_id)
  N, C, H, W, K, _, R, S, _, stride, padding, dilation, group = conv2d_shapes[test_id]
  task = autotvm.task.create("test_tvm/performance/conv2d_hwcn",
                            args=(N, K, C, H, W, R, S, stride, padding, dilation),
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
          s, arg_bufs = conv2d_hwcn(N, K, C, H, W, R, S, stride, padding, dilation)
          func = tvm.build(s, arg_bufs)

  # check correctness
  a_np = np.random.uniform(size=(H, W, C, N)).astype(np.float32)
  w_np = np.random.uniform(size=(R, S, C, K)).astype(np.float32)
  c_np = np.zeros(to_tuple(arg_bufs[2].shape)).astype(np.float32)

  ctx = tvm.gpu(dev)
  a_tvm = tvm.nd.array(a_np, ctx=ctx)
  w_tvm = tvm.nd.array(w_np, ctx=ctx)
  c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
  func(a_tvm, w_tvm, c_tvm)

  # tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

  evaluator = func.time_evaluator(func.entry_name, ctx, number=number)
  print('Time cost of tvm conv2d_hwcn (%d): %f ms' % (test_id, evaluator(a_tvm, w_tvm, c_tvm).mean*1e3))


def pytorch_conv2d_nchw(test_id, use_cudnn=False, number=100, dev=0):
  torch.backends.cudnn.enabled = use_cudnn
  N, C, H, W, K, _, R, S, _, stride, padding, dilation, group = conv2d_shapes[test_id]
  A = torch.rand([N, C, H, W], dtype=torch.float32).cuda("cuda:" + str(dev))
  W = torch.rand([K, C//group, R, S], dtype=torch.float32).cuda("cuda:" + str(dev))

  # warm-up
  torch.nn.functional.conv2d(A, W, stride=stride, padding=padding, dilation=dilation, groups=group)
  torch.cuda.synchronize()
  sum_time = 0.0
  for i in range(number):
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      ans = torch.nn.functional.conv2d(A, W, stride=stride, padding=padding, dilation=dilation, groups=group)
      end.record()

      # Waits for everything to finish running
      torch.cuda.synchronize()

      sum_time += start.elapsed_time(end)
  algo = "cudnn" if use_cudnn else "native"
  print('Time cost of pytorch(%s) conv2d_nchw (%d): %f ms' % (algo, test_id, sum_time / number))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tune", help="tune tvm", action="store_true")
  parser.add_argument("--test", help="test tvm and pytorch", action="store_true")
  parser.add_argument("--number", help="number of runs", type=int, default=100)
  parser.add_argument("--dev", help="which device to run", type=int, default=0)

  args = parser.parse_args()
  if args.tune:
    for test_id in range(len(conv2d_shapes)):
      tvm_conv2d_hwcn(test_id, False, args.number, args.dev)
  
  if args.test:
    for test_id in range(len(conv2d_shapes)):
      tvm_conv2d_hwcn(test_id, True, args.number, args.dev)
      pytorch_conv2d_nchw(test_id, False, args.number, args.dev)
      pytorch_conv2d_nchw(test_id, True, args.number, args.dev)