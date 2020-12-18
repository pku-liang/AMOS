import tvm
import os
import sys
import time
import tvm._ffi
import numpy as np
import argparse

from tvm import tg
from pebble import concurrent
from tensor_graph.core import evaluate_function_for, start_evaluate, stop_evaluate
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
from tensor_graph.core.utils import to_tuple
from tensor_graph.nn.layers import Layer


def zero_pad2d(inputs, padding=0):
    """
    Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : GraphNode
        shape [batch, channel, height, width, m, n]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------
    
    Returns:
    -----------------------------
    GraphOp
        shape [batch, channel, padded_height, padded_width, m, n]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert (len(padding) == 4)
    batch_size, in_channel, height, width, M, N = inputs.shape
    padded_shape = (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3], M, N)
    def _inner_zero_pad2d(batch_size, in_channel, h, w, m, n, inputs, requires_grad=True):
        # Warning, we use "float32" as type of 0
        padding_zero = tvm.tir.expr.const(0, "float32")
        return compute(padded_shape,
            lambda b, c, h, w, i, j: tvm.te.if_then_else(
                                tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                                inputs[b, c, h - padding[0], w - padding[2], i, j],
                                padding_zero
                                ),
            name="zero_pad2d",
            requires_grad=requires_grad)
    return GraphOp(padded_shape, [] , [inputs], _inner_zero_pad2d, name="zero_pad2d")


def capsule_conv2d(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Capsule convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w, in_m, in_k = inputs.shape
    out_channel, channel_per_group, k_h, k_w, k_k, k_n = weight.shape
    assert channel_per_group * groups == in_channel, "%d vs. %d" % (channel_per_group * groups, in_channel)
    out_channel_per_group = out_channel // groups
    assert out_channel_per_group * groups == out_channel
    assert in_k == k_k

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)
    conv_out_shape = (batch_size, out_channel, out_h, out_w, in_m, k_n)
    def _inner(batch_size, out_channel, out_h, out_w, out_m, out_n, channel_per_group,
                                k_w, k_h, k_k, padded, weight, requires_grad=True):
        rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
        rw = tvm.te.reduce_axis((0, k_w), name="rw")
        rh = tvm.te.reduce_axis((0, k_h), name="rh")
        rk = tvm.te.reduce_axis((0, k_k), name="rk")
        return compute(conv_out_shape,
            lambda b, c, h, w, i, j: tvm.te.sum(
                (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                        h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1], i, rk]
                * weight[c, rc, rh, rw, rk, j]),
                axis=[rc, rw, rh, rk]),
                name="capsule_conv2d",
                requires_grad=requires_grad
            )
    conv_out = GraphOp(
        conv_out_shape,
        [channel_per_group, k_w, k_h, k_k],
        [padded, weight],
        _inner,
        name="capsule_conv2d")
    def _inner_bias(batch_size, out_channel, out_h, out_w, out_m, out_n, conv_out, bias, requires_grad=True):
        return compute(
            (batch_size, out_channel, out_h, out_w, out_m, out_n),
            lambda b, c, h, w, i, j: conv_out[b, c, h, w, i, j] + bias[c],
            name="capsule_conv2d_bias",
            requires_grad=requires_grad
            )
    if bias is not None:
        return GraphOp(conv_out_shape, [], [conv_out, bias], _inner_bias, name="capsule_conv2d")
    return conv_out


class CapsuleConv2d(Layer):
  def __init__(self, in_channel, out_channel, kernel_size, capsule_size,
        bias=False, stride=1, padding=0, dilation=1, groups=1, dtype="float32"):
    super(CapsuleConv2d, self).__init__()
    self.in_channel = in_channel
    self.out_channel = out_channel
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2
    capsule_size = (capsule_size, capsule_size) if isinstance(capsule_size, int) else capsule_size
    assert isinstance(capsule_size, (tuple, list)) and len(capsule_size) == 2
    stride = (stride, stride) if isinstance(stride, int) else stride
    assert isinstance(stride, (list, tuple)) and len(stride) == 2
    padding = (padding, padding) if isinstance(padding, int) else padding
    assert isinstance(padding, (tuple, list)) and len(padding) == 2
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    assert isinstance(dilation, (tuple, list)) and len(dilation) == 2
    assert isinstance(groups, int)

    self.kernel_size = kernel_size
    self.capsule_size = capsule_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups

    self.weight = GraphTensor(
      (out_channel, in_channel, *kernel_size, *capsule_size), dtype=dtype, name="capsule_conv2d_weight", requires_grad=True)
    if bias:
      self.bias = GraphTensor((out_channel,), dtype=dtype, name="capsule_conv2d_bias", requires_grad=True)
    else:
      self.bias = None

  def forward(self, inputs):
    return capsule_conv2d(
      inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



def random_initialize_weights(weight_tensors, ctx):
  init = []
  for w in weight_tensors:
    ary = np.random.uniform(-1, 1, to_tuple(w.shape)).astype(w.dtype)
    init.append(tvm.nd.array(ary, ctx))
  return init


def clear_log_files(filenames):
  for filename in filenames:
    if os.path.exists(filename) and os.path.isfile(filename):
      os.remove(filename)


@concurrent.process
def consumer_process(target, dev_id, number=10, repeat=10, reference="", profile_level=0):
  os.environ["TG_PRINT_LEVEL"] = "4"
  execution_log_file="tmp.txt"

  lst = [execution_log_file]
  clear_log_files(lst)

  # create a session
  log_option = tg.create_session_option(
    report_profile=True,
    report_iteration=True,
    report_iteration_period=1,
    autoschedule_topk=20,
    autoschedule_new_trial=10,
    autoschedule_policy="random",
    autoschedule_parallel=1,
    autoschedule_timeout=200.0,
    profile_parallel=1,
    profile_timeout=4.0,
    build_parallel=1,
    build_timeout=1.0,
    execution_explore_probability=0.5,
    execution_parallel=1,
    execution_timeout=100.0,
    execution_log_file=execution_log_file
  )
  sess = tg.create_session(target, dev_id, log_option)
  # get the model
  in_channel = 1
  out_channel = 256
  kernel_size = 9
  in_capsule_size = (1, 1)
  weight_capsule_size = (1, 8)
  model = CapsuleConv2d(
    in_channel, out_channel, kernel_size, weight_capsule_size,
    bias=False, stride=2, padding=0, dilation=1, groups=1, dtype="float32")
  batch = 1
  img_shape = [batch, 1, 28, 28, *in_capsule_size]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")

  # get forward graph and tir graph
  fwd_graph = make_fwd_graph(model, [img_tensor])
  tir_graph = make_tir_graph(fwd_graph, inference=True)

  ctx = tg.get_context_from_session(sess)
  inputs_data = np.random.uniform(-1, 1, img_shape).astype(dtype)
  inputs_bindings = {tir_graph.inputs[0]: tvm.nd.array(inputs_data, ctx)}
  weight_bindings = random_initialize_weights(tir_graph.weights, ctx)
  # initialize weights
  tg.initialize_weights(sess, tir_graph, weight_bindings)
  # add task
  task_id = tg.add_task(sess, tir_graph)
  # replay
  print("test schedules from", reference)
  tg.test_schedule_reference(sess, task_id, reference=reference)
  # execute graph by 'number' iterations
  number = number
  repeats = repeat
  for i in range(repeats):
    beg = time.time()
    tg.run_task(sess, task_id, [inputs_bindings] * number, profile_level=profile_level, save_to="")
    end = time.time()
    print("Average time cost for one iteration:", (end - beg) * 1e3 / number, "ms")
  
  # remember to delete the session before exit
  tg.delete_session(sess)
  return 0


@concurrent.process
def producer_process(
  target, dev_id, delete_log=False, exe_iter=100, max_tune_iter=10000, tune_minutes=4,
  reference="", save_to="", first_stage_number=100000, second_stage_topk_ratio=0.1, no_run=False):
  os.environ["TG_PRINT_LEVEL"] = "4"
  autoschedule_log_file="autoschedule_log.txt"
  autoschedule_profile_file="autoschedule_log_profile.txt"
  build_log_file="build_log.txt"
  evaluate_log_file="evaluate_log.txt"
  execution_log_file="execution_log.txt"

  lst = [autoschedule_log_file, autoschedule_profile_file, build_log_file, evaluate_log_file, execution_log_file]
  if delete_log:
    clear_log_files(lst)

  # create a session
  log_option = tg.create_session_option(
    report_profile=True,
    report_iteration=True,
    report_iteration_period=1,
    autoschedule_topk=20,
    autoschedule_new_trial=4,
    autoschedule_policy="fc_model",
    autoschedule_parallel=1,
    autoschedule_timeout=200.0,
    autoschedule_log_file=autoschedule_log_file,
    profile_parallel=1,
    profile_timeout=4.0,
    build_parallel=1,
    build_timeout=1.0,
    build_log_file=build_log_file,
    execution_explore_probability=0.5,
    execution_parallel=4,
    execution_timeout=100.0,
    execution_log_file=execution_log_file
  )
  sess = tg.create_session(target, dev_id, log_option)
  # get the model
  in_channel = 1
  out_channel = 256
  kernel_size = 9
  in_capsule_size = (1, 1)
  weight_capsule_size = (1, 8)
  model = CapsuleConv2d(
    in_channel, out_channel, kernel_size, weight_capsule_size,
    bias=False, stride=2, padding=0, dilation=1, groups=1, dtype="float32")
  batch = 1
  img_shape = [batch, 1, 28, 28, *in_capsule_size]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")

  # get forward graph and tir graph
  fwd_graph = make_fwd_graph(model, [img_tensor])
  tir_graph = make_tir_graph(fwd_graph, inference=True)

  ctx = tg.get_context_from_session(sess)
  inputs_data = np.random.uniform(-1, 1, img_shape).astype(dtype)
  inputs_bindings = {tir_graph.inputs[0]: tvm.nd.array(inputs_data, ctx)}
  weight_bindings = random_initialize_weights(tir_graph.weights, ctx)
  # initialize weights
  tg.initialize_weights(sess, tir_graph, weight_bindings)
  # add task
  task_id = tg.add_task(sess, tir_graph)
  # tune
  tg.begin_tuning(sess, task_id, max_tune_iter,
    reference=reference, first_stage_number=first_stage_number, second_stage_topk_ratio=second_stage_topk_ratio)
  # execute graph by 'number' iterations
  number = exe_iter
  start_time = time.time()
  while True:
    beg = time.time()
    tg.run_task(sess, task_id, [inputs_bindings] * number, save_to=save_to, no_actual_run=no_run)
    end = time.time()
    if no_run:
      time.sleep(10 * 60)
    else:
      print("Average time cost for one iteration:", (end - beg) * 1e3 / number, "ms", flush=True)
    print("Passing %f min" % ((time.time() - start_time) / 60), flush=True)

    if (time.time() - start_time) / 60 > tune_minutes:
      print("Tuning last for over %f minutes, stop tuning" % tune_minutes)
      tg.end_tuning(sess, task_id)
      break
  
  # remember to delete the session before exit
  tg.delete_session(sess)
  return 0


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--tune", help="tuning", action="store_true")
  parser.add_argument("--test", help="testing", action="store_true")
  parser.add_argument("--delete", help="delete log", action="store_true")
  parser.add_argument("--minutes", help="tuning minutes", type=float, default=4.0 * 60)
  parser.add_argument("--timeout", help="timeout in seconds", type=float, default=10)
  parser.add_argument("--tune_iter", help="max tune iter", type=int, default=10000)
  parser.add_argument("--exe_iter", help="max execution iter", type=int, default=100)
  parser.add_argument("--reference", help="tuning reference", type=str, default="")
  parser.add_argument("--save", help="save to", type=str, default="saved_schedules.txt")
  parser.add_argument("--target", help="target device", type=str, default="llvm")
  parser.add_argument("--device", help="device id", type=int, default=0)
  parser.add_argument("--eval_device", help="evaluate device id", type=int, default=0)
  parser.add_argument("--eval_repeat", help="evaluate repeat", type=int, default=10)
  parser.add_argument("--eval_number", help="evaluate number for each repeat", type=int, default=10)
  parser.add_argument("--profile", help="profile level", type=int, default=0)
  parser.add_argument("--first_stage", help="first stage number", type=int, default=100000)
  parser.add_argument("--second_stage_ratio", help="second stage topk ratio", type=float, default=0.1)
  parser.add_argument("--no_run", help="do not run", action="store_true")

  args = parser.parse_args()

  start_evaluate()
  target = args.target
  dev_id = args.device
  evalute_exit_code = evaluate_function_for(target, args.eval_device, args.timeout)
  if args.tune:
    producer_exit_code = producer_process(
      target, dev_id, args.delete, args.exe_iter, args.tune_iter, args.minutes, args.reference, args.save,
      args.first_stage, args.second_stage_ratio, args.no_run)
    try:
      ret = producer_exit_code.result()
    except Exception as e:
      print(str(e))
  if args.test:
    exit_code = consumer_process(target, dev_id, args.eval_number, args.eval_repeat, args.reference, args.profile)
    try:
      ret = exit_code.result()
    except Exception as e:
      print(str(e))
  stop_evaluate()
  ret = evalute_exit_code.result()
  print("Success!")

  