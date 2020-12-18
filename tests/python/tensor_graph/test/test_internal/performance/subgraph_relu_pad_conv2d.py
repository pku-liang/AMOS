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
from tensor_graph.nn.layers import Layer, Conv2d, ReLU


batch = 1
in_channel = 1024
out_channel = 1024
img_size = 14
kernel_size = 3
stride = 1
padding = 1
dilation = 1
groups = 1


class SubGraph(Layer):
  def __init__(self, in_channel, out_channel, kernel_size,
    bias=False, stride=stride, padding=padding, dilation=dilation, groups=groups, dtype="float32"):
    super(SubGraph, self).__init__()
    self.relu = ReLU()
    self.conv = Conv2d(in_channel, out_channel, kernel_size,
    bias=False, stride=stride, padding=padding, dilation=dilation, groups=groups, dtype="float32")
  
  def forward(self, x):
    return self.conv(self.relu(x))


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
  model = SubGraph(
    in_channel, out_channel, kernel_size,
    bias=False, stride=stride, padding=padding, dilation=dilation, groups=groups, dtype="float32")
  img_shape = [batch, in_channel, img_size, img_size]
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
  model = SubGraph(
    in_channel, out_channel, kernel_size,
    bias=False, stride=stride, padding=padding, dilation=dilation, groups=groups, dtype="float32")
  img_shape = [batch, in_channel, img_size, img_size]
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
    if not no_run:
      beg = time.time()
      tg.run_task(sess, task_id, [inputs_bindings] * number, save_to=save_to)
      end = time.time()
      print("Passing %f min" % ((time.time() - start_time) / 60), flush=True)
      print("Average time cost for one iteration:", (end - beg) * 1e3 / number, "ms", flush=True)
    else:
      time.sleep(600)
      beg = time.time()
      tg.run_task(sess, task_id, [inputs_bindings] * number, save_to=save_to)
      end = time.time()
      print("Passing %f min" % ((time.time() - start_time) / 60), flush=True)
      print("Average time cost for one iteration:", (end - beg) * 1e3 / number, "ms", flush=True)

    if (time.time() - start_time) / 60 > tune_minutes:
      print("Tuning last for over %f minutes, stop tuning" % tune_minutes)
      tg.end_tuning(sess, task_id)
      break
  
  # remember to delete the session before exit
  tg.delete_session(sess)
  return 0


yolo_shapes_b1 = [
  # batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
  # yolo
  (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
  (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
  (1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
  (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
  (1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1),  # conv5   4
  (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
  (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv7   6
  (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
  # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv9
  # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv10
  # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv11
  # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv12
  # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv13
  # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv14
  (1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1),  # conv15      8
  (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
  (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv17    10
  (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
  # # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv19
  # # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv20
  (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
  (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
  (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
  # (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv24
]


shapes = {"yolo": yolo_shapes_b1}


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

  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--in_channel", type=int, default=1024)
  parser.add_argument("--out_channel", type=int, default=1024)
  parser.add_argument("--img_size", type=int, default=14)
  parser.add_argument("--kernel_size", type=int, default=3)
  parser.add_argument("--stride", type=int, default=1)
  parser.add_argument("--padding", type=int, default=1)
  parser.add_argument("--dilation", type=int, default=1)
  parser.add_argument("--groups", type=int, default=1)

  parser.add_argument("--shape", type=str, default="")
  parser.add_argument("--layer", type=int, default=0)


  args = parser.parse_args()

  batch = args.batch
  in_channel = args.in_channel
  out_channel = args.out_channel
  img_size = args.img_size
  kernel_size = args.kernel_size
  stride = args.stride
  padding = args.padding
  dilation = args.dilation
  groups = args.groups

  if args.shape in shapes:
    shape = shapes[args.shape]
    layer = shape[args.layer]
    # batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
    in_channel = layer[1]
    out_channel = layer[4]
    img_size = layer[2]
    kernel_size = layer[6]
    stride = layer[9]
    padding = layer[10]
    dilation = layer[11]
    groups = layer[12]

  start_evaluate()
  target = args.target
  dev_id = args.device
  evalute_exit_code = evaluate_function_for(target, args.eval_device, args.timeout)
  if args.tune:
    producer_exit_code = producer_process(
      target, dev_id, args.delete, args.exe_iter, args.tune_iter, args.minutes, args.reference, args.save,
      args.first_stage, args.second_stage_ratio, no_run=args.no_run)
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

  