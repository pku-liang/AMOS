import tvm
import os
import sys
import time
import tvm._ffi
import numpy as np
from tvm import tg
from pebble import concurrent
from tensor_graph.testing.models import yolo_v1
from tensor_graph.core import evaluate_function_for, start_evaluate, stop_evaluate
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
from tensor_graph.core.utils import to_tuple
from tensor_graph.nn import CELoss, SGD
from tensor_graph.nn.layers import Conv2d


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


def main_process(target, dev_id):
  os.environ["TG_PRINT_LEVEL"] = "4"
  autoschedule_log_file="autoschedule_log.txt"
  autoschedule_profile_file="autoschedule_log_profile.txt"
  build_log_file="build_log.txt"
  evaluate_log_file="evaluate_log.txt"
  execution_log_file="execution_log.txt"

  lst = [autoschedule_log_file, autoschedule_profile_file, build_log_file, evaluate_log_file, execution_log_file]
  clear_log_files(lst)

  # create a session
  log_option = tg.create_session_option(
    report_profile=True,
    report_iteration=True,
    report_iteration_period=1,
    autoschedule_topk=20,
    autoschedule_new_trial=4,
    autoschedule_policy="random",
    autoschedule_parallel=1,
    autoschedule_timeout=200.0,
    autoschedule_log_file=autoschedule_log_file,
    profile_parallel=1,
    profile_timeout=4.0,
    build_parallel=1,
    build_timeout=1.0,
    build_log_file=build_log_file,
    execution_explore_probability=0.5,
    execution_parallel=1,
    execution_timeout=100.0,
    execution_log_file=execution_log_file
  )
  sess = tg.create_session(target, dev_id, log_option)
  # get the model
  model = yolo_v1.yolo_v1()
  batch = 1
  num_classes = 1470
  img_shape = [batch, 3, 448, 448]
  label_shape = [batch, num_classes]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")
  label_tensor = GraphTensor(label_shape, dtype, name="label")
  # get forward graph and tir graph
  fwd_graph = make_fwd_graph(model, [img_tensor])
  loss = CELoss(label_tensor)
  optimizer = SGD(lr=0.002)
  tir_graph = make_tir_graph(fwd_graph, loss=loss, optimizer=optimizer, inference=False)
  tir_graph = tg.inline_graph(tir_graph)

  ctx = tg.get_context_from_session(sess)
  inputs_data = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_data = np.random.uniform(0, 2, label_shape).astype(dtype)
  lr_data = optimizer.get_lr().astype(dtype)
  inputs_bindings = {
    tir_graph.inputs[0]: tvm.nd.array(inputs_data, ctx),
    tir_graph.labels[0]: tvm.nd.array(label_data, ctx),
    tir_graph.lr: tvm.nd.array(lr_data, ctx)
  }
  weight_bindings = random_initialize_weights(tir_graph.weights, ctx)
  # initialize weights
  tg.initialize_weights(sess, tir_graph, weight_bindings)
  # add task
  task_id = tg.add_task(sess, tir_graph)
  tg.print_subgraphs(sess, task_id)
  
  # remember to delete the session before exit
  tg.delete_session(sess)
  return 0


if __name__ == "__main__":
  target = "llvm"
  dev_id = 0
  exit_code = main_process(target, dev_id)

  print("Success!")

  