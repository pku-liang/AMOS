import tvm
import os
import sys
import time
import tvm._ffi
import numpy as np
from tvm import tg
from pebble import concurrent
from tvm.tensor_graph.testing.models import lenet
from tvm.tensor_graph.core import evaluate_function_for, start_evaluate, stop_evaluate
from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
from tvm.tensor_graph.core.utils import to_tuple
from tvm.tensor_graph.nn.layers import Conv2d


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
  model = lenet.lenet5()
  batch = 1
  img_shape = [batch, 1, 32, 32]
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
  # begin tuning
  tuning_iteration = 1000
  tg.begin_tuning(sess, task_id, tuning_iteration)
  # execute graph by 'number' iterations
  number = 100
  repeats = 10
  for i in range(repeats):
    if i == repeats / 2:
      print("Stop tuning")
      tg.end_tuning(sess, task_id)
    beg = time.time()
    tg.run_task(sess, task_id, [inputs_bindings] * number)
    end = time.time()
    print("Average time cost for one iteration:", (end - beg) * 1e3 / number, "ms")
  
  # remember to delete the session before exit
  tg.delete_session(sess)
  return 0


if __name__ == "__main__":
  start_evaluate()
  target = "llvm"
  dev_id = 0
  evalute_exit_code = evaluate_function_for(target, 1)
  exit_code = main_process(target, dev_id)
  try:
    ret = exit_code.result()
  except Exception as e:
    print(e)
  stop_evaluate()
  ret = evalute_exit_code.result()
  print("Success!")

  