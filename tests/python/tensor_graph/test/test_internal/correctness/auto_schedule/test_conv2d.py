import tvm
import os
import sys
import time
import tvm._ffi
import numpy as np
from tvm import tg
from tvm.tensor_graph.nn.layers import Conv2d
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


if __name__ == "__main__":
  os.environ["TG_PRINT_LEVEL"] = "4"
  autoschedule_log_file="autoschedule_log.txt"
  autoschedule_profile_file="autoschedule_log_profile.txt"
  build_log_file="build_log.txt"
  execution_log_file="execution_log.txt"

  lst = [autoschedule_log_file, autoschedule_profile_file, build_log_file, execution_log_file]
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
    execution_timeout=4.0,
    synchronize_subgraph=True,
    execution_log_file=execution_log_file
  )
  sess = tg.create_session("cuda", 0, log_option)
  # get the model
  model = Conv2d(512, 1024, 3, padding=1)
  batch = 1
  img_shape = [batch, 512, 28, 28]
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
  # execute graph by 'number' iterations
  number = 1
  repeats = 10
  for i in range(repeats):
    beg = time.time()
    if i > repeats / 2:
      tg.disable_autoschedule(sess)
    tg.run_task(sess, task_id, [inputs_bindings] * number)
    end = time.time()
    print("Average time cost for one iteration:", (end - beg) * 1e3 / number, "ms")
  
  # remember to delete the session before exit
  tg.delete_session(sess)
  print("Success!")
