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
import argparse

from tvm.tensor_graph.core.transform import ParallelFusionFinder, ParallelFusionApplier

def random_initialize_weights(weight_tensors, ctx):
  init = []
  cnt = 1
  for w in weight_tensors:
    np.random.seed(cnt)
    cnt += 1
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
  model = lenet.lenet5()
  batch = 1
  img_shape = [batch, 1, 32, 32]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")

  # get forward graph and tir graph
  fwd_graph = make_fwd_graph(model, [img_tensor])
  # parallel fusion
  finder = ParallelFusionFinder()
  finder(fwd_graph)
  applier = ParallelFusionApplier(finder.fusion_groups)
  fwd_graph = applier.transform(fwd_graph)

  tir_graph = make_tir_graph(fwd_graph, inference=True)

  ctx = tg.get_context_from_session(sess)
  np.random.seed(0)
  inputs_data = np.random.uniform(-1, 1, img_shape).astype(dtype)
  inputs_bindings = {tir_graph.inputs[0]: tvm.nd.array(inputs_data, ctx)}
  output_keys = [tir_graph.outputs[0]]
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
    outputs_data = tg.get_data_from_session(sess, output_keys)
    print(outputs_data[0], flush=True)
  
  # remember to delete the session before exit
  tg.delete_session(sess)
  return 0


@concurrent.process
def producer_process(
  target, dev_id, delete_log=False, exe_iter=100, max_tune_iter=10000, tune_minutes=4 * 60,
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
    execution_parallel=4,
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
  # parallel fusion
  finder = ParallelFusionFinder()
  finder(fwd_graph)
  applier = ParallelFusionApplier(finder.fusion_groups)
  fwd_graph = applier.transform(fwd_graph)

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

# Our result
# [[ -68.7905    649.16345   -82.88051   277.40057   226.81372  -588.3368
#   -459.29205  -312.90576  -584.4164     34.990578]]

# Pytorch Result
# tensor_graph/testing/pytorch_examples/lenet_annotated.py
#[[ -68.8599,  649.2622,  -82.8930,  277.4005,  226.7232, -588.3690,
#   -459.3513, -312.9294, -584.4176,   35.0095]]