import tvm
import os
import sys
import shutil
import tvm._ffi
import numpy as np
import random
import multiprocessing
import multiprocessing.pool
import psutil
import signal
import queue
from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from ..utils import to_tuple, ERROR



GLOBAL_BUILD_ARGS = None
measure_card_id = 0
measure_number = 10
measure_timeout = 10
measure_parallel = multiprocessing.cpu_count()


class EvaluationContext(object):
  def __init__(self):
    self.task_queue = multiprocessing.Queue()
    self.result_queue = multiprocessing.Queue()
    self.stop = False
    self.dir_name = "evaluate_pool"
    self.file_name = "to_evaluate"
    self.file_path = os.path.join(self.dir_name, self.file_name)
    if not (os.path.exists(self.dir_name) and os.path.isdir(self.dir_name)):
      os.mkdir(self.dir_name)
    elif not (len(os.listdir(self.dir_name)) == 0):
      shutil.rmtree(self.dir_name)
      os.mkdir(self.dir_name)
    


GLOBAL_EVAL_CTX = EvaluationContext()


def set_measure_card_id(new_id):
  global measure_card_id
  measure_card_id = new_id


def set_meaure_number(new_number):
  global measure_number
  measure_number = new_number


def set_measure_timeout(new_timeout):
  global measure_timeout
  measure_timeout = new_timeout


def set_measure_parallel(new_parallel):
  global measure_parallel
  measure_parallel = new_parallel


class NoDaemonProcess(multiprocessing.Process):
  @property
  def daemon(self):
      return False

  @daemon.setter
  def daemon(self, value):
      pass


class NoDaemonContext(type(multiprocessing.get_context())):
  Process = NoDaemonProcess


class NoDaemonPool(multiprocessing.pool.Pool):
  """A no daemon pool version of multiprocessing.Pool.
  This allows us to start new processings inside the worker function"""

  def __init__(self, *args, **kwargs):
    kwargs['context'] = NoDaemonContext()
    super().__init__(*args, **kwargs)

  def __reduce__(self):
    pass


def kill_child_processes(parent_pid, sig=signal.SIGTERM):
  """kill all child processes recursively"""
  try:
    parent = psutil.Process(parent_pid)
  except psutil.NoSuchProcess:
    return
  children = parent.children(recursive=True)
  for process in children:
    try:
      process.send_signal(sig)
    except psutil.NoSuchProcess:
      return


def call_func_with_timeout(timeout, func, args=(), kwargs=None):
  """Call a function with timeout"""
  def func_wrapper(que):
    if kwargs:
      que.put(func(*args, **kwargs))
    else:
      que.put(func(*args))

  que = multiprocessing.Queue(2)
  process = multiprocessing.Process(target=func_wrapper, args=(que,))
  process.start()
  process.join(timeout)

  try:
    res = que.get(block=False)
  except queue.Empty:
    res = TimeoutError()

  # clean queue and process
  kill_child_processes(process.pid)
  process.terminate()
  process.join()
  que.close()
  que.join_thread()
  del process
  del que

  return res


def evaluate_function(sch, tensors, target):
  arrays = []
  ctx = tvm.context(target.target_name, measure_card_id)
  for t in tensors:
    ary = tvm.nd.array(np.random.uniform(-1, 1, to_tuple(t.shape)).astype(t.dtype))
    arrays.append(ary)
  try:
    print("check target:", target)
    print("check context:", ctx.exist)
    func = tvm.build(sch, tensors, target)
    time_evaluator = func.time_evaluator(func.entry_name, ctx, number=measure_number)
  except Exception as e:
    print(e)
    return e
  return time_evaluator(*arrays).mean * 1e3


def measure_function(index, q=None):
  global GLOBAL_BUILD_ARGS

  if GLOBAL_BUILD_ARGS is None:
    raise RuntimeError("No build arguments found!")

  schedules, tensors, target = GLOBAL_BUILD_ARGS

  print("check context outer:", tvm.context(target.target_name, 0).exist)

  sch = schedules[index]
  measure = call_func_with_timeout(measure_timeout, evaluate_function, args=(sch, tensors, target))
  if isinstance(measure, TimeoutError):
    ret = float("inf")
  elif isinstance(measure, Exception):
    ret = float("inf")
  ret = measure
  if q is not None:
    q.put(ret)
  return ret


def measure_multi_function(number):
  processes = []
  ques = []
  res = []
  for i in range(number):
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=measure_function, args=(i, q))
    p.start()
    processes.append(p)
    ques.append(q)
  for p in processes:
    p.join()
  for q in ques:
    res.append(q.get())
  return res


# @tvm._ffi.register_func("tg.autoschedule.judge_schedule")
def judge_schedule(schedules, tensors, target, gflop, policy):
  print("check context outer outer:", tvm.context(target.target_name, 0).exist)
  if policy == "profile":
    global GLOBAL_BUILD_ARGS
    GLOBAL_BUILD_ARGS = (schedules, tensors, target)

    pool = NoDaemonPool(measure_parallel)
    res = pool.map(measure_function, range(len(schedules)))
    
    pool.terminate()
    pool.join()
    del pool

    # res = measure_multi_function(len(schedules))

    ret = []
    for r in res:
      if r == float("inf"):
        ret.append(0.0)
      elif abs(r) < 1e-5:
        ret.append(0.0)
      else:
        ret.append(gflop / r)
    return ret
  elif policy == "model":
    sys.exit("Not implemented policy: model")
  else:
    sys.exit("No support for policy: %s" % policy)


@tvm._ffi.register_func("tg.runtime.evaluate_performance")
def evaluate_performance(modules, name, tensors):
  global GLOBAL_EVAL_CTX
  file_path = GLOBAL_EVAL_CTX.file_path
  for i, module in enumerate(modules):
    module.export_library(file_path + "_" + str(i) + ".so")
  tensor_ctx = []
  for t in tensors:
    tensor_ctx.append({"shape": to_tuple(t.shape), "dtype": str(t.dtype)})
  
  GLOBAL_EVAL_CTX.task_queue.put({"name": name, "tensor_ctx": tuple(tensor_ctx), "number": len(modules)})

  results = GLOBAL_EVAL_CTX.result_queue.get()

  for i in range(len(modules)):
    os.remove(file_path + "_" + str(i) + ".so")

  print("measure:", results, flush=True)
  return results


def start_evaluate():
  global GLOBAL_EVAL_CTX
  GLOBAL_EVAL_CTX.stop = False


def stop_evaluate():
  global GLOBAL_EVAL_CTX
  GLOBAL_EVAL_CTX.stop = True
  GLOBAL_EVAL_CTX.task_queue.put(-1)


# @concurrent.process(timeout=10)
def _evaluate(args):
  idx, target, dev_id, name, tensor_ctx = args
  global GLOBAL_EVAL_CTX
  file_path = GLOBAL_EVAL_CTX.file_path
  arrays = []
  ctx = tvm.context(target, dev_id)

  for t_ctx in tensor_ctx:
    ary = tvm.nd.array(np.random.uniform(-1, 1, t_ctx["shape"]).astype(t_ctx["dtype"]), ctx)
    arrays.append(ary)

  func = tvm.runtime.load_module(file_path + "_" + str(idx) + ".so")
  time_evaluator = func.time_evaluator(name, ctx, number=5)
  result = time_evaluator(*arrays).mean * 1e3
  return result


@concurrent.process(daemon=False)
def evaluate_function_for(target, dev_id, timeout=10):
  global GLOBAL_EVAL_CTX
  while not GLOBAL_EVAL_CTX.stop:
    if not GLOBAL_EVAL_CTX.task_queue.empty():
      name_tensor_ctx = GLOBAL_EVAL_CTX.task_queue.get()
      if isinstance(name_tensor_ctx, int) and name_tensor_ctx == -1:
        break
      name = name_tensor_ctx["name"]
      tensor_ctx = name_tensor_ctx["tensor_ctx"]
      number = name_tensor_ctx["number"]

      with ProcessPool() as pool:
        args = []
        for i in range(number):
          args.append((i, target, dev_id, name, tensor_ctx))
        future = pool.map(_evaluate, args, timeout=timeout)
        iterator = future.result()

        results = []
        while True:
          try:
            result = next(iterator)
          except StopIteration:
            break
          except Exception as error:
            # print("Exception!", type(error), str(error), flush=True)
            result = -1.0
          results.append(result)
        GLOBAL_EVAL_CTX.result_queue.put(results)
  return 0


def set_evaluate_performance(func):
  tvm._ffi.register_func("tg.runtime.evaluate_performance", func, True)