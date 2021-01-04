import tvm
import sys
import os
import tvm._ffi
import numpy as np
import multiprocessing
import multiprocessing.pool
import psutil
import signal
import queue
import copy
import json
import time
import math

from functools import reduce
from .schedule_state import RealScheduleState
from .hardware_config import get_hardware_config
from .schedule_merge import schedule_cuda_merge
from .schedule_allredcue import schedule_cuda_allreduce
from .schedule_buffer_output import schedule_cuda_buffer_output
from .schedule_tiling_and_binding import schedule_cuda_tiling_and_binding
from .schedule_buffer_input import schedule_cuda_buffer_input, create_buffer
from .schedule_unroll import schedule_cuda_unroll
from .utils import tile_axis, tile_axes, reorder_spatial_and_reduce_axes
from ..utils import to_tuple, to_int, can_to_int, to_int_or_None, ASSERT, ERROR

from tvm import auto_tensorize as at, tg
from tvm import auto_scheduler
from tvm.auto_scheduler.cost_model import RandomModel, XGBModel
from tvm.auto_scheduler.search_policy import SketchPolicy

from collections import OrderedDict


def interpret_cuda_schedule(sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
  op_to_id = {}
  op_to_state = {}
  for i, op in enumerate(subgraph.operation_list):
    op_to_id[op] = i
    op_to_state[op] = RealScheduleState("cuda")
  # reverse order, from output to input
  for op in reversed(subgraph.operation_list):
    schedule_cuda_merge(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_buffer_output(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    create_buffer(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_allreduce(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_tiling_and_binding(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_buffer_input(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    schedule_cuda_unroll(op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug)


def interpret_llvm_schedule(sch, tensors, subgraph, multi_entity, hd_config):
  return


@tvm._ffi.register_func("tg.autoschedule.interpret")
def interpret(sch, tensors, subgraph, target, multi_entity):
  with open("trace_debug_autoschedule.log", "a") as debug:
    if str(target.kind) == "cuda":
      hd_config = get_hardware_config("default_cuda", "cuda")
      interpret_cuda_schedule(sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
    elif str(target.kind) == "llvm":
      hd_config = get_hardware_config("default_llvm", "llvm")
      interpret_llvm_schedule(sch, tensors, subgraph, multi_entity, hd_config)
    else:
      ERROR("Currently no support for target", target.kind, type(target.kind))
    return


def set_interpret(func):
  tvm._ffi.register_func("tg.autoschedule.interpret", func, True)


@tvm._ffi.register_func("tg.autoschedule.auto_tensorize_cuda")
def auto_tensorize_cuda(sch, tensors, log_file, trials):
    target = "cuda"
    measure_opt = at.MeasureOptions(
        target=target, timeout=10, number=200, min_repeat_ms=500)
    target_dag = at.compute_dag_from_tensors(tensors)
    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, transform_dump=True)
    if result.defined():
        sch, args = at.get_schedule(result.sch_app, result.params)
        return {sch: args}
    else:
        return {sch: []}


@tvm._ffi.register_func("tg.autoschedule.build_func")
def build_func(
  sch,
  args,
  target,
  target_host,
  name,
  binds):
    binds = {x: y for x, y in binds.items()}
    return tvm.build(sch, args, target, target_host, name, binds)


class TGAutoScheduleContext(object):
  def __init__(self, name, subgraph, measure_option, verbose=False):
    self.measure_option = measure_option
    self.target = tvm.target.Target(measure_option.target)
    self.name = name
    self.subgraph = subgraph
    self.log_name = "tg:" + name + ".log"
    self.logger = open(self.log_name, "a")
    self.best_perf = 1e-10
    self.best_result = None
    self.verbose = verbose
    self.result = None
    self.counter = 0
    self.total_trials = 0

    if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
      with open(self.log_name, "r") as fin:
        print("Loading from %s..." % self.log_name, flush=True)
        for line in fin:
          obj = json.loads(line)
          entity = obj["entity"]
          perf = obj["perf"]
          entity = tg.string_to_multi_schedule_entity(entity)
          result = tg.get_schedule_result_from_entity(
            name, subgraph, self.target, entity)
          if perf > self.best_perf:
            self.best_perf = perf
            self.best_result = result
          # feedback
          tg.get_schedule_result(
            self.name, self.subgraph, self.target,
            self.measure_option.dev_id, self.measure_option.timeout,
            perf, True, result)

  def __del__(self):
    self.logger.close()

  def get_new_schedule(self):
    ret = None
    while ret is None:
      try:
        ret = tg.get_schedule_result(
          self.name, self.subgraph, self.target,
          self.measure_option.dev_id, self.measure_option.timeout)
      except Exception as e:
        if self.verbose:
          print(e)
        pass
    return ret

  def count(self):
    self.counter = (self.counter + 1) % 16
    if self.counter == 0:
      print("\n", flush=True)
      print("Currently best performance:", 1 / self.best_perf, flush=True)

  def auto_schedule(self, trials):
    print("#############################################", flush=True)
    print(self.subgraph.tag, flush=True)
    print("Autoscheduling %s by %d trials..." % (self.log_name, trials), flush=True)
    self.total_trials += trials
    beg = time.time()
    for i in range(trials):
      # this one is get new schedule
      self.result = self.get_new_schedule()
      # measure current result
      sch = self.result.schedule
      args = self.result.tensors
      timecost = at.evaluate_schedule(
        sch, args, self.measure_option, new_process=True)
      # timecost = 1.0
      perf = 1.0 / (timecost + 1e-10)
      if perf > self.best_perf:
        self.best_perf = perf
        self.best_result = self.result
        entity = tg.multi_schedule_entity_to_string(self.result.schedule_entities)
        log = {"entity": entity, "perf": perf}
        print(
          json.dumps(log),
          file=self.logger, flush=True)
        print(".B", end="", flush=True)
      elif timecost != at.MAX_FLOAT:
        print(".N", end="", flush=True)
      self.count()
      # this one is feedback
      tg.get_schedule_result(
        self.name, self.subgraph, self.target,
        self.measure_option.dev_id, self.measure_option.timeout,
        perf, True, self.result)
    end = time.time()
    print("Schedule cost %f seconds" % (end - beg))

    sch = self.result.schedule
    args = self.result.tensors
    return sch, args

  def get_best_schedule(self):
    if self.best_result is not None:
      return self.best_result.schedule, self.best_result.tensors, 1 / self.best_perf
    else:
      return None, None, at.MAX_FLOAT

  def get_measure_opt(self):
    return self.measure_option


class AnsorAutoScheduleContext(object):
  def __init__(self, name, subgraph, measure_option):
    self.name = name
    self.subgraph = subgraph
    self.measure_option = measure_option
    self.target_dag = at.compute_dag_from_tensors(
      [x.output(0) for x in subgraph.root_ops])
    task_name = name

    inputs = self.target_dag.get_inputs()
    args = inputs + list(self.target_dag.tensors)

    self.total_trials = 0

    def task_func():
        return args

    registered_func = auto_scheduler.register_workload(
        task_name, f=task_func)

    target = tvm.target.Target(measure_option.target)

    self.task = auto_scheduler.create_task(
      task_name, (), target,
      hardware_params=auto_scheduler.HardwareParams(
        1024, # cores
        16, # vector bytes
        1024, # cache line bytes
      ))

    task_name = self.task.workload_key[2:-2]
    self.log_name = "ansor:" + task_name + ".log"

    # self.measure_ctx = auto_scheduler.LocalRPCMeasureContext(
    #   priority=measure_option.priority,
    #   timeout=measure_option.timeout,
    #   number=measure_option.number,
    #   repeat=measure_option.repeat,
    #   min_repeat_ms=measure_option.min_repeat_ms,
    #   cooldown_interval=measure_option.cooldown_interval,
    #   enable_cpu_cache_flush=measure_option.enable_cpu_cache_flush)

    self.runner = auto_scheduler.LocalRunner(
        timeout=measure_option.timeout,
        number=measure_option.number,
        repeat=measure_option.repeat,
        min_repeat_ms=measure_option.min_repeat_ms,
        cooldown_interval=measure_option.cooldown_interval,
        enable_cpu_cache_flush=measure_option.enable_cpu_cache_flush)

  def auto_schedule(self, trials, model="xgb"):
    print("#############################################", flush=True)
    print(self.subgraph.tag, flush=True)
    print("Autoscheduling %s by %d trials..." % (self.log_name, trials), flush=True)
    self.total_trials += trials
    sch, args = None, None
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,
        # runner=self.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(self.log_name)],
    )

    if model == "random":
        cost_model = RandomModel()
    elif model == "xgb":
        cost_model = XGBModel()
    else:
        raise RuntimeError("Unsupported model: %s" % model)
    if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
      cost_model.update_from_file(self.log_name)
      search_policy = auto_scheduler.SketchPolicy(
          self.task, cost_model, init_search_callbacks=[
            auto_scheduler.PreloadMeasuredStates(self.log_name)]
      )
    else:
      search_policy = SketchPolicy(self.task, cost_model)
    sch, args = auto_scheduler.auto_schedule(
        self.task, search_policy=search_policy, tuning_options=tune_option)

    return sch, args

  def get_best_schedule(self):
    try:
      inp, res = auto_scheduler.load_best(self.log_name, self.task.workload_key)
      sch, args = self.task.compute_dag.apply_steps_from_state(inp.state)
      return sch, args, np.mean([float(x) * 1e3 for x in res.costs])
    except:
      return None, None, at.MAX_FLOAT

  def get_measure_opt(self):
    return self.measure_option


class AutoTensorizeContext(object):
  @classmethod
  def can_use(cls, name, subgraph, measure_option):
    target_dag = at.compute_dag_from_tensors(
      [x.output(0) for x in subgraph.root_ops])
    measure_option = measure_option
    task_name = name
    log_name = "at:" + task_name + ".log"
    match_result, new_state = at.auto_tensorize_compute(
      target_dag, measure_option.target, log_name, measure_option
    )
    return match_result is not None and new_state is not None

  def __init__(self, name, subgraph, measure_option):
    self.name = name
    self.subgraph = subgraph
    self.target_dag = at.compute_dag_from_tensors(
      [x.output(0) for x in subgraph.root_ops])
    self.measure_option = measure_option
    task_name = name
    self.log_name = "at:" + task_name + ".log"
    self.match_result, self.new_state = at.auto_tensorize_compute(
      self.target_dag, measure_option.target, self.log_name, measure_option
    )

    self.total_trials = 0

    assert self.match_result is not None
    assert self.new_state is not None

    if str(self.measure_option.target) == "cuda":
      self.schedule_gen = at.CUDAScheduleGenerator(
          self.match_result, self.new_state, log_file=self.log_name)
      if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
        self.schedule_gen.load_from_file(self.log_name)
      sc_info = self.schedule_gen.get_schedule_compute_info()
      self.schedule_app = at.CUDAScheduleApplier(self.match_result, sc_info)
      self.checker = at.CUDAProgramChecker(
        arch=at.get_cuda_compute_version(self.measure_option.dev_id))
    else:
      raise RuntimeError(
        "Do not support target: %s" % self.measure_option.target)

    self.builder = at.pebble_local_builder_build
    self.runner = at.pebble_local_runner_run

  def auto_schedule(self, trials):
    print("#############################################", flush=True)
    print(self.subgraph.tag, flush=True)
    print("Autoscheduling %s..." % self.log_name, flush=True)
    self.total_trials += trials
    if trials:
      value, params = at.find_optimized_parameters(
        self.match_result, self.schedule_gen, self.schedule_app,
        self.measure_option, self.checker, trials,  # policy="random",
        builder=self.builder,
        runner=self.runner,
        verbose=False)

    return self.get_best_schedule()

  def get_best_schedule(self):
    if self.schedule_gen.has_entry():
      entry = self.schedule_gen.get_best_entry()
      # we store 1/time_cost in file
      params = entry.record
      sch, args = at.get_schedule(self.schedule_app, params)
      return sch, args, 1 / entry.value * 1e3
    else:
      return None, None, at.MAX_FLOAT

  def get_measure_opt(self):
    return self.measure_option


class AutoScheduleGraphDispatch(object):
  working_set = {}
  results = {}

  @classmethod
  def add_task(cls, name, subgraph, measure_option,
    scheduler_option="auto_tensorize"):
    next_id = len(AutoScheduleGraphDispatch.working_set)
    if scheduler_option == "auto_tensorize":
      if AutoTensorizeContext.can_use(name, subgraph, measure_option):
        ctx = AutoTensorizeContext(name, subgraph, measure_option)
      else:
        # fallback to TG
        print("Fallback to TG")
        ctx = TGAutoScheduleContext(name, subgraph, measure_option)
    elif scheduler_option == "tg":
          ctx = TGAutoScheduleContext(name, subgraph, measure_option)
    elif scheduler_option == "ansor":
      ctx = AnsorAutoScheduleContext(name, subgraph, measure_option)
    else:
      raise RuntimeError("Unknown scheduler: %s" % scheduler_option)
    AutoScheduleGraphDispatch.working_set[next_id] = ctx
    sch, args, perf = ctx.get_best_schedule()
    # if sch is not None:
    #   perf = at.evaluate_schedule(
    #     sch, args, ctx.get_measure_opt(), new_process=True)
    # else:
    #   perf = at.MAX_FLOAT
    AutoScheduleGraphDispatch.results[next_id] = (sch, args, perf)
    return next_id, ctx

  @classmethod
  def remove_task(cls, task_id):
    if task_id in AutoScheduleGraphDispatch.working_set:
      del AutoScheduleGraphDispatch.working_set[task_id]

  @classmethod
  def auto_schedule(cls, selected_ids, trials_lst):
    for tid, trials in zip(selected_ids, trials_lst):
      if not trials:
        continue
      if tid in AutoScheduleGraphDispatch.working_set:
        ctx = AutoScheduleGraphDispatch.working_set[tid]
        sch, args = ctx.auto_schedule(trials)
        sch, args, perf = ctx.get_best_schedule()
        # if sch is not None:
        #   perf = at.evaluate_schedule(
        #     sch, args, ctx.get_measure_opt(), new_process=True)
        # else:
        #   perf = at.MAX_FLOAT
        AutoScheduleGraphDispatch.results[tid] = (sch, args, perf)

  @classmethod
  def query_schedule(cls, tid):
    if tid in AutoScheduleGraphDispatch.results:
      ctx = AutoScheduleGraphDispatch.working_set[tid]
      sch, args, perf = ctx.get_best_schedule()
      # if sch is not None:
      #   perf = at.evaluate_schedule(
      #     sch, args, ctx.get_measure_opt(), new_process=True)
      # else:
      #   perf = at.MAX_FLOAT
      print("Query subgraph task id: %s, perf=%f ms after %d tuning" % (
        str(tid), perf, ctx.total_trials), flush=True)
      return (sch, args, perf)
    else:
      return (None, None, at.MAX_FLOAT)


class AutoScheduleMultiGraphContext(object):
  def __init__(self, name, tir_multi_graph, measure_option,
      scheduler_option="auto_tensorize", gamma=0.02):
    self.tir_multi_graph = tir_multi_graph
    self.performance_trace = {}
    self.schedules = {}
    self.contexts = {}
    self.graph_tag_to_tid = {}
    self.C = {}
    self.alpha = {}
    self.beta = {}
    self.X = {}
    self.gamma = gamma
    graphs = tg.get_graphs_from_tir_multi_graph(tir_multi_graph)
    self.L = len(graphs) * 100
    graphs = OrderedDict(
      sorted([(x.value, y) for x, y in graphs.items()], key=lambda x: x[0]))
    for key, subgraph in graphs.items():
      new_name = name + ":" + str(key)
      if subgraph.tag in self.graph_tag_to_tid:
        continue
      tid, ctx = AutoScheduleGraphDispatch.add_task(
        new_name, subgraph, measure_option, scheduler_option=scheduler_option)
      sch, args, perf = AutoScheduleGraphDispatch.query_schedule(tid)
      self.performance_trace[tid] = [perf]
      self.C[tid] = perf
      self.alpha[tid] = perf / (16*16)
      self.beta[tid] = 1.0
      self.X[tid] = self.calculate_X(tid)
      self.schedules[tid] = (sch, args)
      self.contexts[tid] = ctx
      self.graph_tag_to_tid[subgraph.tag] = tid

  def calculate_X(self, tid):
    raw = math.sqrt(self.C[tid] / (self.alpha[tid] + 1e-10))
    return raw

  def select_next_tasks(self):
    # this is the decision part, currently use the simple decision
    ret = []
    trials = []
    sum_X = reduce(lambda x, y: x + y, self.X.values(), 0.0)

    for tid, lst in self.performance_trace.items():
      ret.append(tid)
      raw = int(max(0, min(self.X[tid] * self.L / sum_X, self.L)))
      trials.append(raw)
      diff = 2 * (self.C[tid] / (self.alpha[tid] * raw) + self.beta[tid] - lst[-1])
      self.alpha[tid] = self.alpha[tid] + self.gamma * diff * self.C[tid] / (
        raw * self.alpha[tid] * self.alpha[tid])
      self.beta[tid] = self.beta[tid] - self.gamma * diff
      self.C[tid] = lst[-1]
      self.X[tid] = self.calculate_X(tid)

    return ret, trials

  def auto_schedule(self):
    tids, trials = self.select_next_tasks()
    AutoScheduleGraphDispatch.auto_schedule(tids, trials)
    for k, lst in self.performance_trace.items():
      sch, args, perf = AutoScheduleGraphDispatch.query_schedule(k)
      self.schedules[k] = (sch, args)
      lst[-1] = perf  # only reserve one

  def get_schedules(self):
    ret = {}
    graphs = tg.get_graphs_from_tir_multi_graph(self.tir_multi_graph)
    graphs = OrderedDict({x.value: y for x, y in graphs.items()})
    for key, subgraph in graphs.items():
      tid = self.graph_tag_to_tid[subgraph.tag]
      sch, args = self.schedules[tid]
      ret[key] = tg.ScheduleTensors(sch, args)
    return ret


class AutoScheduleMultiGraphDispatch(object):
  working_set = {}

  @classmethod
  def add_graph_task(cls, name, tir_multi_graph, measure_option,
    scheduler_option="auto_tensorize"):
    next_id = len(AutoScheduleMultiGraphDispatch.working_set)
    AutoScheduleMultiGraphDispatch.working_set[next_id] = \
      AutoScheduleMultiGraphContext(
        name, tir_multi_graph, measure_option,
        scheduler_option=scheduler_option)
    return next_id

  @classmethod
  def auto_schedule(cls, tid):
    assert tid in cls.working_set
    ctx = cls.working_set[tid]
    ctx.auto_schedule()

  @classmethod
  def get_schedules(cls, tid):
    assert tid in cls.working_set
    ctx = cls.working_set[tid]
    return ctx.get_schedules()
