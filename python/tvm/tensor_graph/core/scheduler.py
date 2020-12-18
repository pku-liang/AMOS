import numpy as np
import heapq
import math
import logging
from collections import namedtuple

from .utils import util_cache, flatten_tir_graph
from .space import ForwardGraphSpace
from .tuner import RandomForwardTuner
from .schedule_generator import LayoutTransform, form_cut_candidates, SingleCut, \
            form_connected_sets, GPUScheduleMasterBaseSet, GPUScheduleMasterSet, \
            GPUScheduleBaseSet
from .knowledge_base import knowledge_base
from .con_graph import PyTIRGraph
from .space import PrimitiveSpace, PartitionSpace, ForwardGraphSpace
from .tuner import RandomPrimitiveTuner, RandomPartitionTuner, RandomForwardTuner


logger = logging.getLogger("tensor_graph")


class ForwardScheduler(object):
  def __init__(self, last_level=True):
    self.last_level = last_level

  def schedule(self, path_key, cur_node):
    fwd_graph, loss, opt = cur_node.inputs
    if cur_node.empty():
      forward_space = ForwardGraphSpace()
      forward_tuner = RandomForwardTuner(forward_space)
      layout_generator = LayoutTransform(
        fwd_graph, forward_space, forward_tuner)
      
      cur_node.set_space(forward_space)
      cur_node.set_tuner(forward_tuner)
      cur_node.add_generator(layout_generator)
    else:
      forward_space = cur_node.space
      forward_tuner = cur_node.tuner
      # currently only layout generator
      layout_generator, = cur_node.generators

    # learn from history


    new_graph = layout_generator.generate()


class TrialEntry(namedtuple("TrialEntry", ("choice", "perf"))):
  def __lt__(self, b):
    return self.perf > b.perf


class SubScheduler(object):
  """
  Base scheduler.

  Args:
  --------------------------
  knowledge_base: KnowledgeBase
  """
  def __init__(self, knowledge_base):
    # the L1 cache
    # the most accurate decisions for each task
    self.memory = {}
    
    # shared knowledge: cost model
    self.knowledge_base = knowledge_base
    # the L2 cache
    # the knowledge generated from general cost model
    self.knowledge = {}

    # the L3 cache
    # the most general decisions
    self.golden_advice = {}

    # specific space for each task
    self.static_spaces = {}

    # record the recent chose decision
    self.recent = {}

    # count how many times this scheduler is used
    self.counter = 0
    # the cycle to update knowledge and save model
    self.save_cycle = 20
    # how much knowledge to keep for each task
    self.topk = 20

    # if true, only use decisions in memory
    self.only_memory = False

  def clear(self):
    self.memory = {}
    self.knowledge = {}
    self.golden_advice = {}
    self.static_spaces = {}
    self.recent = {}

  def set_only_memory(self):
    self.only_memory = True

  def disable_only_memory(self):
    self.only_memory = False

  def flatten_choice(self, x):
    return x

  def flatten_args(self, x):
    return x

  def get_knowledge(self, namespace, choice_list, *args):
    # use the common cost model to evaluate all the choices
    # keep topk choices as knowledge
    lst = self.knowledge_base.query_entry_list(choice_list, self.flatten_choice, *self.flatten_args(args))
    lst = sorted(zip(choice_list, lst), key=lambda x: x[1])
    self.knowledge[args] = list(map(lambda x: x[0], lst))[:self.topk]

  def get_golden_advice(self, namespace, choice_list, *args):
    # this is expert advice
    raise NotImplementedError()

  def get_static_space(self, namespace, *args):
    # get the static space
    raise NotImplementedError()

  def calculate_priority(self, namespace, *args):
    # calculate the priority of each cache
    if self.only_memory:
      return 1.0, 0.0, 0.0
    
    def func1(x):
      return 1 / (math.exp(x) + 1)

    def func2(x):
      return 2 / (math.exp(-x) + 1) - 1

    len_memory = func2(len(self.memory[args]))
    len_knowledge = func1(self.knowledge_base.get_loss(*args))
    len_golden = func2(len(self.golden_advice[args]))
    total = float(len_memory + len_knowledge + len_golden)
    return len_memory / total, len_knowledge / total, len_golden / total

  def propose(self, namespace, *args):
    if args not in self.static_spaces:
      self.get_static_space(namespace, *args)
    space = self.static_spaces[args]

    if args not in self.knowledge or self.counter % self.save_cycle == 0:
      # update knowledge
      self.get_knowledge(namespace, space, *args)
    
    if args not in self.golden_advice:
      self.get_golden_advice(namespace, space, *args)
    
    if args not in self.memory:
      self.memory[args] = []

    p1, p2, p3 = self.calculate_priority(namespace, *args)

    # logger.debug("possibility: %f, %f, %f" % (p1, p2, p3))

    rand = np.random.random()
    if rand <= p1:
      # choose memory
      # the top of memory heap
      # logger.debug("use memory")
      ret = self.memory[args][0].choice
    elif rand <= p2 + p1:
      # choose knowledge
      # logger.debug("use knowledge")
      choice = np.random.randint(0, len(self.knowledge[args]))
      ret = self.knowledge[args][choice]
    else:
      # choose golden
      # logger.debug("use golden")
      choice = np.random.randint(0, len(self.golden_advice[args]))
      ret = self.golden_advice[args][choice]
    self.recent[namespace] = (args, ret)
    return ret

  def propose_random(self, namespace, *args):
    if args not in self.static_spaces:
      self.get_static_space(namespace, *args)
    space = self.static_spaces[args]
    ind = np.random.randint(0, len(space))
    ret = space[ind]
    self.recent[namespace] = (args, ret)
    return ret

  def feedback(self, namespace, perf):
    if namespace in self.recent and len(self.recent[namespace]) > 0:
      args = self.recent[namespace][0]
      heapq.heappush(self.memory[args], TrialEntry(choice=self.recent[namespace][-1], perf=perf))
      args, choice = self.recent[namespace]
      self.knowledge_base.add_entry(perf, self.flatten_args(args), self.flatten_choice(choice))
    
    self.counter += 1
    if self.counter % self.save_cycle == 0:
      knowledge_base.store_allreduce_base()


class AllReduceScheduler(SubScheduler):
  """
  Scheduler for allreduce strategy.

  Args:
  --------------------------
  knowledge_base: KnowledgeBase
  """
  def __init__(self, knowledge_base):
    super(AllReduceScheduler, self).__init__(knowledge_base)

  def flatten_choice(self, x):
    return (*x[0], x[1])

  def flatten_args(self, x):
    return x

  def get_golden_advice(self, namespace, choice_list, num_loops, extent, flops):
    # this is expert advice
    args = (num_loops, extent, flops)
    golden = []
    if extent <= 16:
      golden.extend([[[math.ceil(extent/16), 16], 1], [[16, math.ceil(extent/16)], 0]])
    else:
      golden.extend([[[math.ceil(extent/32), 32], 1], [[32, math.ceil(extent/32)], 0]])
    self.golden_advice[args] = golden

  def get_static_space(self, namespace, num_loops, extent, flops):
    # get the static space
    args = (num_loops, extent, flops)
    split_list = util_cache.query_split(extent, 2)
    use_factor_list = [0, 1]
    space = []
    for use in use_factor_list:
      # do not allow allreduce on thread extent 1
      def _func(x):
        if use == 1:
          return x[1] != 1
        else:
          return x[0] != 1
      space.extend(list(map(lambda x: [x, use], filter(_func, split_list))))
    self.static_spaces[args] = space
    

class DecompositionScheduler(SubScheduler):
  """
  Scheduler for thread block decomposition.

  Args:
  --------------------------
  knowledge_base: KnowledgeBase
  """
  def __init__(self, knowledge_base):
    super(DecompositionScheduler, self).__init__(knowledge_base)
    self.num_golden = 20

  def flatten_choice(self, x):
    ret = []
    for lst in x:
      ret.extend(lst)
    return ret

  def flatten_args(self, x):
    return (*x[0], *x[1:])

  def get_golden_advice(self, namespace, choice_list,
              extent_list, outer_extent, num_inputs, is_allreduce):
    # this is expert advice
    args = (extent_list, outer_extent, num_inputs, is_allreduce)
    golden = []
    
    if is_allreduce == 1:
      expect = [80 * 2, 1]
      def _func(x):
        blocks = 1
        for lst in x:
          blocks *= lst[0]
        return math.pow(blocks - expect[0], 2)
      
      metric = _func
    else:
      expect = [80 * 2, 1, 1024, 1]
      def _func(x):
        blocks = 1
        threads = 1
        for lst in x:
          blocks *= lst[0]
          threads *= lst[2]
        return math.pow(blocks - expect[0], 2) + math.pow(threads - expect[2], 2)
      
      metric = _func

    scores = map(metric, choice_list)
    pairs = zip(scores, choice_list)
    after_sort = sorted(pairs, key=lambda x: x[0])
    
    golden = [x[1] for x in after_sort[:self.num_golden]]

    self.golden_advice[args] = golden

  def get_static_space(self, namespace, 
          extent_list, outer_extent, num_inputs, is_allreduce):
    # get the static space
    args = (extent_list, outer_extent, num_inputs, is_allreduce)
    if is_allreduce == 1:
      decompose_list = util_cache.query_decompose(extent_list, 2)
      space = decompose_list
    else:
      decompose_list = util_cache.query_decompose(extent_list, 4)
      def _func(x):
        threads = 1
        for lst in x:
          threads *= lst[2]
        return threads <= 1024
      space = list(filter(_func, decompose_list))

    self.static_spaces[args] = space


class ReductiveScheduler(SubScheduler):
  """
  Scheduler for reductive axis decomposition.

  Args:
  --------------------------
  knowledge_base: KnowledgeBase
  """
  def __init__(self, knowledge_base):
    super(ReductiveScheduler, self).__init__(knowledge_base)
    self.num_golden = 20

  def flatten_choice(self, x):
    ret = []
    for lst in x:
      ret.extend(lst)
    return ret

  def flatten_args(self, x):
    return (*x[0], *x[1:])

  def get_golden_advice(self, namespace, choice_list,
              extent_list, flops, num_inputs):
    # this is expert advice
    args = (extent_list, flops, num_inputs)
    golden = []
    
    def _func(x):
      
      for lst in x:
        diff = lst[0] + lst[2] - 2 * lst[1]
      return diff * diff
      
    metric = _func

    scores = map(metric, choice_list)
    pairs = zip(scores, choice_list)
    after_sort = sorted(pairs, key=lambda x: x[0])
    
    golden = [x[1] for x in after_sort[:self.num_golden]]

    self.golden_advice[args] = golden

  def get_static_space(self, namespace, 
          extent_list, flops, num_inputs):
    # get the static space
    args = (extent_list, flops, num_inputs)

    space = util_cache.query_decompose(extent_list, 3)

    self.static_spaces[args] = space


class CachePosScheduler(SubScheduler):
  def __init__(self):
    pass


class UnrollScheduler(SubScheduler):
  def __init__(self):
    pass


class PrimitiveScheduler(object):
  def __init__(self, random=False):
    self.allreduce_scheduler = AllReduceScheduler(knowledge_base.get_allreduce_base())
    self.decomposition_scheduler = DecompositionScheduler(knowledge_base.get_decomposition_base())
    self.reductive_scheduler = ReductiveScheduler(knowledge_base.get_reductive_base())
    self.cachepos_scheduler = CachePosScheduler()
    self.unroll_scheduler = UnrollScheduler()

    self.random = random

  def schedule_allreduce(self, namespace, num_loops, extent, flops):
    if self.random:
      return self.allreduce_scheduler.propose_random(namespace, num_loops, extent, flops)
    return self.allreduce_scheduler.propose(namespace, num_loops, extent, flops)

  def schedule_decomposition(self, namespace, extent_list, outer_extent, num_inputs, is_allreduce):
    if self.random:
      return self.decomposition_scheduler.propose_random(namespace, extent_list, outer_extent, num_inputs, is_allreduce)
    return self.decomposition_scheduler.propose(namespace, extent_list, outer_extent, num_inputs, is_allreduce)

  def schedule_reductive(self, namespace, extent_list, flops, num_inputs):
    if self.random:
      return self.reductive_scheduler.propose_random(namespace, extent_list, flops, num_inputs)
    return self.reductive_scheduler.propose(namespace, extent_list, flops, num_inputs)

  def feedback(self, namespace, perf):
    self.allreduce_scheduler.feedback(namespace, perf)
    self.decomposition_scheduler.feedback(namespace, perf)
    self.reductive_scheduler.feedback(namespace, perf)

  def show_recent(self):
    print("############ allreduce ############")
    for k, v in self.allreduce_scheduler.recent.items():
      print("name=", k, "args=", v[0], "choice=", v[1])
    
    print("############ decomposition ############")
    for k, v in self.decomposition_scheduler.recent.items():
      print("name=", k, "args=", v[0], "choice=", v[1])

    print("############ reductive ############")
    for k, v in self.reductive_scheduler.recent.items():
      print("name=", k, "args=", v[0], "choice=", v[1])

  # def __del__(self):
  #   knowledge_base.store_allreduce_base()
  #   knowledge_base.store_decomposition_base()
  #   knowledge_base.store_reductive_base()


global_primitive_scheduler = PrimitiveScheduler(random=True)


def schedule_all(fwd_graph, loss=None, optimizer=None, inference=True):
  prefix = "graph_" + str(id(fwd_graph))
  ########################################################
  # change data layout
  # forward_space = ForwardGraphSpace()
  # forward_tuner = RandomForwardTuner(forward_space)

  # layout_generator = LayoutTransform(
  #   fwd_graph, forward_space, forward_tuner)
  # fwd_graph = layout_generator.generate()
  import time
  beg = time.time()

  if inference:
    finputs, foutputs, fweights = fwd_graph()

    inputs = [x.tvm_tensor for x in finputs]
    weights = [x.tvm_tensor for x in fweights]
    outputs = [x.tvm_tensor for x in foutputs]
    labels = []
    loss = None
    gradients = []
    lr = None
    updates = []
  else:
    assert loss is not None and optimizer is not None
    bgraph = fwd_graph.make_backward(loss, optimizer)

    inputs = [x.tvm_tensor for x in bgraph.inputs]
    weights = [x.tvm_tensor for x in bgraph.weights]
    outputs = [x.tvm_tensor for x in bgraph.outputs]
    labels = [x.tvm_tensor for x in bgraph.labels]
    loss = bgraph.loss.tvm_tensor
    gradients = [x.tvm_tensor for x in bgraph.gradients]
    lr = optimizer.lr_tensor
    updates = [x.tvm_tensor for x in bgraph.updates]

  end = time.time()
  print("make backward graph takes", (end - beg) * 1e3, "ms")

  ########################################################
  # make tir graph

  tgraph = PyTIRGraph(
    inputs,
    labels,
    outputs,
    weights,
    loss,
    gradients,
    lr,
    updates)

  ########################################################
  # subgraph partition
  partition_space = PartitionSpace()
  partition_tuner = RandomPartitionTuner(partition_space)

  cut_candidates = form_cut_candidates(tgraph)

  for i, candidate in enumerate(cut_candidates):
    name = prefix + "graph_cut_" + str(i)
    partition_generator = SingleCut(
      tgraph, name, candidate, partition_space, partition_tuner)
    partition_generator.generate()

  tgraph.partition_graph()

  end = time.time()
  print("prepare takes", (end - beg) * 1e3, "ms")

  ########################################################
  # update the op stat dict of subgraphs
  # do auto-schedule
  scheduler = global_primitive_scheduler

  for mark, subgraph in tgraph.subgraphs.items():
    c_list = form_connected_sets(
      subgraph, subgraph.op_stat_dict, subgraph.tensors, subgraph.ops, subgraph.down_graph)

    subgraph.c_list = c_list

    need_schedule = tgraph.create_schedule_for(mark=mark)
    if not need_schedule:
      continue
    sch = tgraph.schedules[mark]
    
    for i, connected_set in enumerate(c_list):
      name = prefix + "subgraph_" + str(mark) + "_connect_" + str(i)
      subgraph.connected_sets[name] = connected_set
      assert not connected_set.empty()
      
      if connected_set.has_master():
        if connected_set.iso_base():
          ScheduleGenerator = GPUScheduleMasterBaseSet
        else:
          ScheduleGenerator = GPUScheduleMasterSet

        primitive_generator = ScheduleGenerator(
          name, subgraph, connected_set, subgraph.down_graph, subgraph.op_stat_dict, scheduler)
      else:
        ScheduleGenerator = GPUScheduleBaseSet
        primitive_generator = ScheduleGenerator(
          name, connected_set, scheduler)

      primitive_generator.generate(sch)
  
  # scheduler.show_recent()
  
  return tgraph


def reschedule_subgraph(fwd_graph, tir_graph, mark):
  prefix = "graph_" + str(id(fwd_graph))

  ########################################################
  # update the op stat dict of subgraphs
  # do auto-schedule
  scheduler = global_primitive_scheduler

  subgraph = tir_graph.subgraphs[mark]
  c_list = subgraph.c_list

  tir_graph.create_schedule_for(mark=mark, force=True)

  sch = tir_graph.schedules[mark]
  
  for i, connected_set in enumerate(c_list):
    name = prefix + "subgraph_" + str(mark) + "_connect_" + str(i)
    subgraph.connected_sets[name] = connected_set
    assert not connected_set.empty()
    
    if connected_set.has_master():
      if connected_set.iso_base():
        ScheduleGenerator = GPUScheduleMasterBaseSet
      else:
        ScheduleGenerator = GPUScheduleMasterSet

      primitive_generator = ScheduleGenerator(
        name, subgraph, connected_set, subgraph.down_graph, subgraph.op_stat_dict, scheduler)
    else:
      ScheduleGenerator = GPUScheduleBaseSet
      primitive_generator = ScheduleGenerator(
        name, connected_set, scheduler)

    primitive_generator.generate(sch)
  
  # scheduler.show_recent()
  