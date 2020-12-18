import tvm
import math

from functools import reduce

from .utils import flatten_tir_graph, to_int, power_of_x_near
from .transform import LayoutChangeFinder, LayoutChangeApplier
from .transform import ParallelFusionFinder, ParallelFusionApplier


class ForwardGenerator(object):
  def __init__(self):
    pass

  def generate(self):
    raise NotImplementedError()


class LayoutTransform(ForwardGenerator):
  def __init__(self, fwd_graph, space, tuner):
    self.fwd_graph = fwd_graph
    self.space = space
    self.tuner = tuner
    lcf = LayoutChangeFinder()
    lcf(fwd_graph)
    self.batch_like_dim_dict = {}
    self.max_dim = 0
    for (k, v) in lcf.batch_like_dim_dict.items():
      print("batch", k, v)
      self.batch_like_dim_dict[k] = list(sorted(list(set(v))))
      self.max_dim = max(len(self.batch_like_dim_dict[k]), self.max_dim)
    self.lca = LayoutChangeApplier(self.batch_like_dim_dict, level=self.max_dim, order="des")

  def generate(self):
    return self.transform()

  def transform(self):
    self.space.add_layout(self.max_dim)
    level, order = self.tuner.propose_layout()
    self.lca.level = level
    self.lca.order = order
    return self.lca(self.fwd_graph)


# TODO: implement ParallelFusion
# class ParallelFusion(ForwardGenerator):
#   def __init__(self, fwd_graph, space, tuner):
#     raise NotImplementedError()
#     # super(ParallelFusion).__init__()
#     # self.fwd_graph = fwd_graph
#     # self.space = space
#     # self.tuner = tuner
#     # pff = ParallelFusionFinder()
#     # pff(fwd_graph)
#     # self.pfa = ParallelFusionApplier()
#
#   def generate(self):
#     return self.transform()
#
#   def transform(self):
#     self.space.add_layout(None)
#     level, order = self.tuner.propose_layout()
#     return self.pfa(self.fwd_graph)


class CutCandidate(object):
  def __init__(self):
    self.op_list = []
    self.op_set = set()
    self.base = None

  def add(self, op):
    if self.base is None:
      self.base = op
    self.op_list.append(op)
    self.op_set.add(op)

  def empty(self):
    return len(self.op_list) == 0

  def size(self):
    return len(self.op_list)

  def __getitem__(self, index):
    return self.op_list[index]

  def has(self, op):
    return op in self.op_set

  def __repr__(self):
    return "CutCandidate"

  def __str__(self):
    return self.__repr__()


# for a partition generator, it handles a whole graph.
def form_cut_candidates(tir_graph):
  """Form a list of cut candidate.

    Parameters
    ----------
    tir_graph : PyTIRGraph
        The graph from which cut candidate are formed.
    
    Returns
    -------
    list of CutCandidate
        The formed list of CutCandidate.
  """
  if tir_graph.loss is not None:
    loss = [tir_graph.loss]
  else:
    loss = []
  root_tensors = tir_graph.outputs + loss + \
    tir_graph.gradients + tir_graph.updates
  root_ops = [x.op for x in root_tensors]

  ret = []

  def _must_cut(cur):
    if not isinstance(cur, tvm.te.tensor.ComputeOp):
      return False
    num_consumer = 0
    for i in range(cur.num_outputs):
      if cur.output(i) in tir_graph.down_graph:
        num_consumer += len(tir_graph.down_graph[cur.output(i)])
    if num_consumer > 1:
      return True

  visited = set()
  
  def _helper(cur, cut_candidate):
    if not isinstance(cur, tvm.te.tensor.ComputeOp):
      return
    if cur in visited:
      return
    visited.add(cur)
    if tir_graph.op_stat_dict[cur].injective:
      if _must_cut(cur):
        cut_candidate = CutCandidate()
        cut_candidate.add(cur)
        ret.append(cut_candidate)
      else:
        cut_candidate.add(cur)
      for t in cur.input_tensors:
        _helper(t.op, cut_candidate)
    else:
      for t in cur.input_tensors:
        cut_candidate = CutCandidate()
        ret.append(cut_candidate)
        _helper(t.op, cut_candidate) 

  for op in root_ops:
    cut_candidate = CutCandidate()
    ret.append(cut_candidate)
    _helper(op, cut_candidate)

  non_empty = list(filter(lambda x: not x.empty(), ret))
  return non_empty


class PartitionGenerator(object):
  def __init__(self):
    pass

  def generate(self):
    raise NotImplementedError()


class SingleCut(PartitionGenerator):
  def __init__(self, tir_graph, name, cut_candidate, space, tuner):
    self.tir_graph = tir_graph
    self.name = name
    self.cut_candidate = cut_candidate
    self.space = space
    self.tuner = tuner

  def generate(self):
    self.make_cut()

  def make_cut(self):
    num_candidate = self.cut_candidate.size()
    self.space.add_partition(self.name, num_candidate, default=[num_candidate])
    choice = self.tuner.propose_partition(self.name)
    if choice == num_candidate:
      # no cut
      pass
    else:
      cut_point = self.cut_candidate[choice]
      
      visited = set()

      def _helper(cur, head):
        if cur in visited:
          return
        visited.add(cur)
        if cur == cut_point:
          self.tir_graph.op_stat_dict[cur].head = False
          head = False
        for t in cur.input_tensors:
          if self.cut_candidate.has(t.op):
            _helper(t.op, head)

      _helper(self.cut_candidate.op_list[0], True)
      

class ConnectedSet(object):
  """
  prologue can be {}
  master can be []
  epilogue can be []

  the order of prologue and epilogue has no effect on schedules
  """
  def __init__(self, prologue, master, epilogue, base):
    assert isinstance(prologue, dict)
    assert isinstance(epilogue, (list, tuple))
    self.inputs = {}
    self.prologue = prologue
    self.master = master
    self.epilogue = epilogue
    self.base = base

  def has_master(self):
    return len(self.master) > 0

  def iso_base(self):
    return not (len(self.master) == 1 and self.master[0] == self.base)

  def empty(self):
    return self.base is None

  def __repr__(self):
    ret = "ConnectedSet\n"
    ret += "prologue=" + str(self.prologue) + "\n"
    ret += "master=" + str(self.master) + "\n"
    ret += "epilogue=" + str(self.epilogue) + "\n"
    ret += "base=" + str(self.base) + "\n"
    return ret

  def __str__(self):
    return self.__repr__()

# for a primitive generator, it handles a subgraph.
# In a subgraph, every two pair of adjacent nodes
# may have a relation that indicates that they should
# be fused together.
def form_connected_sets(subgraph, op_stat_dict, tensors, ops, down_graph):
  """Form a list of connected set.

    Parameters
    ----------
    subgraph     : TIRSubgraph
        The subgraph from which connected sets are formed.

    op_stat_dict : dict from tvm.te.tensor.ComputeOp to PyOpState
        Used to check attributes of an operation.

    tensors      : list of tvm.te.tensor.Tensor

    ops          : list of tvm.te.tensor.ComputeOp

    down_graph:  : dict from tvm.te.tensor.Tensor to 
        list of tvm.te.tensor.ComputeOp
        Source tensors to their consumers.
    
    Returns
    -------
    list of ConnectedSet
        The formed list of ConnectedSet, usually of size 1.
  """
  def _is_root(op, down_graph):
    is_root = True
    for i in range(op.num_outputs):
      if op.output(i) in down_graph:
        is_root = False
        break
    return is_root
  # these ops can be base nodes
  root_ops = filter(lambda x: _is_root(x, down_graph), ops)

  connected_set_list = []

  def can_fuse(pre_stat, post_stat):
    if pre_stat.num_consumers > 1:
      # do not fuse multi-output
      return False
    if pre_stat.reductive and post_stat.reductive:
      # do not fuse reductive nodes
      return False
    if pre_stat.injective and post_stat.injective:
      return not ((not pre_stat.head) and post_stat.head)
    if pre_stat.injective and post_stat.reductive:
      return pre_stat.head
    if pre_stat.reductive and post_stat.injective:
      return not post_stat.head
    return False

  visited = set()

  def helper(cur, connected_set, cur_master=None):
    if cur in visited:
      return
    visited.add(cur)

    add_base = False

    if cur not in op_stat_dict:
      return
    # we use helper from bottom to up
    # so it should be the base node
    if connected_set.base is None:
      add_base = True
      connected_set.base = cur
    
    if op_stat_dict[cur].reductive:
      connected_set.master.append(cur)
      cur_master = cur
    elif not add_base:
      if cur_master is not None:
        connected_set.prologue[cur] = cur_master
      else:
        connected_set.epilogue.append(cur)
    #   if connected_set.master is None:
    #     if op_stat_dict[cur].reductive:
    #       # this is the master
    #       connected_set.master = cur
    # elif connected_set.master is None:
    #   if op_stat_dict[cur].reductive:
    #     # this is the master
    #     connected_set.master = cur
    #   else:
    #     # this is injective
    #     # we are still in epilogue
    #     connected_set.epilogue.append(cur)
    # else:
    #   # we are in prologue
    #   connected_set.prologue.append(cur)
    
    # propagate up
    for t in cur.input_tensors:
      if t.op in op_stat_dict:
        pre_stat = op_stat_dict[t.op]
        post_stat = op_stat_dict[cur]
        if can_fuse(pre_stat, post_stat):
          helper(t.op, connected_set, cur_master=cur_master)
        else:
          new_set = ConnectedSet({}, [], [], None)
          connected_set_list.append(new_set)
          helper(t.op, new_set, cur_master=cur_master)
      elif isinstance(t.op, tvm.te.tensor.PlaceholderOp):
        if cur_master is not None:
          if t not in connected_set.inputs:
            connected_set.inputs[t] = []
          connected_set.inputs[t].append(cur_master)
        else:
          if t not in connected_set.inputs:
            connected_set.inputs[t] = []
          connected_set.inputs[t].append(connected_set.base)
  
  for op in root_ops:
    new_set = ConnectedSet({}, [], [], None)
    connected_set_list.append(new_set)
    helper(op, new_set, cur_master=None)

  non_empty = list(filter(lambda x: not x.empty(), connected_set_list))
  return non_empty


class PrimitiveGenerator(object):
  def __init__(self, connected_set, scheduler):
    self.sch = None
    self.connected_set = connected_set
    self.scheduler = scheduler

  def generate(self):
    raise NotImplementedError()


def decide_allreduce(subgraph, connected_set, down_graph, ratio=2.0):
  base = connected_set.base
  masters = connected_set.master
  if not connected_set.has_master():
    return False
  
  allreduce = True

  for i, master in enumerate(masters):
    spatial = 1
    reduction = 1
    for axis in base.axis:
      ext = to_int(axis.dom.extent)
      spatial *= ext
    for axis in master.reduce_axis:
      ext = to_int(axis.dom.extent)
      reduction *= ext
    if reduction < ratio * spatial:
      allreduce = False
      return allreduce

  if len(masters) > 1:
    return False
  
  return allreduce

  
class GPUScheduleMasterBaseSet(PrimitiveGenerator):
  """
  Schedule generator for connected set that
  contains both master nodes and base node.

  Args:
  --------------------------
  name: string
    the namespace of this generator,
    used for schedule space knob
  
  subgraph: PyTIRSubGraph
  
  connected_set: ConnectedSet

  down_graph: dict of tensor to list of ops
    this is used to get the consumer
    operators of a tvm tensor

  op_stat_dict: dict of operation to state

  scheduler: Scheduler
  """
  def __init__(self, name, subgraph, connected_set, down_graph, op_stat_dict, scheduler):
    super(GPUScheduleMasterBaseSet, self).__init__(connected_set, scheduler)
    self.name = name
    self.subgraph = subgraph
    self.op_stat_dict = op_stat_dict
    self.down_graph = down_graph

    # read caches, tensor->cache
    self.read_shared_caches = {}
    self.read_local_caches = {}

    self.schedule_allreduce = decide_allreduce(
      subgraph, connected_set, down_graph)
    # such case we take conservative decisions
    # this decision is coupled with the following
    # schedules, so be careful when changing this
    # decision
    if self.schedule_allreduce:
      if len(connected_set.master) != 1:
        self.schedule_allreduce = False

    # kernel scope
    self.kernel_scope = None
    # spatial axis
    self.bx = None
    self.by = None
    self.bz = None
    self.vx = None
    self.vy = None
    self.vz = None
    self.tx = None
    self.ty = None
    self.tz = None
    self.ix = None
    self.iy = None
    self.iz = None
    # the extent of threads
    self.ext_tx = -1
    self.ext_ty = -1
    self.ext_tz = -1
    # reduce axis
    self.rx_list = {}
    self.ry_list = {}
    self.rz_list = {}
    # if we do rfactor
    self.rf = None

  def generate(self, sch):
    """
    generate the whole schedule primitives
    for the given schedule
    """
    # use the given schedule
    self.sch = sch
    self.create_cache()
    self.thread_block_decomposition()
    self.schedule_reductive()
    self.cache_fetch()
    self.fuse_prologue()
    self.fuse_epilogue()
    # self.unroll_loop()

  def create_cache(self):
    """
    prepare shared/local read/write cache
    for parallel reduction, we don't use
    cache currently as the support in tvm
    is not general
    """
    if self.schedule_allreduce:
      # currently, we don't consider cache for allreduce
      # prepare thread axis
      tx = tvm.te.thread_axis("threadIdx.x")

      # prepare master and base
      masters = self.connected_set.master
      base = self.connected_set.base

      # according to the decision of allreduce
      # we know there is only one master here
      for i, master in enumerate(masters):
        reduce_axis = self.sch[master].op.reduce_axis
        # only rfactor the biggest reduce axis
        to_sort = zip(reduce_axis, [to_int(x.dom.extent) for x in reduce_axis])
        after_sort = list(sorted(to_sort, key=lambda x: x[1]))
        # axis is the dim that has the largest extent
        axis = after_sort[-1][0]

        # filter out extent=1
        # prefer extent=32

        # self.space.add_split(self.name, "tile_k", axis, nparts=2, filters=[[1, 1]], default=[[-1, 32], [32, -1]])
        # factors = self.tuner.propose_split(self.name, "tile_k")

        # whether use inner axis to do allreduce

        # self.space.add_rfactor(self.name)
        # use_factor = self.tuner.propose_rfactor(self.name)

        num_loops = len(reduce_axis)
        extent = to_int(axis.dom.extent)
        stat = self.op_stat_dict[master]
        # self.num_add = 0
        # self.num_mul = 0
        # self.num_div = 0
        # self.num_branch = 0
        # self.num_logic = 0
        # self.num_special = 0
        flops = stat.num_add + stat.num_mul + stat.num_div
        factors, use_factor = self.scheduler.schedule_allreduce(self.name, num_loops, extent, flops)

        # TODO: why rfactor only takes tensor as inputs?
        if use_factor == 1:
          outer, inner = self.sch[master].split(axis, factor=factors[1])
          MF = self.sch.rfactor(master.output(0), inner)
          self.ext_tx = factors[1]
        else:
          outer, inner = self.sch[master].split(axis, nparts=factors[0])
          MF = self.sch.rfactor(master.output(0), outer)
          self.ext_tx = factors[0]
        
        # record the important attributes
        self.rf = MF
        self.tx = self.sch[master].op.reduce_axis[0]

        # parallel reduction
        self.sch[master].bind(self.tx, tx)
        self.sch[MF].compute_at(self.sch[master], self.tx)
        self.sch[master].set_store_predicate(tx.var.equal(0))
        self.sch[base].set_store_predicate(tx.var.equal(0))
    else:
      # for all the input data, create cache
      # for inp in self.subgraph.inputs.keys():
      #   if inp in self.connected_set.inputs and len(self.connected_set.inputs[inp]) > 1:
      #     # one input is shared by many consumers
      #     # TODO: how to use compute at in such case?
      #     # currently, we don't use cache for it
      #     continue
      #   inp_shared = self.sch.cache_read(inp, "shared", self.down_graph[inp])
      #   inp_local = self.sch.cache_read(inp_shared, "local", self.down_graph[inp])
      #   self.read_shared_caches[inp] = inp_shared
      #   self.read_local_caches[inp] = inp_local
      
      # the reductive op is regarded as output cache
      masters = self.connected_set.master
      for master in masters:
        self.sch[master].set_scope("local")

  def thread_block_decomposition(self):
    # prepare thread axis
    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    vx = tvm.te.thread_axis("vthread")
    vy = tvm.te.thread_axis("vthread")
    vz = tvm.te.thread_axis("vthread")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    # TODO: the logic of allreduce and non-allreduce are quite
    # similar, we should write cleaner code in future.
    # For now, just use the long code blocks.
    if self.schedule_allreduce:
      # prepare the base
      base = self.connected_set.base
      org_spatial_axis = self.sch[base].op.axis
      num_inputs = len(base.input_tensors)

      # do not care about axis with dim = 1
      # spatial_axis = list(filter(lambda x: to_int(x.dom.extent) > 1, spatial_axis))
      spatial_axis = []
      left_axis = []
      for x in org_spatial_axis:
        if to_int(x.dom.extent) > 1:
          spatial_axis.append(x)
        else:
          left_axis.append(x)
      
      # make the kernel scope
      # kernel_scope = spatial_axis[0]
      # kernel_scope, left = self.sch[base].split(kernel_scope, nparts=1)
      # spatial_axis[0] = left
      # self.kernel_scope = kernel_scope

      num_dim = len(spatial_axis)
      if num_dim == 0:
        ox, ix = self.sch[base].split(self.sch[base].op.axis[0], nparts=1)
        vx, ix = self.sch[base].split(ix, nparts=1)
        tx, ix = self.sch[base].split(ix, nparts=1)
        self.bx = ox
        self.vx = vx
        self.tx = tx
        self.ix = ix
        self.sch[base].bind(self.bx, bx)
      elif num_dim == 1:
        # self.space.add_split(self.name, "tile_x", spatial_axis[0], nparts=2)
        # factors = self.tuner.propose_split(self.name, "tile_x")

        ext_x = to_int(spatial_axis[0].dom.extent)
        factors, = self.scheduler.schedule_decomposition(
                        self.name, (ext_x,), 1, num_inputs, 1)

        split_axis = []
        axis = spatial_axis[0]
        for f in factors[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis.append(outer)
        split_axis.append(axis)
        self.bx = split_axis[0]
        self.ix = split_axis[1]
        self.sch[base].bind(self.bx, bx)
      elif num_dim == 2:
        # self.space.add_split(self.name, "tile_y", spatial_axis[0], nparts=2)
        # self.space.add_split(self.name, "tile_x", spatial_axis[1], nparts=2)
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_y = to_int(spatial_axis[0].dom.extent)
        ext_x = to_int(spatial_axis[1].dom.extent)
        factors_x, factors_y = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y), 1, num_inputs, 1)

        split_axis_y = []
        axis = spatial_axis[0]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = spatial_axis[1]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.by = split_axis_y[0]
        self.iy = split_axis_y[1]
        self.bx = split_axis_x[0]
        self.ix = split_axis_x[1]
        self.sch[base].reorder(self.by, self.bx, *left_axis,
                               self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.by, by)
      elif num_dim == 3:
        # self.space.add_split(self.name, "tile_z", spatial_axis[0], nparts=2)
        # self.space.add_split(self.name, "tile_y", spatial_axis[1], nparts=2)
        # self.space.add_split(self.name, "tile_x", spatial_axis[2], nparts=2)
        # factors_z = self.tuner.propose_split(self.name, "tile_z")
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_z = to_int(spatial_axis[0].dom.extent)
        ext_y = to_int(spatial_axis[1].dom.extent)
        ext_x = to_int(spatial_axis[2].dom.extent)
        factors_x, factors_y, factors_z = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y, ext_z), 1, num_inputs, 1)

        split_axis_z = []
        axis = spatial_axis[0]
        for f in factors_z[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_z.append(outer)
        split_axis_z.append(axis)
        split_axis_y = []
        axis = spatial_axis[1]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = spatial_axis[2]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.bz = split_axis_z[0]
        self.iz = split_axis_z[1]
        self.by = split_axis_y[0]
        self.iy = split_axis_y[1]
        self.bx = split_axis_x[0]
        self.ix = split_axis_x[1]
        self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                               self.iz, self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.bz, bz)
      else:
        # the dim number > 3
        # find the smallest dims
        to_sort = zip(
          spatial_axis, [to_int(x.dom.extent) for x in spatial_axis])
        # ascending
        after_sort = list(sorted(to_sort, key=lambda x: x[1]))
        outer_extent = reduce(lambda x, y: x * y, [x[1] for x in after_sort[:-2]], 1)
        after_sort = [x[0] for x in after_sort]
        # fuse the small axis together
        # parallel them through block z dim
        self.sch[base].reorder(*after_sort)
        fuse_axis = self.sch[base].fuse(*after_sort[:-2])

        # self.space.add_split(self.name, "tile_y", spatial_axis[-2], nparts=2)
        # self.space.add_split(self.name, "tile_x", spatial_axis[-1], nparts=2)
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_y = to_int(spatial_axis[-2].dom.extent)
        ext_x = to_int(spatial_axis[-1].dom.extent)
        factors_x, factors_y = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y), outer_extent, num_inputs, 1)

        split_axis_y = []
        axis = after_sort[-2]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = after_sort[-1]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.bz = fuse_axis
        self.by = split_axis_y[0]
        self.iy = split_axis_y[1]
        self.bx = split_axis_x[0]
        self.ix = split_axis_x[1]
        self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                               self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.bz, bz)
    else: # non-allreduce
      base = self.connected_set.base
      org_spatial_axis = self.sch[base].op.axis
      num_inputs = len(base.input_tensors)
      
      # do not care about axis with dim = 1
      # spatial_axis = list(filter(lambda x: to_int(x.dom.extent) > 1, spatial_axis))
      spatial_axis = []
      left_axis = []
      for x in org_spatial_axis:
        if to_int(x.dom.extent) > 1:
          spatial_axis.append(x)
        else:
          left_axis.append(x)
      
      # make the kernel scope
      # kernel_scope = spatial_axis[0]
      # kernel_scope, left = self.sch[base].split(kernel_scope, nparts=1)
      # spatial_axis[0] = left
      # self.kernel_scope = kernel_scope
      num_dim = len(spatial_axis)
      if num_dim == 0:
        ox, ix = self.sch[base].split(self.sch[base].op.axis[0], nparts=1)
        vx, ix = self.sch[base].split(ix, nparts=1)
        tx, ix = self.sch[base].split(ix, nparts=1)
        self.bx = ox
        self.vx = vx
        self.tx = tx
        self.ix = ix
        self.sch[base].bind(self.bx, bx)
      elif num_dim == 1:
        # self.space.add_split(
        #   self.name, "tile_x", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
        # factors = self.tuner.propose_split(self.name, "tile_x")

        ext_x = to_int(spatial_axis[0].dom.extent)
        factors, = self.scheduler.schedule_decomposition(
                        self.name, (ext_x,), 1, num_inputs, 0)

        self.ext_tx = factors[2]
        split_axis = []
        axis = spatial_axis[0]
        for f in factors[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis.append(outer)
        split_axis.append(axis)
        self.bx = split_axis[0]
        self.vx = split_axis[1]
        self.tx = split_axis[2]
        self.ix = split_axis[3]
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.vx, vx)
        self.sch[base].bind(self.tx, tx)
      elif num_dim == 2:
        # self.space.add_split(
        #   self.name, "tile_y", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
        # self.space.add_split(
        #   self.name, "tile_x", spatial_axis[1], nparts=4, default=[[-1, 2, 32, -1]])
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_y = to_int(spatial_axis[0].dom.extent)
        ext_x = to_int(spatial_axis[1].dom.extent)
        factors_x, factors_y = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y), 1, num_inputs, 0)

        self.ext_ty = factors_y[2]
        self.ext_tx = factors_x[2]
        split_axis_y = []
        axis = spatial_axis[0]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = spatial_axis[1]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.by = split_axis_y[0]
        self.vy = split_axis_y[1]
        self.ty = split_axis_y[2]
        self.iy = split_axis_y[3]
        self.bx = split_axis_x[0]
        self.vx = split_axis_x[1]
        self.tx = split_axis_x[2]
        self.ix = split_axis_x[3]
        self.sch[base].reorder(self.by, self.bx, *left_axis,
                               self.vy, self.vx,
                               self.ty, self.tx, 
                               self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.vx, vx)
        self.sch[base].bind(self.tx, tx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.vy, vy)
        self.sch[base].bind(self.ty, ty)
      elif num_dim == 3:
        # self.space.add_split(
        #   self.name, "tile_z", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
        # self.space.add_split(
        #   self.name, "tile_y", spatial_axis[1], nparts=4, default=[[-1, 2, 32, -1]])
        # self.space.add_split(
        #   self.name, "tile_x", spatial_axis[2], nparts=4, default=[[-1, 2, 32, -1]])
        # factors_z = self.tuner.propose_split(self.name, "tile_z")
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_z = to_int(spatial_axis[0].dom.extent)
        ext_y = to_int(spatial_axis[1].dom.extent)
        ext_x = to_int(spatial_axis[2].dom.extent)
        factors_x, factors_y, factors_z = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y, ext_z), 1, num_inputs, 0)

        self.ext_tz = factors_z[2]
        self.ext_ty = factors_y[2]
        self.ext_tx = factors_x[2]
        split_axis_z = []
        axis = spatial_axis[0]
        for f in factors_z[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_z.append(outer)
        split_axis_z.append(axis)
        split_axis_y = []
        axis = spatial_axis[1]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = spatial_axis[2]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.bz = split_axis_z[0]
        self.vz = split_axis_z[1]
        self.tz = split_axis_z[2]
        self.iz = split_axis_z[3]
        self.by = split_axis_y[0]
        self.vy = split_axis_y[1]
        self.ty = split_axis_y[2]
        self.iy = split_axis_y[3]
        self.bx = split_axis_x[0]
        self.vx = split_axis_x[1]
        self.tx = split_axis_x[2]
        self.ix = split_axis_x[3]
        self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                               self.vz, self.vy, self.vx,
                               self.tz, self.ty, self.tx, 
                               self.iz, self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.vx, vx)
        self.sch[base].bind(self.tx, tx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.vy, vy)
        self.sch[base].bind(self.ty, ty)
        self.sch[base].bind(self.bz, bz)
        self.sch[base].bind(self.vz, vz)
        self.sch[base].bind(self.tz, tz)
      else:
        # the dim number > 3
        # find the smallest dims
        to_sort = zip(
          spatial_axis, [to_int(x.dom.extent) for x in spatial_axis])
        # ascending
        after_sort = list(sorted(to_sort, key=lambda x: x[1]))
        outer_extent = reduce(lambda x, y: x * y, [x[1] for x in after_sort[:-2]], 1)
        after_sort = [x[0] for x in after_sort]
        # fuse the small axis together
        # parallel them through block z dim
        self.sch[base].reorder(*after_sort)
        fuse_axis = self.sch[base].fuse(*after_sort[:-2])

        # self.space.add_split(
        #   self.name, "tile_y", spatial_axis[-2], nparts=4, default=[[-1, 2, 32, -1]])
        # self.space.add_split(
        #   self.name, "tile_x", spatial_axis[-1], nparts=4, default=[[-1, 2, 32, -1]])
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_y = to_int(spatial_axis[-2].dom.extent)
        ext_x = to_int(spatial_axis[-1].dom.extent)
        factors_x, factors_y = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y), outer_extent, num_inputs, 0)

        self.ext_ty = factors_y[2]
        self.ext_tx = factors_x[2]
        split_axis_y = []
        axis = after_sort[-2]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = after_sort[-1]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.bz = fuse_axis
        self.by = split_axis_y[0]
        self.vy = split_axis_y[1]
        self.ty = split_axis_y[2]
        self.iy = split_axis_y[3]
        self.bx = split_axis_x[0]
        self.vx = split_axis_x[1]
        self.tx = split_axis_x[2]
        self.ix = split_axis_x[3]
        self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                               self.vy, self.vx,
                               self.ty, self.tx, 
                               self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.vx, vx)
        self.sch[base].bind(self.tx, tx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.vy, vy)
        self.sch[base].bind(self.ty, ty)
        self.sch[base].bind(self.bz, bz)

  def schedule_reductive(self):
    if self.schedule_allreduce:
      # put the reductive into base loop nest
      masters = self.connected_set.master
      base = self.connected_set.base
      assert self.bx is not None
      self.sch[masters[0]].compute_at(self.sch[base], self.bx)
    else:
      masters = self.connected_set.master
      base = self.connected_set.base
      # put the reductive into base loop nest
      assert self.tx is not None

      for count, master in enumerate(masters):
        self.sch[master].compute_at(self.sch[base], self.tx)
        # get all the reduce axis
        reduce_axis = self.sch[master].op.reduce_axis

        if master not in self.rz_list:
          self.rz_list[master] = []
          self.ry_list[master] = []
          self.rx_list[master] = []
        collectors = [self.rz_list[master], self.ry_list[master], self.rx_list[master]]

        ext_list = []
        for i, axis in enumerate(reduce_axis):
          ext_list.append(to_int(axis.dom.extent))

        stat = self.op_stat_dict[master]
        # self.num_add = 0
        # self.num_mul = 0
        # self.num_div = 0
        # self.num_branch = 0
        # self.num_logic = 0
        # self.num_special = 0
        flops = stat.num_add + stat.num_mul + stat.num_div
        num_inputs = len(master.input_tensors)

        factor_list = self.scheduler.schedule_reductive(self.name+"_"+str(count), tuple(ext_list), flops, num_inputs)

        for i, axis in enumerate(reduce_axis):
          # name = "m_"+str(count)+"_tile_r"+str(i)
          # self.space.add_split(self.name, name, axis, nparts=3)
          # factors = self.tuner.propose_split(self.name, name)

          factors = factor_list[i]
          j = 0
          for f in factors[:-1]:
            outer, axis = self.sch[master].split(axis, nparts=f)
            collectors[j].append(outer)
            j += 1
          collectors[j].append(axis)
        inner_axis = self.sch[master].op.axis
        self.sch[master].reorder(*self.rz_list[master], *self.ry_list[master], *self.rx_list[master], *inner_axis)
    
  def cache_fetch(self):
    # it's hard to know the extent of cache array
    # so the schedule decision here may damage performance
    # TODO: find better solutions
    if self.schedule_allreduce:
      return
    else:
      return
      tx = tvm.te.thread_axis("threadIdx.x")
      ty = tvm.te.thread_axis("threadIdx.y")
      tz = tvm.te.thread_axis("threadIdx.z")

      num_reduce_list = []
      num_spatial_list = []

      for i, (tensor, shared) in enumerate(self.read_shared_caches.items()):
        master = self.connected_set.inputs[tensor][0]
        if master != self.connected_set.base:
          num_reduce_list.append(len(self.rx_list[master]))
          num_spatial_list.append(len(self.sch[master].op.axis))
      
      # shared_pos_list, local_pos_list, do_vectorize_list = \
      #   self.scheduler.schedule_cache_pos(self.name, num_reduce_list, num_spatial_list)

      shared_pos_list = [-1 for x in self.read_shared_caches]
      local_pos_list = [-1 for x in self.read_shared_caches]
      do_vectorize_list = [0 for x in self.read_shared_caches]

      pos = 0
      for i, (tensor, shared) in enumerate(self.read_shared_caches.items()):
        master = self.connected_set.inputs[tensor][0]
        if master == self.connected_set.base:
          assert self.ix is not None
          base = master
          local = self.read_local_caches[tensor]
          self.sch[shared].compute_at(self.sch[base], self.ix)
          self.sch[local].compute_at(self.sch[base], self.ix)
        else:
          candidates = self.rx_list[master] + list(self.sch[master].op.axis)

          local = self.read_local_caches[tensor]

          # name = "cache_pos_"+str(i)
          # self.space.add_cache_pos(self.name, name, num_candidates, 2)
          # positions = self.tuner.propose_cache_pos(self.name, name)
          # share_pos, local_pos = positions
          # name = "vectorize_"+str(i)
          # self.space.add_vectorize(self.name, name)
          # do_vectorize = self.tuner.propose_vectorize(self.name, name)

          share_pos = shared_pos_list[pos]
          local_pos = local_pos_list[pos]
          do_vectorize = do_vectorize_list[pos]
          pos += 1

          self.sch[shared].compute_at(self.sch[master], candidates[share_pos])
          self.sch[local].compute_at(self.sch[master], candidates[local_pos])

          # cooperative fetch and vectorization
          # spatial_axis = self.sch[shared].op.axis
          # num_dim = len(spatial_axis)
          # if num_dim == 1:
          #   if self.ext_tx < 0:
          #     self.ext_tx = 32
          #   outer, inner = self.sch[shared].split(spatial_axis[0], nparts=self.ext_tx)
          #   self.sch[shared].bind(outer, tx)
          #   if do_vectorize:
          #     _, inner = self.sch[shared].split(inner, factor=4)
          #     self.sch[shared].vectorize(inner)
          # elif num_dim == 2:
          #   if self.ext_tx < 0:
          #     self.ext_tx = 32
          #   if self.ext_ty < 0:
          #     self.ext_ty = power_of_x_near(2, math.ceil(1024 / self.ext_tx))
          #   xo, xi = self.sch[shared].split(spatial_axis[1], nparts=self.ext_tx)
          #   yo, yi = self.sch[shared].split(spatial_axis[0], nparts=self.ext_ty)
          #   self.sch[shared].reorder(yo, xo, yi, xi)
          #   self.sch[shared].bind(xo, tx)
          #   self.sch[shared].bind(yo, ty)
          #   if do_vectorize:
          #     _, inner = self.sch[shared].split(xi, factor=4)
          #     self.sch[shared].vectorize(inner)
          # elif num_dim == 3:
          #   if self.ext_tx < 0:
          #     self.ext_tx = 16
          #   if self.ext_ty < 0:
          #     self.ext_ty = 16
          #   if self.ext_tz < 0:
          #     self.ext_tz = power_of_x_near(2, math.ceil(1024 / (self.ext_tx * self.ext_ty)))
          #   xo, xi = self.sch[shared].split(spatial_axis[2], nparts=self.ext_tx)
          #   yo, yi = self.sch[shared].split(spatial_axis[1], nparts=self.ext_ty)
          #   zo, zi = self.sch[shared].split(spatial_axis[0], nparts=self.ext_tz)
          #   self.sch[shared].reorder(zo, yo, xo, zi, yi, xi)
          #   self.sch[shared].bind(xo, tx)
          #   self.sch[shared].bind(yo, ty)
          #   self.sch[shared].bind(zo, tz)
          #   if do_vectorize:
          #     _, inner = self.sch[shared].split(xi, factor=4)
          #     self.sch[shared].vectorize(inner)
          # else:
          #   if self.ext_tx < 0:
          #     self.ext_tx = 16
          #   if self.ext_ty < 0:
          #     self.ext_ty = 8
          #   if self.ext_tz < 0:
          #     self.ext_tz = power_of_x_near(2, math.ceil(1024 / (self.ext_tx * self.ext_ty)))
          #   xo, xi = self.sch[shared].split(spatial_axis[-1], nparts=self.ext_tx)
          #   yo, yi = self.sch[shared].split(spatial_axis[-2], nparts=self.ext_ty)
          #   zo, zi = self.sch[shared].split(spatial_axis[-3], nparts=self.ext_tz)
          #   self.sch[shared].reorder(zo, yo, xo, zi, yi, xi, *spatial_axis[:-3])
          #   self.sch[shared].bind(xo, tx)
          #   self.sch[shared].bind(yo, ty)
          #   self.sch[shared].bind(zo, tz)
          #   if do_vectorize:
          #     _, inner = self.sch[shared].split(spatial_axis[-4], factor=4)
          #     self.sch[shared].vectorize(inner)

  def fuse_prologue(self):
    if self.schedule_allreduce:
      assert self.rf is not None
      inner_most_pos = self.sch[self.rf].op.reduce_axis[-1]
      for op in self.connected_set.prologue:
        self.sch[op].compute_at(self.sch[self.rf], inner_most_pos)
    else:
      for op, master in self.connected_set.prologue.items():
        inner_most_pos = self.sch[master].op.axis[-1]
        self.sch[op].compute_at(self.sch[master], inner_most_pos)

  def fuse_epilogue(self):
    base = self.connected_set.base
    assert self.ix is not None
    inner_most_pos = self.ix
    for op in self.connected_set.epilogue:
      self.sch[op].compute_at(self.sch[base], inner_most_pos)

  def unroll_loop(self):
    kernel_scope = None
    for candidate in [self.kernel_scope, self.bz, self.by, self.bx]:
      if candidate is not None:
        kernel_scope = candidate
        break
    if kernel_scope is not None:
      base = self.connected_set.base

      # self.space.add_unroll(self.name, default=[128, 256, 512, 1024, 1500])
      # step, explicit = self.tuner.propose_unroll(self.name)
      
      step, explicit = self.scheduler.schedule_unroll(self.name)

      self.sch[base].pragma(kernel_scope, 'auto_unroll_max_step', step)
      self.sch[base].pragma(kernel_scope, 'unroll_explicit', explicit)

  
class GPUScheduleMasterSet(PrimitiveGenerator):
  """
  Schedule generator for connected set that
  contains only one master node.

  Args:
  --------------------------
  name: string
    the namespace of this generator,
    used for schedule space knob
  
  subgraph: PyTIRSubGraph
  
  connected_set: ConnectedSet

  down_graph: dict of tensor to list of ops
    this is used to get the consumer
    operators of a tvm tensor

  op_stat_dict: dict of operation to state

  scheduler: Scheduler
  """
  def __init__(self, name, subgraph, connected_set, down_graph, op_stat_dict, scheduler):
    super(GPUScheduleMasterSet, self).__init__(connected_set, scheduler)
    assert len(connected_set.master) == 1
    self.name = name
    self.subgraph = subgraph
    self.op_stat_dict = op_stat_dict
    self.down_graph = down_graph

    self.read_shared_caches = {}
    self.read_local_caches = {}

    self.write_local_cache = None

    # we know there is only one master here
    # so if decide allreduce, then we do allreduce
    self.schedule_allreduce = decide_allreduce(
      subgraph, connected_set, down_graph)

    # kernel scope
    self.kernel_scope = None
    # spatial axis
    self.bx = None
    self.by = None
    self.bz = None
    self.vx = None
    self.vy = None
    self.vz = None
    self.tx = None
    self.ty = None
    self.tz = None
    self.ix = None
    self.iy = None
    self.iz = None
    # the extent of threads
    self.ext_tx = -1
    self.ext_ty = -1
    self.ext_tz = -1
    # reduce axis
    self.rx_list = []
    self.ry_list = []
    self.rz_list = []
    # if we do rfactor
    self.rf = None

  def generate(self, sch):
    self.sch = sch
    self.create_cache()
    self.thread_block_decomposition()
    self.schedule_reductive()
    self.cache_fetch()
    self.fuse_prologue()
    self.fuse_epilogue()
    # self.unroll_loop()

  def create_cache(self):
    if self.schedule_allreduce:
      # currently, we don't consider cache for allreduce
      tx = tvm.te.thread_axis("threadIdx.x")
      masters = self.connected_set.master

      for master in masters:
        reduce_axis = self.sch[master].op.reduce_axis
        # only rfactor the biggest reduce axis
        to_sort = zip(reduce_axis, [to_int(x.dom.extent) for x in reduce_axis])
        after_sort = list(sorted(to_sort, key=lambda x: x[1]))
        axis = after_sort[-1][0]

        # self.space.add_split(self.name, "tile_k", axis, nparts=2, filters=[[1, 1]], default=[[-1, 32], [32, -1]])
        # factors = self.tuner.propose_split(self.name, "tile_k")
        # self.space.add_rfactor(self.name)
        # use_factor = self.tuner.propose_rfactor(self.name)

        num_loops = len(reduce_axis)
        extent = to_int(axis.dom.extent)
        stat = self.op_stat_dict[master]
        # self.num_add = 0
        # self.num_mul = 0
        # self.num_div = 0
        # self.num_branch = 0
        # self.num_logic = 0
        # self.num_special = 0
        flops = stat.num_add + stat.num_mul + stat.num_div
        factors, use_factor = self.scheduler.schedule_allreduce(self.name, num_loops, extent, flops)

        # TODO: why rfactor only takes tensor as inputs?
        if use_factor:
          outer, inner = self.sch[master].split(axis, factor=factors[1])
          MF = self.sch.rfactor(master.output(0), inner)
          self.ext_tx = factors[1]
        else:
          outer, inner = self.sch[master].split(axis, nparts=factors[0])
          MF = self.sch.rfactor(master.output(0), outer)
          self.ext_tx = factors[0]
        self.rf = MF
        self.tx = self.sch[master].op.reduce_axis[0]
        self.sch[master].bind(self.tx, tx)
        self.sch[MF].compute_at(self.sch[master], self.tx)
        self.sch[master].set_store_predicate(tx.var.equal(0))
    else:
      # for all the input data, create cache
      # for inp in self.subgraph.inputs.keys():
      #   inp_shared = self.sch.cache_read(inp, "shared", self.down_graph[inp])
      #   inp_local = self.sch.cache_read(inp_shared, "local", self.down_graph[inp])
      #   self.read_shared_caches[inp] = inp_shared
      #   self.read_local_caches[inp] = inp_local
      # create cache for master node
      masters = self.connected_set.master
      for master in masters:
        # TODO: what if more than one output?
        local = self.sch.cache_write(master.output(0), "local")
        self.write_local_cache = local

  def thread_block_decomposition(self):
    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    vx = tvm.te.thread_axis("vthread")
    vy = tvm.te.thread_axis("vthread")
    vz = tvm.te.thread_axis("vthread")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")
    # TODO: the logic of allreduce and non-allreduce are quite
    # similar, we should write cleaner code in future.
    # For now, just use the long code blocks.
    if self.schedule_allreduce:
      # base is the same as master
      masters = self.connected_set.master
      base = masters[0]
      num_inputs = 1
      org_spatial_axis = self.sch[base].op.axis
      # do not care about axis with dim = 1
      # spatial_axis = list(filter(lambda x: to_int(x.dom.extent) > 1, spatial_axis))
      spatial_axis = []
      left_axis = []
      for x in org_spatial_axis:
        if to_int(x.dom.extent) > 1:
          spatial_axis.append(x)
        else:
          left_axis.append(x)
      # make the kernel scope
      # kernel_scope = spatial_axis[0]
      # kernel_scope, left = self.sch[base].split(kernel_scope, nparts=1)
      # spatial_axis[0] = left
      # self.kernel_scope = kernel_scope
      num_dim = len(spatial_axis)
      if num_dim == 0:
        ox, ix = self.sch[base].split(self.sch[base].op.axis[0], nparts=1)
        vx, ix = self.sch[base].split(ix, nparts=1)
        tx, ix = self.sch[base].split(ix, nparts=1)
        self.bx = ox
        self.vx = vx
        self.tx = tx
        self.ix = ix
        self.sch[base].bind(self.bx, bx)
      elif num_dim == 1:
        # self.space.add_split(self.name, "tile_x", spatial_axis[0], nparts=2)
        # factors = self.tuner.propose_split(self.name, "tile_x")

        ext_x = to_int(spatial_axis[0].dom.extent)
        factors, = self.scheduler.schedule_decomposition(
                        self.name, (ext_x,), 1, num_inputs, 1)
        
        split_axis = []
        axis = spatial_axis[0]
        for f in factors[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis.append(outer)
        split_axis.append(axis)
        self.bx = split_axis[0]
        self.ix = split_axis[1]
        self.sch[base].bind(self.bx, bx)
      elif num_dim == 2:
        # self.space.add_split(self.name, "tile_y", spatial_axis[0], nparts=2)
        # self.space.add_split(self.name, "tile_x", spatial_axis[1], nparts=2)
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_y = to_int(spatial_axis[0].dom.extent)
        ext_x = to_int(spatial_axis[1].dom.extent)
        factors_x, factors_y = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y), 1, num_inputs, 1)

        split_axis_y = []
        axis = spatial_axis[0]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = spatial_axis[1]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.by = split_axis_y[0]
        self.iy = split_axis_y[1]
        self.bx = split_axis_x[0]
        self.ix = split_axis_x[1]
        self.sch[base].reorder(self.by, self.bx, *left_axis,
                               self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.by, by)
      elif num_dim == 3:
        # self.space.add_split(self.name, "tile_z", spatial_axis[0], nparts=2)
        # self.space.add_split(self.name, "tile_y", spatial_axis[1], nparts=2)
        # self.space.add_split(self.name, "tile_x", spatial_axis[2], nparts=2)
        # factors_z = self.tuner.propose_split(self.name, "tile_z")
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_z = to_int(spatial_axis[0].dom.extent)
        ext_y = to_int(spatial_axis[1].dom.extent)
        ext_x = to_int(spatial_axis[2].dom.extent)
        factors_x, factors_y, factors_z = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y, ext_z), 1, num_inputs, 1)

        split_axis_z = []
        axis = spatial_axis[0]
        for f in factors_z[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_z.append(outer)
        split_axis_z.append(axis)
        split_axis_y = []
        axis = spatial_axis[1]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = spatial_axis[2]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.bz = split_axis_z[0]
        self.iz = split_axis_z[1]
        self.by = split_axis_y[0]
        self.iy = split_axis_y[1]
        self.bx = split_axis_x[0]
        self.ix = split_axis_x[1]
        self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                               self.iz, self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.bz, bz)
      else:
        # the dim number > 3
        # find the smallest dims
        to_sort = zip(
          spatial_axis, [to_int(x.dom.extent) for x in spatial_axis])
        # ascending
        after_sort = list(sorted(to_sort, key=lambda x: x[1]))
        outer_extent = reduce(lambda x, y: x * y, [x[1] for x in after_sort[:-2]], 1)
        after_sort = [x[0] for x in after_sort]
        # fuse the small axis together
        # parallel them through block z dim
        self.sch[base].reorder(*after_sort)
        fuse_axis = self.sch[base].fuse(*after_sort[:-2])
        # self.space.add_split(self.name, "tile_y", spatial_axis[-2], nparts=2)
        # self.space.add_split(self.name, "tile_x", spatial_axis[-1], nparts=2)
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_y = to_int(spatial_axis[-2].dom.extent)
        ext_x = to_int(spatial_axis[-1].dom.extent)
        factors_x, factors_y = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y), outer_extent, num_inputs, 1)

        split_axis_y = []
        axis = after_sort[-2]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = after_sort[-1]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.bz = fuse_axis
        self.by = split_axis_y[0]
        self.iy = split_axis_y[1]
        self.bx = split_axis_x[0]
        self.ix = split_axis_x[1]
        self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                               self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.bz, bz)
    else: # non-allreduce
      masters = self.connected_set.master
      base = masters[0]
      num_inputs = 1
      org_spatial_axis = self.sch[base].op.axis
      # do not care about axis with dim = 1
      # spatial_axis = list(filter(lambda x: to_int(x.dom.extent) > 1, spatial_axis))
      spatial_axis = []
      left_axis = []
      for x in org_spatial_axis:
        if to_int(x.dom.extent) > 1:
          spatial_axis.append(x)
        else:
          left_axis.append(x)
      # make the kernel scope
      # kernel_scope = spatial_axis[0]
      # kernel_scope, left = self.sch[base].split(kernel_scope, nparts=1)
      # spatial_axis[0] = left
      # self.kernel_scope = kernel_scope
      num_dim = len(spatial_axis)
      if num_dim == 0:
        ox, ix = self.sch[base].split(self.sch[base].op.axis[0], nparts=1)
        vx, ix = self.sch[base].split(ix, nparts=1)
        tx, ix = self.sch[base].split(ix, nparts=1)
        self.bx = ox
        self.vx = vx
        self.tx = tx
        self.ix = ix
        self.sch[base].bind(self.bx, bx)
      elif num_dim == 1:
        # self.space.add_split(
        #   self.name, "tile_x", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
        # factors = self.tuner.propose_split(self.name, "tile_x")

        ext_x = to_int(spatial_axis[0].dom.extent)
        factors, = self.scheduler.schedule_decomposition(
                        self.name, (ext_x,), 1, num_inputs, 0)

        self.ext_tx = factors[2]
        split_axis = []
        axis = spatial_axis[0]
        for f in factors[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis.append(outer)
        split_axis.append(axis)
        self.bx = split_axis[0]
        self.vx = split_axis[1]
        self.tx = split_axis[2]
        self.ix = split_axis[3]
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.vx, vx)
        self.sch[base].bind(self.tx, tx)
      elif num_dim == 2:
        # self.space.add_split(
        #   self.name, "tile_y", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
        # self.space.add_split(
        #   self.name, "tile_x", spatial_axis[1], nparts=4, default=[[-1, 2, 32, -1]])
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_y = to_int(spatial_axis[0].dom.extent)
        ext_x = to_int(spatial_axis[1].dom.extent)
        factors_x, factors_y = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y), 1, num_inputs, 0)

        self.ext_ty = factors_y[2]
        self.ext_tx = factors_x[2]
        split_axis_y = []
        axis = spatial_axis[0]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = spatial_axis[1]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.by = split_axis_y[0]
        self.vy = split_axis_y[1]
        self.ty = split_axis_y[2]
        self.iy = split_axis_y[3]
        self.bx = split_axis_x[0]
        self.vx = split_axis_x[1]
        self.tx = split_axis_x[2]
        self.ix = split_axis_x[3]
        self.sch[base].reorder(self.by, self.bx, *left_axis,
                               self.vy, self.vx,
                               self.ty, self.tx, 
                               self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.vx, vx)
        self.sch[base].bind(self.tx, tx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.vy, vy)
        self.sch[base].bind(self.ty, ty)
      elif num_dim == 3:
        # self.space.add_split(
        #   self.name, "tile_z", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
        # self.space.add_split(
        #   self.name, "tile_y", spatial_axis[1], nparts=4, default=[[-1, 2, 32, -1]])
        # self.space.add_split(
        #   self.name, "tile_x", spatial_axis[2], nparts=4, default=[[-1, 2, 32, -1]])
        # factors_z = self.tuner.propose_split(self.name, "tile_z")
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_z = to_int(spatial_axis[0].dom.extent)
        ext_y = to_int(spatial_axis[1].dom.extent)
        ext_x = to_int(spatial_axis[2].dom.extent)
        factors_x, factors_y, factors_z = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y, ext_z), 1, num_inputs, 0)

        self.ext_tz = factors_z[2]
        self.ext_ty = factors_y[2]
        self.ext_tx = factors_x[2]
        split_axis_z = []
        axis = spatial_axis[0]
        for f in factors_z[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_z.append(outer)
        split_axis_z.append(axis)
        split_axis_y = []
        axis = spatial_axis[1]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = spatial_axis[2]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.bz = split_axis_z[0]
        self.vz = split_axis_z[1]
        self.tz = split_axis_z[2]
        self.iz = split_axis_z[3]
        self.by = split_axis_y[0]
        self.vy = split_axis_y[1]
        self.ty = split_axis_y[2]
        self.iy = split_axis_y[3]
        self.bx = split_axis_x[0]
        self.vx = split_axis_x[1]
        self.tx = split_axis_x[2]
        self.ix = split_axis_x[3]
        self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                               self.vz, self.vy, self.vx,
                               self.tz, self.ty, self.tx, 
                               self.iz, self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.vx, vx)
        self.sch[base].bind(self.tx, tx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.vy, vy)
        self.sch[base].bind(self.ty, ty)
        self.sch[base].bind(self.bz, bz)
        self.sch[base].bind(self.vz, vz)
        self.sch[base].bind(self.tz, tz)
      else:
        # the dim number > 3
        # find the smallest dims
        to_sort = zip(
          spatial_axis, [to_int(x.dom.extent) for x in spatial_axis])
        # ascending
        after_sort = list(sorted(to_sort, key=lambda x: x[1]))
        outer_extent = reduce(lambda x, y: x * y, [x[1] for x in after_sort[:-2]], 1)
        after_sort = [x[0] for x in after_sort]
        # fuse the small axis together
        # parallel them through block z dim
        self.sch[base].reorder(*after_sort)
        fuse_axis = self.sch[base].fuse(*after_sort[:-2])

        # self.space.add_split(
        #   self.name, "tile_y", spatial_axis[-2], nparts=4, default=[[-1, 2, 32, -1]])
        # self.space.add_split(
        #   self.name, "tile_x", spatial_axis[-1], nparts=4, default=[[-1, 2, 32, -1]])
        # factors_y = self.tuner.propose_split(self.name, "tile_y")
        # factors_x = self.tuner.propose_split(self.name, "tile_x")

        ext_y = to_int(spatial_axis[-2].dom.extent)
        ext_x = to_int(spatial_axis[-1].dom.extent)
        factors_x, factors_y = self.scheduler.schedule_decomposition(
                                  self.name, (ext_x, ext_y), outer_extent, num_inputs, 0)
        
        self.ext_ty = factors_y[2]
        self.ext_tx = factors_x[2]
        split_axis_y = []
        axis = after_sort[-2]
        for f in factors_y[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_y.append(outer)
        split_axis_y.append(axis)
        split_axis_x = []
        axis = after_sort[-1]
        for f in factors_x[:-1]:
          outer, axis = self.sch[base].split(axis, nparts=f)
          split_axis_x.append(outer)
        split_axis_x.append(axis)
        self.bz = fuse_axis
        self.by = split_axis_y[0]
        self.vy = split_axis_y[1]
        self.ty = split_axis_y[2]
        self.iy = split_axis_y[3]
        self.bx = split_axis_x[0]
        self.vx = split_axis_x[1]
        self.tx = split_axis_x[2]
        self.ix = split_axis_x[3]
        self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                               self.vy, self.vx,
                               self.ty, self.tx, 
                               self.iy, self.ix)
        self.sch[base].bind(self.bx, bx)
        self.sch[base].bind(self.vx, vx)
        self.sch[base].bind(self.tx, tx)
        self.sch[base].bind(self.by, by)
        self.sch[base].bind(self.vy, vy)
        self.sch[base].bind(self.ty, ty)
        self.sch[base].bind(self.bz, bz)

  def schedule_reductive(self):
    if self.schedule_allreduce:
     pass
    else:
      assert self.write_local_cache is not None
      master = self.write_local_cache
      masters = self.connected_set.master
      base = masters[0]
      # put the reductive into base loop nest
      assert self.tx is not None
      self.sch[master].compute_at(self.sch[base], self.tx)
      # get all the reduce axis
      reduce_axis = self.sch[master].op.reduce_axis

      collectors = [self.rz_list, self.ry_list, self.rx_list]

      ext_list = []
      for i, axis in enumerate(reduce_axis):
        ext_list.append(to_int(axis.dom.extent))

      stat = self.op_stat_dict[base]
      # self.num_add = 0
      # self.num_mul = 0
      # self.num_div = 0
      # self.num_branch = 0
      # self.num_logic = 0
      # self.num_special = 0
      flops = stat.num_add + stat.num_mul + stat.num_div
      num_inputs = len(master.op.input_tensors)

      factor_list = self.scheduler.schedule_reductive(self.name, tuple(ext_list), flops, num_inputs)

      for i, axis in enumerate(reduce_axis):
        # name = "tile_r"+str(i)
        # self.space.add_split(self.name, name, axis, nparts=3)
        # factors = self.tuner.propose_split(self.name, name)

        factors = factor_list[i]

        j = 0
        for f in factors[:-1]:
          outer, axis = self.sch[master].split(axis, nparts=f)
          collectors[j].append(outer)
          j += 1
        collectors[j].append(axis)
      inner_axis = self.sch[master].op.axis
      self.sch[master].reorder(*self.rz_list, *self.ry_list, *self.rx_list, *inner_axis)
    
  def cache_fetch(self):
    # it's hard to know the extent of cache array
    # so the schedule decision here may damage performance
    # TODO: find better solutions
    if self.schedule_allreduce:
      return
    else:
      return
      tx = tvm.te.thread_axis("threadIdx.x")
      ty = tvm.te.thread_axis("threadIdx.y")
      tz = tvm.te.thread_axis("threadIdx.z")
      assert self.write_local_cache is not None
      master = self.write_local_cache
      candidates = self.rx_list + list(self.sch[master].op.axis)

      num_reduce_list = []
      num_spatial_list = []

      for i, (tensor, shared) in enumerate(self.read_shared_caches.items()):
        num_reduce_list.append(len(self.rx_list))
        num_spatial_list.append(len(self.sch[master].op.axis))
      
      # shared_pos_list, local_pos_list, do_vectorize_list = \
      #   self.scheduler.schedule_cache_pos(self.name, num_reduce_list, num_spatial_list)
      shared_pos_list = [-1 for x in self.read_shared_caches]
      local_pos_list = [-1 for x in self.read_shared_caches]
      do_vectorize_list = [0 for x in self.read_shared_caches]

      for i, (tensor, shared) in enumerate(self.read_shared_caches.items()):
        local = self.read_local_caches[tensor]

        # name = "cache_pos_"+str(i)
        # self.space.add_cache_pos(self.name, name, num_candidates, 2)
        # positions = self.tuner.propose_cache_pos(self.name, name)
        # share_pos, local_pos = positions
        # name = "vectorize_"+str(i)
        # self.space.add_vectorize(self.name, name)
        # do_vectorize = self.tuner.propose_vectorize(self.name, name)

        share_pos = shared_pos_list[i]
        local_pos = local_pos_list[i]
        do_vectorize = do_vectorize_list[i]

        self.sch[shared].compute_at(self.sch[master], candidates[share_pos])
        self.sch[local].compute_at(self.sch[master], candidates[local_pos])
        # cooperative fetch and vectorization
        # spatial_axis = self.sch[shared].op.axis
        # num_dim = len(spatial_axis)
        # if num_dim == 1:
        #   if self.ext_tx < 0:
        #     self.ext_tx = 32
        #   outer, inner = self.sch[shared].split(spatial_axis[0], nparts=self.ext_tx)
        #   self.sch[shared].bind(outer, tx)
        #   if do_vectorize == 1:
        #     _, inner = self.sch[shared].split(inner, factor=4)
        #     self.sch[shared].vectorize(inner)
        # elif num_dim == 2:
        #   if self.ext_tx < 0:
        #     self.ext_tx = 32
        #   if self.ext_ty < 0:
        #     self.ext_ty = power_of_x_near(2, math.ceil(1024 / self.ext_tx))
        #   xo, xi = self.sch[shared].split(spatial_axis[1], nparts=self.ext_tx)
        #   yo, yi = self.sch[shared].split(spatial_axis[0], nparts=self.ext_ty)
        #   self.sch[shared].reorder(yo, xo, yi, xi)
        #   self.sch[shared].bind(xo, tx)
        #   self.sch[shared].bind(yo, ty)
        #   if do_vectorize == 1:
        #     _, inner = self.sch[shared].split(xi, factor=4)
        #     self.sch[shared].vectorize(inner)
        # elif num_dim == 3:
        #   if self.ext_tx < 0:
        #     self.ext_tx = 16
        #   if self.ext_ty < 0:
        #     self.ext_ty = 16
        #   if self.ext_tz < 0:
        #     self.ext_tz = power_of_x_near(2, 1024 // (self.ext_tx * self.ext_ty))
        #   xo, xi = self.sch[shared].split(spatial_axis[2], nparts=self.ext_tx)
        #   yo, yi = self.sch[shared].split(spatial_axis[1], nparts=self.ext_ty)
        #   zo, zi = self.sch[shared].split(spatial_axis[0], nparts=self.ext_tz)
        #   self.sch[shared].reorder(zo, yo, xo, zi, yi, xi)
        #   self.sch[shared].bind(xo, tx)
        #   self.sch[shared].bind(yo, ty)
        #   self.sch[shared].bind(zo, tz)
        #   if do_vectorize == 1:
        #     _, inner = self.sch[shared].split(xi, factor=4)
        #     self.sch[shared].vectorize(inner)
        # else:
        #   if self.ext_tx < 0:
        #     self.ext_tx = 16
        #   if self.ext_ty < 0:
        #     self.ext_ty = 8
        #   if self.ext_tz < 0:
        #     self.ext_tz = power_of_x_near(2, math.ceil(1024 / (self.ext_tx * self.ext_ty)))
        #   xo, xi = self.sch[shared].split(spatial_axis[-1], nparts=self.ext_tx)
        #   yo, yi = self.sch[shared].split(spatial_axis[-2], nparts=self.ext_ty)
        #   zo, zi = self.sch[shared].split(spatial_axis[-3], nparts=self.ext_tz)
        #   self.sch[shared].reorder(zo, yo, xo, zi, yi, xi, *spatial_axis[:-3])
        #   self.sch[shared].bind(xo, tx)
        #   self.sch[shared].bind(yo, ty)
        #   self.sch[shared].bind(zo, tz)
        #   if do_vectorize == 1:
        #     _, inner = self.sch[shared].split(spatial_axis[-4], factor=4)
        #     self.sch[shared].vectorize(inner)

  def fuse_prologue(self):
    if self.schedule_allreduce:
      assert self.rf is not None
      inner_most_pos = self.sch[self.rf].op.reduce_axis[-1]
      for op in self.connected_set.prologue:
        self.sch[op].compute_at(self.sch[self.rf], inner_most_pos)
    else:
      assert self.write_local_cache is not None
      master = self.write_local_cache
      inner_most_pos = self.sch[master].op.axis[-1]
      for op in self.connected_set.prologue.keys():
        self.sch[op].compute_at(self.sch[master], inner_most_pos)

  def fuse_epilogue(self):
    assert len(self.connected_set.epilogue) == 0

  def unroll_loop(self):
    kernel_scope = None
    for candidate in [self.kernel_scope, self.bz, self.by, self.bx]:
      if candidate is not None:
        kernel_scope = candidate
        break
    if kernel_scope is not None:
      base = self.connected_set.master[0]

      # self.space.add_unroll(self.name, default=[128, 256, 512, 1024, 1500])
      # step, explicit = self.tuner.propose_unroll(self.name)

      step, explicit = self.scheduler.schedule_unroll(self.name)

      self.sch[base].pragma(kernel_scope, 'auto_unroll_max_step', step)
      self.sch[base].pragma(kernel_scope, 'unroll_explicit', explicit)


class GPUScheduleBaseSet(PrimitiveGenerator):
  """
  Schedule generator for connected set that
  contains only base node.

  Args:
  --------------------------
  name: string
    the namespace of this generator,
    used for schedule space knob
  
  connected_set: ConnectedSet

  scheduler: Scheduler
  """
  def __init__(self, name, connected_set, scheduler):
    super(GPUScheduleBaseSet, self).__init__(connected_set, scheduler)
    self.name = name

    # kernel scope
    self.kernel_scope = None
    # spatial axis
    self.bx = None
    self.by = None
    self.bz = None
    self.vx = None
    self.vy = None
    self.vz = None
    self.tx = None
    self.ty = None
    self.tz = None
    self.ix = None
    self.iy = None
    self.iz = None

  def generate(self, sch):
    self.sch = sch
    self.thread_block_decomposition()
    self.fuse_prologue()
    self.fuse_epilogue()
    # self.unroll_loop()

  def thread_block_decomposition(self):
    bx = tvm.te.thread_axis("blockIdx.x")
    by = tvm.te.thread_axis("blockIdx.y")
    bz = tvm.te.thread_axis("blockIdx.z")
    vx = tvm.te.thread_axis("vthread")
    vy = tvm.te.thread_axis("vthread")
    vz = tvm.te.thread_axis("vthread")
    tx = tvm.te.thread_axis("threadIdx.x")
    ty = tvm.te.thread_axis("threadIdx.y")
    tz = tvm.te.thread_axis("threadIdx.z")

    base = self.connected_set.base
    num_inputs = len(base.input_tensors)
    org_spatial_axis = self.sch[base].op.axis
    # do not care about axis with dim = 1
    # spatial_axis = list(filter(lambda x: to_int(x.dom.extent) > 1, spatial_axis))
    spatial_axis = []
    left_axis = []
    for x in org_spatial_axis:
      if to_int(x.dom.extent) > 1:
        spatial_axis.append(x)
      else:
        left_axis.append(x)
    # make the kernel scope
    # kernel_scope = spatial_axis[0]
    # kernel_scope, left = self.sch[base].split(kernel_scope, nparts=1)
    # spatial_axis[0] = left
    # self.kernel_scope = kernel_scope
    num_dim = len(spatial_axis)
    if num_dim == 0:
      ox, ix = self.sch[base].split(self.sch[base].op.axis[0], nparts=1)
      vx, ix = self.sch[base].split(ix, nparts=1)
      tx, ix = self.sch[base].split(ix, nparts=1)
      self.bx = ox
      self.vx = vx
      self.tx = tx
      self.ix = ix
      self.sch[base].bind(self.bx, bx)
    elif num_dim == 1:
      # self.space.add_split(
      #   self.name, "tile_x", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
      # factors = self.tuner.propose_split(self.name, "tile_x")

      ext_x = to_int(spatial_axis[0].dom.extent)
      factors, = self.scheduler.schedule_decomposition(
                      self.name, (ext_x,), 1, num_inputs, 0)

      self.ext_tx = factors[2]
      split_axis = []
      axis = spatial_axis[0]
      for f in factors[:-1]:
        outer, axis = self.sch[base].split(axis, nparts=f)
        split_axis.append(outer)
      split_axis.append(axis)
      self.bx = split_axis[0]
      self.vx = split_axis[1]
      self.tx = split_axis[2]
      self.ix = split_axis[3]
      self.sch[base].bind(self.bx, bx)
      self.sch[base].bind(self.vx, vx)
      self.sch[base].bind(self.tx, tx)
    elif num_dim == 2:
      # self.space.add_split(
      #   self.name, "tile_y", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
      # self.space.add_split(
      #   self.name, "tile_x", spatial_axis[1], nparts=4, default=[[-1, 2, 32, -1]])
      # factors_y = self.tuner.propose_split(self.name, "tile_y")
      # factors_x = self.tuner.propose_split(self.name, "tile_x")

      ext_y = to_int(spatial_axis[0].dom.extent)
      ext_x = to_int(spatial_axis[1].dom.extent)
      factors_x, factors_y = self.scheduler.schedule_decomposition(
                                self.name, (ext_x, ext_y), 1, num_inputs, 0)

      self.ext_ty = factors_y[2]
      self.ext_tx = factors_x[2]
      split_axis_y = []
      axis = spatial_axis[0]
      for f in factors_y[:-1]:
        outer, axis = self.sch[base].split(axis, nparts=f)
        split_axis_y.append(outer)
      split_axis_y.append(axis)
      split_axis_x = []
      axis = spatial_axis[1]
      for f in factors_x[:-1]:
        outer, axis = self.sch[base].split(axis, nparts=f)
        split_axis_x.append(outer)
      split_axis_x.append(axis)
      self.by = split_axis_y[0]
      self.vy = split_axis_y[1]
      self.ty = split_axis_y[2]
      self.iy = split_axis_y[3]
      self.bx = split_axis_x[0]
      self.vx = split_axis_x[1]
      self.tx = split_axis_x[2]
      self.ix = split_axis_x[3]
      self.sch[base].reorder(self.by, self.bx, *left_axis,
                              self.vy, self.vx,
                              self.ty, self.tx, 
                              self.iy, self.ix)
      self.sch[base].bind(self.bx, bx)
      self.sch[base].bind(self.vx, vx)
      self.sch[base].bind(self.tx, tx)
      self.sch[base].bind(self.by, by)
      self.sch[base].bind(self.vy, vy)
      self.sch[base].bind(self.ty, ty)
    elif num_dim == 3:
      # self.space.add_split(
      #   self.name, "tile_z", spatial_axis[0], nparts=4, default=[[-1, 2, 32, -1]])
      # self.space.add_split(
      #   self.name, "tile_y", spatial_axis[1], nparts=4, default=[[-1, 2, 32, -1]])
      # self.space.add_split(
      #   self.name, "tile_x", spatial_axis[2], nparts=4, default=[[-1, 2, 32, -1]])
      # factors_z = self.tuner.propose_split(self.name, "tile_z")
      # factors_y = self.tuner.propose_split(self.name, "tile_y")
      # factors_x = self.tuner.propose_split(self.name, "tile_x")

      ext_z = to_int(spatial_axis[0].dom.extent)
      ext_y = to_int(spatial_axis[1].dom.extent)
      ext_x = to_int(spatial_axis[2].dom.extent)
      factors_x, factors_y, factors_z = self.scheduler.schedule_decomposition(
                                self.name, (ext_x, ext_y, ext_z), 1, num_inputs, 0)

      self.ext_tz = factors_z[2]
      self.ext_ty = factors_y[2]
      self.ext_tx = factors_x[2]
      split_axis_z = []
      axis = spatial_axis[0]
      for f in factors_z[:-1]:
        outer, axis = self.sch[base].split(axis, nparts=f)
        split_axis_z.append(outer)
      split_axis_z.append(axis)
      split_axis_y = []
      axis = spatial_axis[1]
      for f in factors_y[:-1]:
        outer, axis = self.sch[base].split(axis, nparts=f)
        split_axis_y.append(outer)
      split_axis_y.append(axis)
      split_axis_x = []
      axis = spatial_axis[2]
      for f in factors_x[:-1]:
        outer, axis = self.sch[base].split(axis, nparts=f)
        split_axis_x.append(outer)
      split_axis_x.append(axis)
      self.bz = split_axis_z[0]
      self.vz = split_axis_z[1]
      self.tz = split_axis_z[2]
      self.iz = split_axis_z[3]
      self.by = split_axis_y[0]
      self.vy = split_axis_y[1]
      self.ty = split_axis_y[2]
      self.iy = split_axis_y[3]
      self.bx = split_axis_x[0]
      self.vx = split_axis_x[1]
      self.tx = split_axis_x[2]
      self.ix = split_axis_x[3]
      self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                              self.vz, self.vy, self.vx,
                              self.tz, self.ty, self.tx, 
                              self.iz, self.iy, self.ix)
      self.sch[base].bind(self.bx, bx)
      self.sch[base].bind(self.vx, vx)
      self.sch[base].bind(self.tx, tx)
      self.sch[base].bind(self.by, by)
      self.sch[base].bind(self.vy, vy)
      self.sch[base].bind(self.ty, ty)
      self.sch[base].bind(self.bz, bz)
      self.sch[base].bind(self.vz, vz)
      self.sch[base].bind(self.tz, tz)
    else:
      # the dim number > 3
      # find the smallest dims
      to_sort = zip(
        spatial_axis, [to_int(x.dom.extent) for x in spatial_axis])
      # ascending
      after_sort = list(sorted(to_sort, key=lambda x: x[1]))
      outer_extent = reduce(lambda x, y: x * y, [x[1] for x in after_sort[:-2]], 1)
      after_sort = [x[0] for x in after_sort]
      # fuse the small axis together
      # parallel them through block z dim
      self.sch[base].reorder(*after_sort)
      fuse_axis = self.sch[base].fuse(*after_sort[:-2])

      # self.space.add_split(
      #   self.name, "tile_y", spatial_axis[-2], nparts=4, default=[[-1, 2, 32, -1]])
      # self.space.add_split(
      #   self.name, "tile_x", spatial_axis[-1], nparts=4, default=[[-1, 2, 32, -1]])
      # factors_y = self.tuner.propose_split(self.name, "tile_y")
      # factors_x = self.tuner.propose_split(self.name, "tile_x")

      ext_y = to_int(spatial_axis[-2].dom.extent)
      ext_x = to_int(spatial_axis[-1].dom.extent)
      factors_x, factors_y = self.scheduler.schedule_decomposition(
                                self.name, (ext_x, ext_y), outer_extent, num_inputs, 0)

      self.ext_ty = factors_y[2]
      self.ext_tx = factors_x[2]
      split_axis_y = []
      axis = after_sort[-2]
      for f in factors_y[:-1]:
        outer, axis = self.sch[base].split(axis, nparts=f)
        split_axis_y.append(outer)
      split_axis_y.append(axis)
      split_axis_x = []
      axis = after_sort[-1]
      for f in factors_x[:-1]:
        outer, axis = self.sch[base].split(axis, nparts=f)
        split_axis_x.append(outer)
      split_axis_x.append(axis)
      self.bz = fuse_axis
      self.by = split_axis_y[0]
      self.vy = split_axis_y[1]
      self.ty = split_axis_y[2]
      self.iy = split_axis_y[3]
      self.bx = split_axis_x[0]
      self.vx = split_axis_x[1]
      self.tx = split_axis_x[2]
      self.ix = split_axis_x[3]
      self.sch[base].reorder(self.bz, self.by, self.bx, *left_axis,
                              self.vy, self.vx,
                              self.ty, self.tx, 
                              self.iy, self.ix)
      self.sch[base].bind(self.bx, bx)
      self.sch[base].bind(self.vx, vx)
      self.sch[base].bind(self.tx, tx)
      self.sch[base].bind(self.by, by)
      self.sch[base].bind(self.vy, vy)
      self.sch[base].bind(self.ty, ty)
      self.sch[base].bind(self.bz, bz)

  def fuse_prologue(self):
    base = self.connected_set.base
    assert self.ix is not None
    inner_most_pos = self.ix
    for op in self.connected_set.prologue:
      self.sch[op].compute_at(self.sch[base], inner_most_pos)

  def fuse_epilogue(self):
    base = self.connected_set.base
    assert self.ix is not None
    inner_most_pos = self.ix
    for op in self.connected_set.epilogue:
      self.sch[op].compute_at(self.sch[base], inner_most_pos)

  def unroll_loop(self):
    kernel_scope = None
    for candidate in [self.kernel_scope, self.bz, self.by, self.bx]:
      if candidate is not None:
        kernel_scope = candidate
        break
    if kernel_scope is not None:
      base = self.connected_set.base

      # self.space.add_unroll(self.name, default=[128, 256, 512, 1024, 1500])
      # step, explicit = self.tuner.propose_unroll(self.name)

      step, explicit = self.scheduler.schedule_unroll(self.name)

      self.sch[base].pragma(kernel_scope, 'auto_unroll_max_step', step)
      self.sch[base].pragma(kernel_scope, 'unroll_explicit', explicit)


