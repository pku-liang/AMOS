import tvm
import time
import numpy as np
from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tvm.tensor_graph.nn import MSELoss, SGD
from tvm.tensor_graph.core.schedule_generator import ConnectedSet, GPUScheduleBaseSet, \
      GPUScheduleMasterBaseSet, form_connected_sets, GPUScheduleMasterSet, \
      SingleCut, form_cut_candidates, LayoutTransform
                                  
from tvm.tensor_graph.core.utils import flatten_tir_graph
from tvm.tensor_graph.core.space import PrimitiveSpace, PartitionSpace, ForwardGraphSpace
from tvm.tensor_graph.core.tuner import RandomPrimitiveTuner, RandomPartitionTuner, RandomForwardTuner
from tvm.tensor_graph.core.scheduler import PrimitiveScheduler as Scheduler


def test1():
  print("test 1 ########################")
  H = 32
  W = 16
  L = 8

  def _gemm(M, N, K, A, B, requires_grad=True):
    k = tvm.te.reduce_axis([0, K])
    return compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), requires_grad=requires_grad)

  def _add(M, N, A, B, requires_grad=True):
    return compute([M, N], lambda i, j: A[i, j] + B[i, j], requires_grad=requires_grad)

  def _add_const(k):
    def _inner(M, N, A, requires_grad=True):
      return compute([M, N], lambda i, j: A[i, j] + k, requires_grad=requires_grad)
    return _inner

  A = GraphTensor([H, L], name="A")
  B = GraphTensor([L, W], name="B")
  C = GraphOp([H, L], [], [A], _add_const(1), name="C")
  D = GraphOp([L, W], [], [B], _add_const(2), name="D")
  E = GraphOp([H, W], [L], [C, D], _gemm, name="E")
  F = GraphTensor([H, W], name="F")
  G = GraphOp([H, W], [], [E, F], _add, name="G")
  He = GraphOp([H, W], [], [G], _add_const(3), name="H")

  label = GraphTensor([H, W], name="label")

  mse_loss = MSELoss(label)
  sgd = SGD(0.002)
  fgraph = ForwardGraph([A], [G], [B, F])

  forward_space = ForwardGraphSpace()
  forward_tuner = RandomForwardTuner(forward_space)

  layout_generator = LayoutTransform(fgraph, forward_space, forward_tuner)
  fgraph = layout_generator.generate()

  bgraph = fgraph.make_backward(mse_loss, sgd)

  # prepare data
  dtype = "float32"
  A_np = np.random.uniform(-1, 1, [H, L]).astype(dtype)
  label_np = np.random.uniform(-1, 1, [H, W]).astype(dtype)

  inputs = [x.tvm_tensor for x in bgraph.inputs]
  weights = [x.tvm_tensor for x in bgraph.weights]
  outputs = [x.tvm_tensor for x in bgraph.outputs]
  labels = [x.tvm_tensor for x in bgraph.labels]
  loss = bgraph.loss.tvm_tensor
  gradients = [x.tvm_tensor for x in bgraph.gradients]
  updates = [x.tvm_tensor for x in bgraph.updates]

  tgraph = PyTIRGraph(
    inputs,
    labels,
    outputs,
    weights,
    loss,
    gradients,
    updates)

  partition_space = PartitionSpace()
  partition_tuner = RandomPartitionTuner(partition_space)

  cut_candidates = form_cut_candidates(tgraph)

  print(cut_candidates)

  for i, candidate in enumerate(cut_candidates):
    name = "graph_cut_" + str(i)
    partition_generator = SingleCut(tgraph, name, candidate, partition_space, partition_tuner)
    partition_generator.generate()

  for op, stat in tgraph.op_stat_dict.items():
    print(op, " head=", stat.head)

  tgraph.partition_graph()
  tgraph.create_schedule()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "cuda"
  dev = 0

  for mark, subgraph in tgraph.subgraphs.items():
    tensors = list(subgraph.outputs.keys()) + list(subgraph.loss.keys()) \
      + list(subgraph.gradients.keys()) + list(subgraph.updates.keys())
    ops = [x.op for x in tensors]
    op_list, down_graph = flatten_tir_graph(ops, output_first=True)
    op_stat_dict = {}
    for op in op_list:
      v = tgraph.op_map[op]
      if v in tgraph.op_stat_dict:
        op_stat_dict[op] = tgraph.op_stat_dict[v]

    c_list = form_connected_sets(subgraph, op_stat_dict, tensors, ops, down_graph)
    print("c_list_length=", len(c_list))
    print("check connected set")
    for connected_set in c_list:
      print(connected_set)
    primitive_space = PrimitiveSpace()
    primitive_tuner = RandomPrimitiveTuner(primitive_space)

    sch = tgraph.schedules[mark]
    scheduler = Scheduler()
    for i, connected_set in enumerate(c_list):
      name = "subgraph_" + str(mark) + "_connect_" + str(i)
      assert not connected_set.empty()
      if connected_set.has_master():
        if connected_set.iso_base():
          PrimitiveScheduler = GPUScheduleMasterBaseSet
        else:
          PrimitiveScheduler = GPUScheduleMasterSet

        primitive_generator = PrimitiveScheduler(
          name, subgraph, connected_set, down_graph, op_stat_dict, scheduler)
      else:
        PrimitiveScheduler = GPUScheduleBaseSet
        primitive_generator = PrimitiveScheduler(
          name, connected_set, scheduler)

      primitive_generator.generate(sch)
      print(tvm.lower(sch, tgraph.bufs[mark], simple_mode=True))

  tgraph.build(target)

  tgraph.set_inputs({bgraph.inputs[0].tvm_tensor: A_np})
  tgraph.set_labels({bgraph.labels[0].tvm_tensor: label_np})
  tgraph.allocate_buffer(target, dev)

  beg = time.time()
  for mark in tgraph.call_order:
    func = tgraph.functions[mark]
    bufs = tgraph.bufs[mark]
    real_bufs = [tgraph.tvm_array_dict[tgraph.subgraphs[mark].index[x]] for x in bufs]
    func_beg = time.time()
    func(*real_bufs)
    func_end = time.time()
    print((func_end - func_beg) * 1e3, "ms")
  end = time.time()

  print("End to end time:", (end - beg) * 1e3, "ms")
  print("Success!")


if __name__ == "__main__":
  test1()