import tvm
import numpy as np
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tensor_graph.nn import MSELoss, SGD
from tensor_graph.core.schedule_generator import ConnectedSet, GPUScheduleBaseSet, \
      GPUScheduleMasterBaseSet, form_connected_sets, GPUScheduleMasterSet
                                  
from tensor_graph.core.utils import flatten_tir_graph
from tensor_graph.core.space import PrimitiveSpace
from tensor_graph.core.tuner import RandomPrimitiveTuner
from tensor_graph.core.scheduler import PrimitiveScheduler


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
  H = GraphOp([H, W], [], [G], _add_const(3), name="H")

  H_tensor, params = H({})
  inputs = [params[x].tvm_tensor for x in [A, B]]
  weights = [params[x].tvm_tensor for x in [F]]
  outputs = [params[x].tvm_tensor for x in [H]]

  tgraph = PyTIRGraph(
    inputs,
    [],
    outputs,
    weights,
    None,
    [],
    [])

  C_tvm_tensor = params[C].tvm_tensor
  D_tvm_tensor = params[D].tvm_tensor
  G_tvm_tensor = params[G].tvm_tensor
  H_tvm_tensor = params[H].tvm_tensor
  tgraph.op_stat_dict[C_tvm_tensor.op].head = True
  tgraph.op_stat_dict[D_tvm_tensor.op].head = True
  tgraph.op_stat_dict[G_tvm_tensor.op].head = False
  tgraph.op_stat_dict[H_tvm_tensor.op].head = False

  tgraph.partition_graph()
  tgraph.create_schedule()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "cuda"
  dev = 0

  subgraph = tgraph.subgraphs[0]
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
  print(c_list)
  space = PrimitiveSpace()
  tuner = RandomPrimitiveTuner(space)
  sch = tgraph.schedules[0]

  scheduler = PrimitiveScheduler()
  for connected_set in c_list:
    scheduler = GPUScheduleMasterBaseSet("test", subgraph, connected_set, down_graph, op_stat_dict, scheduler)
    scheduler.generate(sch)
  
  print(tvm.lower(sch, tgraph.bufs[0], simple_mode=True))
  # tgraph.build(target)

  print("Success!")


def test2():
  print("test 2 ########################")
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

  E_tensor, params = E({})
  inputs = [params[x].tvm_tensor for x in [A]]
  weights = [params[x].tvm_tensor for x in [B]]
  outputs = [params[x].tvm_tensor for x in [E]]

  tgraph = PyTIRGraph(
    inputs,
    [],
    outputs,
    weights,
    None,
    [],
    [])

  C_tvm_tensor = params[C].tvm_tensor
  D_tvm_tensor = params[D].tvm_tensor
  tgraph.op_stat_dict[C_tvm_tensor.op].head = True
  tgraph.op_stat_dict[D_tvm_tensor.op].head = True

  tgraph.partition_graph()
  tgraph.create_schedule()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "cuda"
  dev = 0

  subgraph = tgraph.subgraphs[0]
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
  print(c_list)
  space = PrimitiveSpace()
  tuner = RandomPrimitiveTuner(space)
  sch = tgraph.schedules[0]
  scheduler = PrimitiveScheduler()
  for connected_set in c_list:
    scheduler = GPUScheduleMasterSet("test", subgraph, connected_set, down_graph, op_stat_dict, scheduler)
    scheduler.generate(sch)
  
  print(tvm.lower(sch, tgraph.bufs[0], simple_mode=True))
  # tgraph.build(target)

  print("Success!")


def test3():
  print("test 3 ########################")
  H = 32
  W = 16
  L = 8

  def _add(M, N, A, B, requires_grad=True):
    return compute([M, N], lambda i, j: A[i, j] + B[i, j], requires_grad=requires_grad)

  def _add_const(k):
    def _inner(M, N, A, requires_grad=True):
      return compute([M, N], lambda i, j: A[i, j] + k, requires_grad=requires_grad)
    return _inner

  A = GraphTensor([H, W], name="A")
  B = GraphTensor([H, W], name="B")
  C = GraphOp([H, W], [], [A], _add_const(1), name="C")
  D = GraphOp([H, W], [], [B], _add_const(2), name="D")
  E = GraphOp([H, W], [], [C, D], _add, name="E")
  F = GraphTensor([H, W], name="F")
  G = GraphOp([H, W], [], [E, F], _add, name="G")
  H = GraphOp([H, W], [], [G], _add_const(3), name="H")

  H_tensor, params = H({})
  inputs = [params[x].tvm_tensor for x in [A, B]]
  weights = [params[x].tvm_tensor for x in [F]]
  outputs = [params[x].tvm_tensor for x in [H]]

  tgraph = PyTIRGraph(
    inputs,
    [],
    outputs,
    weights,
    None,
    [],
    [])

  C_tvm_tensor = params[C].tvm_tensor
  D_tvm_tensor = params[D].tvm_tensor
  E_tvm_tensor = params[E].tvm_tensor
  G_tvm_tensor = params[G].tvm_tensor
  H_tvm_tensor = params[H].tvm_tensor
  tgraph.op_stat_dict[C_tvm_tensor.op].head = True
  tgraph.op_stat_dict[D_tvm_tensor.op].head = True
  tgraph.op_stat_dict[E_tvm_tensor.op].head = True
  tgraph.op_stat_dict[G_tvm_tensor.op].head = True
  tgraph.op_stat_dict[H_tvm_tensor.op].head = True

  tgraph.partition_graph()
  tgraph.create_schedule()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "cuda"
  dev = 0

  subgraph = tgraph.subgraphs[0]
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
  print(c_list)
  space = PrimitiveSpace()
  tuner = RandomPrimitiveTuner(space)
  sch = tgraph.schedules[0]
  scheduler = PrimitiveScheduler()
  for connected_set in c_list:
    scheduler = GPUScheduleBaseSet("test", connected_set, scheduler)
    scheduler.generate(sch)
  
  print(tvm.lower(sch, tgraph.bufs[0], simple_mode=True))
  # tgraph.build(target)

  print("Success!")


def test4():
  print("test 4 ########################")
  H = 32
  W = 16
  L = 2048

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

  E_tensor, params = E({})
  inputs = [params[x].tvm_tensor for x in [A]]
  weights = [params[x].tvm_tensor for x in [B]]
  outputs = [params[x].tvm_tensor for x in [E]]

  tgraph = PyTIRGraph(
    inputs,
    [],
    outputs,
    weights,
    None,
    [],
    [])

  C_tvm_tensor = params[C].tvm_tensor
  D_tvm_tensor = params[D].tvm_tensor
  tgraph.op_stat_dict[C_tvm_tensor.op].head = True
  tgraph.op_stat_dict[D_tvm_tensor.op].head = True

  tgraph.partition_graph()
  tgraph.create_schedule()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "cuda"
  dev = 0

  subgraph = tgraph.subgraphs[0]
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
  print(c_list)
  space = PrimitiveSpace()
  tuner = RandomPrimitiveTuner(space)
  sch = tgraph.schedules[0]
  scheduler = PrimitiveScheduler()
  for connected_set in c_list:
    scheduler = GPUScheduleMasterSet("test", subgraph, connected_set, down_graph, op_stat_dict, scheduler)
    scheduler.generate(sch)
  
  print(tvm.lower(sch, tgraph.bufs[0], simple_mode=True))
  # tgraph.build(target)

  print("Success!")


def test5():
  print("test 5 ########################")
  H = 32
  W = 16
  L = 2048

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
  H = GraphOp([H, W], [], [G], _add_const(3), name="H")

  H_tensor, params = H({})
  inputs = [params[x].tvm_tensor for x in [A, B]]
  weights = [params[x].tvm_tensor for x in [F]]
  outputs = [params[x].tvm_tensor for x in [H]]

  tgraph = PyTIRGraph(
    inputs,
    [],
    outputs,
    weights,
    None,
    [],
    [])

  C_tvm_tensor = params[C].tvm_tensor
  D_tvm_tensor = params[D].tvm_tensor
  G_tvm_tensor = params[G].tvm_tensor
  H_tvm_tensor = params[H].tvm_tensor
  tgraph.op_stat_dict[C_tvm_tensor.op].head = True
  tgraph.op_stat_dict[D_tvm_tensor.op].head = True
  tgraph.op_stat_dict[G_tvm_tensor.op].head = False
  tgraph.op_stat_dict[H_tvm_tensor.op].head = False

  tgraph.partition_graph()
  tgraph.create_schedule()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "cuda"
  dev = 0

  subgraph = tgraph.subgraphs[0]
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
  print(c_list)
  space = PrimitiveSpace()
  tuner = RandomPrimitiveTuner(space)
  sch = tgraph.schedules[0]
  scheduler = PrimitiveScheduler()
  for connected_set in c_list:
    scheduler = GPUScheduleMasterBaseSet("test", subgraph, connected_set, down_graph, op_stat_dict, scheduler)
    scheduler.generate(sch)
  
  print(tvm.lower(sch, tgraph.bufs[0], simple_mode=True))
  # tgraph.build(target)

  print("Success!")


if __name__ == "__main__":
  test1()
  test2()
  test3()
  test4()
  test5()