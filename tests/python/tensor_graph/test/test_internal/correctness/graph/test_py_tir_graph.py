import tvm
import time
import numpy as np

from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tvm.tensor_graph.nn import CELoss, SGD
from tvm.tensor_graph.core.schedule_generator import form_connected_sets, \
      SingleCut, form_cut_candidates, LayoutTransform
                                  
from tvm.tensor_graph.core.utils import flatten_tir_graph
from tvm.tensor_graph.core.space import PartitionSpace, ForwardGraphSpace
from tvm.tensor_graph.core.tuner import RandomPartitionTuner, RandomForwardTuner


def test1():
  print("test 1 ##############################")
  batch = 64
  num_classes = 1000
  img_shape = [batch, num_classes]
  weight_shape = [num_classes, num_classes]
  label_shape = [batch, num_classes]
  dtype = "float32"

  def _gemm(M, N, K, A, B, requires_grad=True):
    k = tvm.te.reduce_axis([0, K])
    return compute([M, N], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), requires_grad=requires_grad)
  
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  weight_tensor = GraphTensor(weight_shape, dtype=dtype, name="weight")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  # get output_tensor
  output_tensor = GraphOp(img_shape, [num_classes], [img_tensor, weight_tensor], _gemm, name="gemm")

  # get the weights tensors
  weights_tensors = [weight_tensor]

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_np = np.random.uniform(-1, 1,label_shape).astype(dtype)

  ce_loss = CELoss(label_tensor)
  sgd = SGD(0.002)
  fwd_graph = ForwardGraph([img_tensor], [output_tensor], weights_tensors)

  # change data layout
  forward_space = ForwardGraphSpace()
  forward_tuner = RandomForwardTuner(forward_space)

  layout_generator = LayoutTransform(fwd_graph, forward_space, forward_tuner)
  fgraph = layout_generator.generate()

  # autodiff
  bgraph = fgraph.make_backward(ce_loss, sgd)

  # make tir graph
  inputs = [x.tvm_tensor for x in bgraph.inputs]
  weights = [x.tvm_tensor for x in bgraph.weights]
  outputs = [x.tvm_tensor for x in bgraph.outputs]
  labels = [x.tvm_tensor for x in bgraph.labels]
  loss = bgraph.loss.tvm_tensor
  gradients = [x.tvm_tensor for x in bgraph.gradients]
  lr = bgraph.lr.tvm_tensor
  updates = [x.tvm_tensor for x in bgraph.updates]

  tgraph = PyTIRGraph(
    inputs,
    labels,
    outputs,
    weights,
    loss,
    gradients,
    lr,
    updates)

  # subgraph partition
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

  # create schedules
  tgraph.create_schedule()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "cuda"
  dev = 0

  # update the op stat dict of subgraphs
  # do auto-schedule
  for mark, subgraph in tgraph.subgraphs.items():
    print("subgraph", mark)
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
  
  print("Success!")



if __name__ == "__main__":
  test1()