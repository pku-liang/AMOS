import tvm
import time
import numpy as np
import torch
import random

from tensor_graph.testing.models import capsule_tg as capsule
from tensor_graph.testing.pytorch_examples import resnet_annotated
from tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tensor_graph.nn import MarginLoss, SGD
from tensor_graph.core.schedule_generator import ConnectedSet, GPUScheduleBaseSet, \
      GPUScheduleMasterBaseSet, form_connected_sets, GPUScheduleMasterSet, \
      SingleCut, form_cut_candidates, LayoutTransform
                                  
from tensor_graph.core.utils import flatten_tir_graph
from tensor_graph.core.space import PrimitiveSpace, PartitionSpace, ForwardGraphSpace
from tensor_graph.core.tuner import RandomPrimitiveTuner, RandomPartitionTuner, RandomForwardTuner


batch = 21
num_cap = 33
channel = 1
hw = 28
img_shape = [batch, channel, hw, hw]
num_classes = 10
label_shape = [batch, num_classes]


def test1():
  print("test 1 ##############################")
  model = capsule.get_model(batch, num_cap)
  print("The parameters in capsule")
  for w in model.weights():
    print(w)


def test2():
  print("test 2 ##############################")

  dtype = "float32"

  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  model = capsule.get_model(batch, num_cap)
  # get output_tensor
  output_tensor = model(img_tensor)

  # get the weights tensors
  weights_tensors = []
  for w in model.weights():
    weights_tensors.append(w)
    print(w.shape)
  print("len(weights_tensors):", len(weights_tensors))

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  random_onehot_label = []
  for b in range(0, batch):
    random_onehot_label.append(random.randint(0, num_classes-1))
  label_np_torch = torch.tensor(random_onehot_label)
  label_np = np.eye(num_classes)[random_onehot_label].astype(dtype)
  print("random_onehot_label", random_onehot_label)
  print("label_np_torch", label_np_torch)
  print("lable_np", label_np)

  margin_loss = MarginLoss(label_tensor)
  sgd = SGD(1)
  fwd_graph = ForwardGraph([img_tensor], [output_tensor], weights_tensors)

  # #change data layout
  # forward_space = ForwardGraphSpace()
  # forward_tuner = RandomForwardTuner(forward_space)
  

  # layout_generator = LayoutTransform(fwd_graph, forward_space, forward_tuner)
  # fgraph = layout_generator.generate()

  fgraph = fwd_graph
  # autodiff
  bgraph = fgraph.make_backward(margin_loss, sgd)

  sch, bufs = bgraph.create_schedule()
  # print(tvm.lower(sch, bufs, simple_mode=True))
  target = "llvm"
  dev = 0
  naive_func = bgraph.build(sch, bufs, target)
  
  # make tir graph
  inputs = [x.tvm_tensor for x in bgraph.inputs]
  weights = [x.tvm_tensor for x in bgraph.weights]
  outputs = [x.tvm_tensor for x in bgraph.outputs]
  labels = [x.tvm_tensor for x in bgraph.labels]
  loss = bgraph.loss.tvm_tensor
  gradients = [x.tvm_tensor for x in bgraph.gradients]
  updates = [x.tvm_tensor for x in bgraph.updates]
  # labels = []
  # loss = None
  # gradients = []
  # updates = []


  tgraph = PyTIRGraph(
    [x.tvm_tensor for x in bgraph.inputs],
    [x.tvm_tensor for x in bgraph.labels],
    [x.tvm_tensor for x in bgraph.outputs],
    [x.tvm_tensor for x in bgraph.weights],
    bgraph.loss.tvm_tensor,
    [x.tvm_tensor for x in bgraph.gradients],
    bgraph.lr.tvm_tensor,
    [x.tvm_tensor for x in bgraph.updates])

  print("after tir graph")

   # apply config
  # 1. modify op stat list -> head, tail
  # 2. make subgraphs


  tgraph.partition_graph()
  # 3. create schedule
  tgraph.create_schedule()
  # 4. modify schedule
  tgraph.build(target)
  # allocate buffer
  # only the first call has effect
  lr_np = sgd.get_lr().astype("float32")
  tgraph.set_inputs({bgraph.inputs[0].tvm_tensor: img_np})
  tgraph.set_labels({bgraph.labels[0].tvm_tensor: label_np})
  tgraph.set_lr(lr_np)
  tgraph.allocate_buffer(target, dev)

  ctx = tvm.context(target, dev)
  # copy the data (do not use reference)
  A_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.inputs[0].tvm_tensor).asnumpy(), ctx)
  label_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.labels[0].tvm_tensor).asnumpy(), ctx)
  weights_tvm = [tvm.nd.array(tgraph.get_tvm_array(x.tvm_tensor).asnumpy(), ctx) for x in bgraph.weights]
  # B_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[0].tvm_tensor).asnumpy(), ctx)
  # bias_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[1].tvm_tensor).asnumpy(), ctx)
  # E_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.weights[2].tvm_tensor).asnumpy(), ctx)
  lr_tvm = tvm.nd.array(tgraph.get_tvm_array(bgraph.lr.tvm_tensor).asnumpy(), ctx)
  updates_tvm = [tvm.nd.array(x.asnumpy(), ctx) for x in tgraph.get_updates()]
  naive_func(A_tvm, label_tvm, *weights_tvm, lr_tvm, *updates_tvm)
  print("naive_func successful!")

  # # subgraph partition
  # partition_space = PartitionSpace()
  # partition_tuner = RandomPartitionTuner(partition_space)

  # cut_candidates = form_cut_candidates(tgraph)

  # # print(cut_candidates)

  # for i, candidate in enumerate(cut_candidates):
  #   name = "graph_cut_" + str(i)
  #   partition_generator = SingleCut(tgraph, name, candidate, partition_space, partition_tuner)
  #   partition_generator.generate()

  # # for op, stat in tgraph.op_stat_dict.items():
  # #   print(op, " head=", stat.head)

  # tgraph.partition_graph()

  print("num subgraphs:", len(tgraph.subgraphs))


  np_weight = []
  for item in tgraph.get_updates():
    np_weight.append(item.asnumpy())    
  
  
  # print("------------BEFORE FUNC----------------")
  # print("Checking loss!")
  # print(tgraph.get_loss(0).asnumpy())
  # print("Checking Gradients!")
  # for item in tgraph.get_gradients():
  #     print(item.asnumpy())
  # print("checking weight")
  # for item in tgraph.get_updates():
  #   print(item.asnumpy())
  # print("------------------------------")

  for mark in tgraph.call_order:
    print("enter once")
    func = tgraph.functions[mark]
    bufs = tgraph.bufs[mark]
    print("bufs", bufs)
    print("--------")
    print("tgraph.subgraphs[mark].index:", tgraph.subgraphs[mark].index)
    print("----------")
    print("tgraph.subgraphs[mark].index.keys():", tgraph.subgraphs[mark].index.keys())
    real_bufs = [tgraph.tvm_array_dict[tgraph.subgraphs[mark].index[x]] for x in bufs]
    print("I am here before func")
    func(*real_bufs)
    #print("I am here after func")

  print("------------AFTER FUNC----------------")
  print("Checking loss!")
  print(tgraph.get_loss(0).asnumpy())
  print("Checking Gradients!")
  for item in tgraph.get_gradients():
      print(item.asnumpy())


if __name__ == "__main__":
  test1()
  test2()