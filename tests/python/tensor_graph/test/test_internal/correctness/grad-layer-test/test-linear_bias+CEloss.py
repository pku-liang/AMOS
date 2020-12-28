import tvm
import time
import numpy as np
import torch
import random


from tvm.tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential
from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tvm.tensor_graph.nn import CELoss, SGD
from tvm.tensor_graph.core.schedule_generator import ConnectedSet, GPUScheduleBaseSet, \
      GPUScheduleMasterBaseSet, form_connected_sets, GPUScheduleMasterSet, \
      SingleCut, form_cut_candidates, LayoutTransform
                                  
from tvm.tensor_graph.core.utils import flatten_tir_graph
from tvm.tensor_graph.core.space import PrimitiveSpace, PartitionSpace, ForwardGraphSpace
from tvm.tensor_graph.core.tuner import RandomPrimitiveTuner, RandomPartitionTuner, RandomForwardTuner

batch = 10
img_shape = [batch, 256]
num_classes = 100
label_shape = [batch, num_classes]

def pytorch_result(img_np, label_np, params_np):
  img_torch = torch.tensor(img_np, requires_grad=False)
  label_torch = torch.tensor(label_np, requires_grad=False)
  params_torch = [torch.tensor(x, requires_grad=True) for x in params_np]
  weights_torch = []
  L1 = torch.nn.Linear(img_shape[1], num_classes, bias=True)

  layers = [L1]
  for i, l in enumerate(layers):
    if (i == 0):
      # print("before transpose:", params_torch[i].shape)
      # print("after transpose:", torch.t(params_torch[i]).shape)
      l.weight = torch.nn.Parameter(params_torch[i])
      l.bias = torch.nn.Parameter(params_torch[i+1])
      l.zero_grad()
    else:
      l.weight = torch.nn.Parameter(params_torch[i])
    weights_torch.append(l.weight)
    weights_torch.append(l.bias)
  print("Torch weight is here !!")
  for item in weights_torch:
    print("weights_torch", item.shape, item)

  t1 = L1(img_torch)
  # t2 = act2(t1)
  # t3 = torch.nn.functional.avg_pool2d(t2, 2)

  # t4 = C2(t3)
  # t5 = act2(t4)
  # t6 = torch.nn.functional.avg_pool2d(t5, 2)

  # t7 = C3(t6)
  # t8 = act2(t7)

  # t9 = t8.squeeze()

  # t10 = L1(t9)
  # t11 = act2(L2(t10))

  # t12 = torch.nn.functional.softmax(torch.log(t11+1e-5), dim=1)

  loss = torch.nn.functional.cross_entropy(t1, label_torch)

  loss.backward()

  grads = [x.grad for x in weights_torch]

  return loss, grads

def test1():
  print("test 1 ##############################")
  model = Linear(img_shape[1], num_classes, True)
  print("The parameters in Linear")
  for w in model.weights():
    print(w)


def test2():
  print("test 2 ##############################")

  dtype = "float32"
  model = Linear(img_shape[1], num_classes, True)
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

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

  ce_loss = CELoss(label_tensor)
  sgd = SGD(1)
  fwd_graph = ForwardGraph([img_tensor], [output_tensor], weights_tensors)

  # #change data layout
  # forward_space = ForwardGraphSpace()
  # forward_tuner = RandomForwardTuner(forward_space)
  

  # layout_generator = LayoutTransform(fwd_graph, forward_space, forward_tuner)
  # fgraph = layout_generator.generate()

  fgraph = fwd_graph
  # autodiff
  bgraph = fgraph.make_backward(ce_loss, sgd)

  # make tir graph
  inputs = [x.tvm_tensor for x in bgraph.inputs]
  weights = [x.tvm_tensor for x in bgraph.weights]
  outputs = [x.tvm_tensor for x in bgraph.outputs]
  labels = [x.tvm_tensor for x in bgraph.labels]
  loss = bgraph.loss.tvm_tensor
  gradients = [x.tvm_tensor for x in bgraph.gradients]
  updates = [x.tvm_tensor for x in bgraph.updates]
  print("weights' shape is:")
  for item in weights:
    print(item.shape)
  print("updates' shape is:")
  for item in updates:
    print(item.shape)
  # labels = []
  # loss = None
  # gradients = []
  # updates = []

  tgraph = PyTIRGraph(
    inputs,
    labels,
    outputs,
    weights,
    loss,
    gradients,
    updates)

  print("after tir graph")

  # subgraph partition
  partition_space = PartitionSpace()
  partition_tuner = RandomPartitionTuner(partition_space)

  cut_candidates = form_cut_candidates(tgraph)

  # print(cut_candidates)

  for i, candidate in enumerate(cut_candidates):
    name = "graph_cut_" + str(i)
    partition_generator = SingleCut(tgraph, name, candidate, partition_space, partition_tuner)
    partition_generator.generate()

  # for op, stat in tgraph.op_stat_dict.items():
  #   print(op, " head=", stat.head)

  tgraph.partition_graph()

  print("num subgraphs:", len(tgraph.subgraphs))

  target = "llvm"
  dev = 0

  tgraph.create_schedule()
  tgraph.build(target)
  

  tgraph.set_inputs({bgraph.inputs[0].tvm_tensor: img_np})
  tgraph.set_labels({bgraph.labels[0].tvm_tensor: label_np})
  tgraph.allocate_buffer(target, dev)

  np_weight = []
  for item in tgraph.get_updates():
    np_weight.append(item.asnumpy())

  torch_loss, torch_grad = pytorch_result(img_np, label_np_torch, np_weight)
    
  
  
  print("------------BEFORE FUNC----------------")
  print("Checking loss!")
  print(tgraph.get_loss(0).asnumpy())
  print("Checking Gradients!")
  for item in tgraph.get_gradients():
      print(item.asnumpy())
  print("checking weight")
  for item in tgraph.get_updates():
    print(item.asnumpy())
  print("------------------------------")

    

  for mark in tgraph.call_order:
    func = tgraph.functions[mark]
    bufs = tgraph.bufs[mark]
    real_bufs = [tgraph.tvm_array_dict[tgraph.subgraphs[mark].index[x]] for x in bufs]
    #print("I am here before func")
    func(*real_bufs)
    #print("I am here after func")

  print("------------AFTER FUNC----------------")
  print("Checking loss!")
  print(tgraph.get_loss(0).asnumpy())
  print("Checking Gradients!")
  for item in tgraph.get_gradients():
      print(item.asnumpy())
  # print("checking weights")
  # for item in tgraph.get_updates():
  #   print(item.asnumpy())
  print("------------------------------")

  print("----------------torch loss!-------------")
  print(torch_loss)
  print("torch_grad")
  print(torch_grad)
  
  assert(len(tgraph.get_gradients()) == len(torch_grad))
  for i, each_grad in enumerate(tgraph.get_gradients()):
    tvm.testing.assert_allclose(each_grad.asnumpy(), torch_grad[i].numpy(), atol=1e-5, rtol=1e-30)
  print("Success!")


if __name__ == "__main__":
  test1()
  test2()