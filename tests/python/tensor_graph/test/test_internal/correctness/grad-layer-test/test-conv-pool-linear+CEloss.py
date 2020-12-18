import tvm
import time
import numpy as np
import torch
import random


from tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
from tensor_graph.nn import CELoss, SGD
from tensor_graph.core.schedule_generator import ConnectedSet, GPUScheduleBaseSet, \
      GPUScheduleMasterBaseSet, form_connected_sets, GPUScheduleMasterSet, \
      SingleCut, form_cut_candidates, LayoutTransform
                                  
from tensor_graph.core.utils import flatten_tir_graph
from tensor_graph.core.space import PrimitiveSpace, PartitionSpace, ForwardGraphSpace
from tensor_graph.core.tuner import RandomPrimitiveTuner, RandomPartitionTuner, RandomForwardTuner

batch = 5
channel = 3
out_channel = 10
hw = 224
img_shape = [batch, channel, hw, hw]
num_classes = 10
label_shape = [batch, num_classes]

def pytorch_result(img_np, label_np, params_np):
  img_torch = torch.tensor(img_np, requires_grad=False)
  label_torch = torch.tensor(label_np, requires_grad=False)
  params_torch = [torch.tensor(x, requires_grad=True) for x in params_np]
  weights_torch = []
  conv_torch = torch.nn.Conv2d(channel, out_channel, kernel_size=7, stride=2, padding=3, bias=False)
  pool_torch = torch.nn.AdaptiveAvgPool2d((1,1))
  relu_torch = torch.nn.ReLU()
  linear_torch = torch.nn.Linear(img_shape[1], num_classes, bias=False)

  layers = [conv_torch, linear_torch]
  for i, l in enumerate(layers):
    if (i < 1000):
      # print("before transpose:", params_torch[i].shape)
      # print("after transpose:", torch.t(params_torch[i]).shape)
      l.weight = torch.nn.Parameter(params_torch[i])
      l.zero_grad()
    else:
      l.weight = torch.nn.Parameter(params_torch[i])
    weights_torch.append(l.weight)
  print("Torch weight is here !!")
  for item in weights_torch:
    print("weights_torch", item.shape, item)

  t1 = conv_torch(img_torch)
  t2 = pool_torch(t1)
  t3 = t2.view(batch, out_channel)
  t4 = relu_torch(t3)
  t5 = linear_torch(t4)

  loss = torch.nn.functional.cross_entropy(t5, label_torch)

  loss.backward()

  grads = [x.grad for x in weights_torch]

  return loss, grads

class MyModel(Layer):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv_layer = Conv2d(channel, out_channel, kernel_size=7, stride=2, padding=3, bias=False)
    self.pool_layer = GlobalAvgPool2d(keep_dim=False)
    self.relu_layer = ReLU()
    self.linear_layer = Linear(out_channel, num_classes)
        
  def forward(self, x):
    out = self.conv_layer(x)
    out = self.pool_layer(out)
    out = self.relu_layer(out)
    out = self.linear_layer(out)
    return out


def test1():
  print("test 1 ##############################")
  model = MyModel()
  print("The parameters in conv-pool-linear:")
  for w in model.weights():
    print(w)


def test2():
  print("test 2 ##############################")

  dtype = "float32"

  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  model = MyModel()
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