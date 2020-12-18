import tvm
import os
import sys
import time
import tvm._ffi
import numpy as np
from tvm import tg
from tensor_graph.testing.models import lenet, resnet
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
from tensor_graph.nn import CELoss, SGD
from tensor_graph.core.utils import to_tuple
from tensor_graph.nn.layers import Conv2d


def string_skeleton(sk):
  s = ""
  s += ("merge: " + str(sk.merge) + "\n")
  s += ("do_tiling_and_binding: " + str(sk.do_tiling_and_binding) + "\n")
  s += ("buffer_output: " + str(sk.buffer_output) + "\n")
  s += ("use_allreduce: " + str(sk.use_allreduce) + "\n")
  s += ("buffer_input: [" + ", ".join([str(x) for x in sk.buffer_input]) + "]" + "\n")
  return s


def test_lenet_forward_llvm():
  print("############ test lenet forward llvm ################")
  # get forward graph and tir graph
  model = lenet.lenet5()
  batch = 1
  img_shape = [batch, 1, 32, 32]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")
  fwd_graph = make_fwd_graph(model, [img_tensor])
  tir_graph = make_tir_graph(fwd_graph, inference=True)
  
  tag_set = set()
  max_skeleton = 0
  skeletons = tg.get_schedule_skeletons(tir_graph, "llvm")
  for op, sk_list in zip(tir_graph.operation_list, skeletons):
    print(">>>>>>>>>>>>>>>>>>>")
    print("the operation:", op)
    tag_set.add(op.tag)
    print("tag:", op.tag)
    print("body:", op.body)
    print(">>>>>>>>>>>>>>>>>>>")
    max_skeleton = max(max_skeleton, len(sk_list))
    print("the skeletons:", len(sk_list))
    for sk in sk_list:
      print("--------------------")
      print(string_skeleton(sk))

  print("total ops:", len(tir_graph.operation_list))
  print("total tags:", len(tag_set))
  print("max skeleton size:", max_skeleton)
  print("Success!")


def test_lenet_backward_llvm():
  print("############ test lenet backward llvm ################")
  # get forward graph and tir graph
  model = lenet.lenet5()
  batch = 1
  img_shape = [batch, 1, 32, 32]
  label_shape = [batch, 10]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")
  label_tensor = GraphTensor(label_shape, dtype, name="label")
  fwd_graph = make_fwd_graph(model, [img_tensor])
  opt = SGD(0.002)
  loss = CELoss([label_tensor])
  tir_graph = make_tir_graph(fwd_graph, inference=False, loss=loss, optimizer=opt)
  
  tag_set = set()
  max_skeleton = 0
  skeletons = tg.get_schedule_skeletons(tir_graph, "llvm")
  for op, sk_list in zip(tir_graph.operation_list, skeletons):
    print(">>>>>>>>>>>>>>>>>>>")
    print("the operation:", op)
    tag_set.add(op.tag)
    print("tag:", op.tag)
    print("body:", op.body)
    print(">>>>>>>>>>>>>>>>>>>")
    max_skeleton = max(max_skeleton, len(sk_list))
    print("the skeletons:", len(sk_list))
    for sk in sk_list:
      print("--------------------")
      print(string_skeleton(sk))

  print("total ops:", len(tir_graph.operation_list))
  print("total tags:", len(tag_set))
  print("max skeleton size:", max_skeleton)
  print("Success!")


def test_resnet_forward_llvm():
  print("############ test resnet forward llvm ################")
  # get forward graph and tir graph
  model = resnet.resnet50()
  batch = 1
  img_shape = [batch, 3, 224, 224]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")
  fwd_graph = make_fwd_graph(model, [img_tensor])
  tir_graph = make_tir_graph(fwd_graph, inference=True)
  
  tag_set = set()
  max_skeleton = 0
  skeletons = tg.get_schedule_skeletons(tir_graph, "llvm")
  for op, sk_list in zip(tir_graph.operation_list, skeletons):
    print(">>>>>>>>>>>>>>>>>>>")
    print("the operation:", op)
    tag_set.add(op.tag)
    print("tag:", op.tag)
    print("body:", op.body)
    print(">>>>>>>>>>>>>>>>>>>")
    max_skeleton = max(max_skeleton, len(sk_list))
    print("the skeletons:", len(sk_list))
    for sk in sk_list:
      print("--------------------")
      print(string_skeleton(sk))

  print("total ops:", len(tir_graph.operation_list))
  print("total tags:", len(tag_set))
  print("max skeleton size:", max_skeleton)
  print("Success!")


def test_resnet_backward_llvm():
  print("############ test resnet backward llvm ################")
  # get forward graph and tir graph
  model = resnet.resnet50()
  batch = 1
  img_shape = [batch, 3, 224, 224]
  label_shape = [batch, 1000]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")
  label_tensor = GraphTensor(label_shape, dtype, name="label")
  fwd_graph = make_fwd_graph(model, [img_tensor])
  opt = SGD(0.002)
  loss = CELoss([label_tensor])
  tir_graph = make_tir_graph(fwd_graph, inference=False, loss=loss, optimizer=opt)
  
  tag_set = set()
  max_skeleton = 0
  skeletons = tg.get_schedule_skeletons(tir_graph, "llvm")
  for op, sk_list in zip(tir_graph.operation_list, skeletons):
    print(">>>>>>>>>>>>>>>>>>>")
    print("the operation:", op)
    tag_set.add(op.tag)
    print("input tensors:", op.input_tensors)
    print("input operations:", [x.op for x in op.input_tensors])
    print("tag:", op.tag)
    print("body:", op.body)
    print(">>>>>>>>>>>>>>>>>>>")
    max_skeleton = max(max_skeleton, len(sk_list))
    print("the skeletons:", len(sk_list))
    for sk in sk_list:
      print("--------------------")
      print(string_skeleton(sk))

  print("total ops:", len(tir_graph.operation_list))
  print("total tags:", len(tag_set))
  print("max skeleton size:", max_skeleton)
  print("Success!")


if __name__ == "__main__":
  test_lenet_forward_llvm()
  test_lenet_backward_llvm()
  test_resnet_forward_llvm()
  test_resnet_backward_llvm()
  

  