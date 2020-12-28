import tvm
import os
import sys
import time
import tvm._ffi
import numpy as np
from tvm import tg
from tvm.tensor_graph.testing.models import lenet, resnet
from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
from tvm.tensor_graph.nn import CELoss, SGD
from tvm.tensor_graph.core.utils import to_tuple
from tvm.tensor_graph.nn.layers import Conv2d


def test_lenet_forward_cuda():
  print("############ test lenet forward cuda ################")
  # get forward graph and tir graph
  model = lenet.lenet5()
  batch = 1
  img_shape = [batch, 1, 32, 32]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data")
  fwd_graph = make_fwd_graph(model, [img_tensor])
  tir_graph = make_tir_graph(fwd_graph, inference=True)
  
  tag_set = set()
  entities = tg.get_schedule_entities(tir_graph, "cuda", 200)
  for i, en in enumerate(entities):
    print("No." + str(i))
    for op, e in zip(tir_graph.operation_list, en.entities):
      print(">>>>>>>>>>>>>>>>>>>")
      print("the operation:", op)
      tag_set.add(op.tag)
      print("tag:", op.tag)
      print("body:", op.body)
      print("-------------------")
      before = tg.schedule_entity_to_string(e)
      print(before)
      assert (tg.schedule_entity_to_string(tg.string_to_schedule_entity(before)) == before)
      print()

  print("total ops:", len(tir_graph.operation_list))
  print("total tags:", len(tag_set))
  print("Success!")


if __name__ == "__main__":
  test_lenet_forward_cuda()
  

  
