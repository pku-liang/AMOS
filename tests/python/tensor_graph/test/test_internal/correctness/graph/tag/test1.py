import tvm
import numpy as np
from tvm import tg
from tensor_graph.core import compute, GraphOp, GraphTensor, ForwardGraph, \
  make_fwd_graph, make_tir_graph
from tensor_graph.testing.models import yolo_v1
from tensor_graph.nn import CELoss, SGD
from tensor_graph.core.utils import to_tuple


def test1():
  print("test 1 ##############################")
  batch = 64
  img_shape = [batch, 3, 448, 448]
  num_classes = 1470
  label_shape = [batch, num_classes]
  dtype = "float32"
  model = yolo_v1.yolo_v1()
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_np = np.random.uniform(-1, 1, [batch, num_classes]).astype(dtype)

  ce_loss = CELoss(label_tensor)
  sgd = SGD(0.002)

  fwd_graph = make_fwd_graph(model, [img_tensor])
  tir_graph = make_tir_graph(fwd_graph, loss=ce_loss, optimizer=sgd, inference=False)
  
  tag_dict = {}
  for op in tir_graph.operation_list:
    if op.tag not in tag_dict:
      tag_dict[op.tag] = []
    tag_dict[op.tag].append(op)

  for tag, lst in tag_dict.items():
    num = len(lst)
    for i in range(1, num):
      for j, inp in enumerate(lst[0].input_tensors):
        if to_tuple(inp.shape) != to_tuple(lst[i].input_tensors[j].shape):
          print(lst[0].body, lst[0].input_tensors)
          print(lst[i].body, lst[i].input_tensors)



  print("Success")


if __name__ == "__main__":
  test1()