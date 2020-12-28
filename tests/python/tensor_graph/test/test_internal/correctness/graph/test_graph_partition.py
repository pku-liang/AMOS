import tvm._ffi
import tvm
import sys
import time
import numpy as np
from tvm.tensor_graph.testing.models import resnet

from tvm.tensor_graph.core import ForwardGraph, GraphTensor
from tvm.tensor_graph.nn import CELoss, SGD


def partition(fwd_graph, loss=None, optimizer=None, inference=True):
  import time
  beg = time.time()

  if inference:
    finputs, foutputs, fweights = fwd_graph()

    inputs = [x.tvm_tensor for x in finputs]
    weights = [x.tvm_tensor for x in fweights]
    outputs = [x.tvm_tensor for x in foutputs]
    labels = []
    loss = None
    gradients = []
    lr = None
    updates = []

    tgraph = tvm.tg.make_tir_graph_inference(inputs, outputs, weights)
  else:
    assert loss is not None and optimizer is not None
    bgraph = fwd_graph.make_backward(loss, optimizer)

    inputs = [x.tvm_tensor for x in bgraph.inputs]
    weights = [x.tvm_tensor for x in bgraph.weights]
    outputs = [x.tvm_tensor for x in bgraph.outputs]
    labels = [x.tvm_tensor for x in bgraph.labels]
    loss = bgraph.loss.tvm_tensor
    gradients = [x.tvm_tensor for x in bgraph.gradients]
    lr = optimizer.lr_tensor
    updates = [x.tvm_tensor for x in bgraph.updates]

    tgraph = tvm.tg.make_tir_graph_training(inputs, labels, outputs, weights, loss, gradients, lr, updates)

  end = time.time()
  print("make backward graph takes", (end - beg) * 1e3, "ms")

  print(dir(tgraph))

  beg = time.time()
  multi_graph = tvm.tg.make_tir_multi_graph(tgraph)
  end = time.time()
  print("make multi graph takes", (end - beg) * 1e3, "ms")

  print(dir(multi_graph))
  
  return multi_graph


def test1(file=sys.stdout):
  print("test 1 ##############################")
  batch = 64
  img_shape = [batch, 3, 224, 224]
  num_classes = 1000
  label_shape = [batch, num_classes]
  dtype = "float32"
  model = resnet.resnet50(num_classes=1000)
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  # get output_tensor
  output_tensor = model(img_tensor)

  # get the weights tensors
  weights_tensors = []
  for w in model.weights():
    weights_tensors.append(w)

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_np = np.random.uniform(-1, 1, [batch, num_classes]).astype(dtype)

  ce_loss = CELoss(label_tensor)
  sgd = SGD(0.002)
  fwd_graph = ForwardGraph([img_tensor], [output_tensor], weights_tensors)

  partition(fwd_graph, loss=ce_loss, optimizer=sgd, inference=False)


  print("Success", file=file)


if __name__ == "__main__":
  test1()