import tvm
import time
import numpy as np
try:
    import torch.multiprocessing as _multi
except ImportError:
    import multiprocessing as _multi
multi = _multi.get_context("fork")

from tensor_graph.nn.layers import Layer, BatchNorm2d, Conv2d, ReLU
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph
                                  
from tensor_graph.core.scheduler import schedule_all
from tensor_graph.core.build_graph import build_all
from tensor_graph.core.runtime import run_all

batch = 64
in_channel = 3
out_channel = 64
H = 224
W = 224
kernel_size = 3
stride = 1
padding = 1
dilation = 1
groups = 1
dtype = "float32"
img_shape = [batch, in_channel, H, W]


def conv_bn_relu_tg():
  print("run tensor graph ############################")
  class ConvBnReLU(Layer):
    def __init__(self, in_channel, out_channel, kernel_size,
                bias=False, stride=1, padding=0, dilation=1, groups=1):
      super(ConvBnReLU, self).__init__()
      self.conv = Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)
      self.bn = BatchNorm2d(out_channel)
      self.relu = ReLU()

    def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      return self.relu(x)
  
  block = ConvBnReLU(
    in_channel, out_channel, kernel_size,
    stride=stride, padding=padding, dilation=dilation, groups=groups)
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")

  # get output_tensor
  output_tensor = block(img_tensor)
  
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")

  # get output_tensor
  output_tensor = block(img_tensor)

  # get the weights tensors
  weights_tensors = []
  for w in block.weights():
    weights_tensors.append(w)

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)

  fwd_graph = ForwardGraph([img_tensor], [output_tensor], weights_tensors)

  tir_graph = schedule_all(fwd_graph)

  print("different subgraphs:", len(set(tir_graph.subgraph_features.values())))
  print("direrent ops:", len(set(tir_graph.op_feature_dict.values())))

  tmp = {}
  for f in set(tir_graph.op_feature_dict.values()):
    if f.split(")")[-1] not in tmp:
      tmp[f.split(")")[-1]] = []
    tmp[f.split(")")[-1]].append(f)
  
  for k, v in tmp.items():
    print(k)
    for vv in v:
      print("    ", vv)

  print("####################################################")
  tmp = {}
  for f in set(tir_graph.subgraph_features.values()):
    key = ";".join([x.split(")")[-1] for x in f.split(";")])
    if key not in tmp:
      tmp[key] = []
    tmp[key].append(f)
  
  for k, v in tmp.items():
    print(k)
    for vv in v:
      print("    ", vv)

  target = "cuda"
  dev = 0

  build_all(fwd_graph, tir_graph, target=target, build_trial=1)

  try:
    run_all(tir_graph, [img_np], target=target, dev=dev)
  except Exception as e:
    print("run error:", e)

  print("Success")


def conv_bn_relu_torch():
  print("run pytorch ############################")
  import torch
  import torch.nn as nn

  class ConvBnReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                bias=False, stride=1, padding=0, dilation=1, groups=1):
      super(ConvBnReLU, self).__init__()
      self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                    padding=dilation, groups=groups, bias=False, dilation=dilation)
      self.bn = nn.BatchNorm2d(out_channel)
      self.relu = nn.ReLU()

    def forward(self, x):
      x = self.conv(x)
      x = self.bn(x)
      return self.relu(x)

  model = ConvBnReLU(in_channel, out_channel, kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups)
  model = model.to("cuda")

  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  img_torch = torch.tensor(img_np).to("cuda")

  beg = time.time()
  out = model(img_torch)
  end = time.time()
  print("torch time cost=", (end - beg) * 1e3, "ms")

if __name__ == "__main__":
  conv_bn_relu_tg()
  # conv_bn_relu_torch()