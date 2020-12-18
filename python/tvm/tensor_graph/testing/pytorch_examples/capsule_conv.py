import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse


class CapsuleConv2d(nn.Module):
  def __init__(self, in_channel, out_channel, kernel_size, input_capsule_size, weight_capsule_size,
      bias=False, stride=1, padding=0, dilation=1, groups=1):
    super(CapsuleConv2d, self).__init__()
    self.in_channel = in_channel
    self.out_channel = out_channel
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2
    input_capsule_size = (input_capsule_size, input_capsule_size) if isinstance(input_capsule_size, int) else input_capsule_size
    assert isinstance(input_capsule_size, (tuple, list)) and len(input_capsule_size) == 2
    weight_capsule_size = (weight_capsule_size, weight_capsule_size) if isinstance(weight_capsule_size, int) else weight_capsule_size
    assert isinstance(weight_capsule_size, (tuple, list)) and len(weight_capsule_size) == 2
    assert input_capsule_size[1] == weight_capsule_size[0]
    stride = (stride, stride) if isinstance(stride, int) else stride
    assert isinstance(stride, (list, tuple)) and len(stride) == 2
    padding = (padding, padding) if isinstance(padding, int) else padding
    assert isinstance(padding, (tuple, list)) and len(padding) == 2
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    assert isinstance(dilation, (tuple, list)) and len(dilation) == 2
    assert isinstance(groups, int)

    self.kernel_size = kernel_size
    self.input_capsule_size = input_capsule_size
    self.weight_capsule_size = weight_capsule_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups

    for i in range(weight_capsule_size[0]):
      for j in range(weight_capsule_size[1]):
        setattr(self, "conv2d_" + str(i) + "_" + str(j), nn.Conv2d(
          self.in_channel, self.out_channel, kernel_size=self.kernel_size, stride=self.stride,
          padding=self.padding, groups=self.groups, bias=bias))

  def forward(self, x):
    out_lst = []
    for i in range(self.input_capsule_size[0]):
      for j in range(self.weight_capsule_size[1]):
        out_ij = 0
        for k in range(self.input_capsule_size[1]):
          x_ik = x[:, :, :, :, i, k].reshape(*x.shape[:4])
          out_ij = getattr(self, "conv2d_" + str(k) + "_" + str(j))(x_ik) + out_ij
          out_lst.append(out_ij.unsqueeze(-1))
    out = torch.cat(out_lst, dim=-1)
    out = out.reshape(*out.shape[:-1], self.input_capsule_size[0], self.weight_capsule_size[1])
    return out


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--device", type=int, default=0)
  parser.add_argument("--batch", type=int, default=20)
  parser.add_argument("--repeats", type=int, default=10)
  parser.add_argument("--number", type=int, default=10)

  args = parser.parse_args()

  torch.backends.cudnn.enabled = False
  in_channel = 1
  out_channel = 256
  kernel_size = 9
  in_capsule_size = (1, 1)
  weight_capsule_size = (1, 8)
  model = CapsuleConv2d(
    in_channel, out_channel, kernel_size, in_capsule_size, weight_capsule_size,
    bias=False, stride=2, padding=0, dilation=1, groups=1).cuda("cuda:" + str(args.device))
  batch = args.batch
  dtype = "float32"
  img = np.random.uniform(-1, 1, [batch, in_channel, 28, 28, 1, 1]).astype(dtype)
  img_tensor = torch.tensor(img).cuda("cuda:" + str(args.device))
  output = model(img_tensor)
  from .utils import count_nodes
  node_count, node_names = count_nodes(output)
  number = args.number
  repeats = args.repeats
  torch.cuda.synchronize()
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for i in range(repeats):
    for j in range(number):
      output = model(img_tensor)
  end.record()
  torch.cuda.synchronize()
  total = start.elapsed_time(end)
  print("Average cost for one iteration:", total / (repeats * number), "ms.")