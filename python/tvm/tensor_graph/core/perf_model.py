import torch
import torch.nn as nn


class AllreduceModel(nn.Module):
  def __init__(self, in_feature=6, device="cpu"):
    super(AllreduceModel, self).__init__()
    self.device = device
    # num_loops, extent, flops, [outer, inner, use_factor]
    features = [256, 512]
    self.d1 = nn.Linear(in_feature, features[0])
    self.d2 = nn.Linear(features[0], features[1])
    self.d3 = nn.Linear(features[1], 1)
    self.act = torch.relu

    for m in [self.d1, self.d2, self.d3]:
      nn.init.xavier_uniform_(m.weight)

  def forward(self, x):
    x = self.d1(x)
    x = self.act(x)
    x = self.d2(x)
    x = self.act(x)
    x = self.d3(x)
    return x


class DecompositionModel(nn.Module):
  def __init__(self, is_allreduce=0, num_loops=1, device="cpu"):
    super(DecompositionModel, self).__init__()
    self.device = device

    features = [256, 512]
    if is_allreduce == 1:
      if num_loops > 3:
        self.d1 = nn.Linear(3+3*2, features[0])
      else:
        self.d1 = nn.Linear(3+3*num_loops, features[0])
    else:
      if num_loops > 3:
        self.d1 = nn.Linear(3+5*2, features[0])
      else:
        self.d1 = nn.Linear(3+5*num_loops, features[0])

    self.d2 = nn.Linear(features[0], features[1])
    self.d3 = nn.Linear(features[1], 1)
    self.act = torch.relu

    for m in [self.d1, self.d2, self.d3]:
      nn.init.xavier_uniform_(m.weight)

  def forward(self, x):
    x = self.d1(x)
    x = self.act(x)
    x = self.d2(x)
    x = self.act(x)
    x = self.d3(x)
    return x


class ReductiveModel(nn.Module):
  def __init__(self, num_loops=1, device="cpu"):
    super(ReductiveModel, self).__init__()
    self.device = device

    features = [256, 512]
    self.d1 = nn.Linear(2 + 4 * num_loops, features[0])

    self.d2 = nn.Linear(features[0], features[1])
    self.d3 = nn.Linear(features[1], 1)
    self.act = torch.relu

    for m in [self.d1, self.d2, self.d3]:
      nn.init.xavier_uniform_(m.weight)

  def forward(self, x):
    x = self.d1(x)
    x = self.act(x)
    x = self.d2(x)
    x = self.act(x)
    x = self.d3(x)
    return x