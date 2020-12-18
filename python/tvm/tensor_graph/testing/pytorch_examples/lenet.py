import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import time


class LeNet5(nn.Module):

  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False)
    self.s2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.conv3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False)
    self.s4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.conv5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=False)

    self.fc6 = nn.Linear(120, 84, bias=False)
    self.output = nn.Linear(84, 10, bias=False)
    self.relu = nn.ReLU()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.relu(x)
    x = self.s2(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.s4(x)
    x = self.conv5(x)
    x = self.relu(x)
    x = x.mean(-1).mean(-1)
    x = self.fc6(x)
    x = self.relu(x)
    x = self.output(x)
    return x


class LeNet5Repeat(nn.Module):
  def __init__(self):
    super(LeNet5Repeat, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False)
    self.conv1_1 = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=False)
    self.conv1_2 = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=False)
    self.s2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.conv3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False)
    self.s4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.conv5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=False)

    self.fc6 = nn.Linear(120, 84, bias=False)
    self.output = nn.Linear(84, 10, bias=False)
    self.relu = nn.ReLU()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.conv1_1(x)
    x = self.conv1_2(x)
    x = self.relu(x)
    x = self.s2(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.s4(x)
    x = self.conv5(x)
    x = self.relu(x)
    x = x.mean(-1).mean(-1)
    x = self.fc6(x)
    x = self.relu(x)
    x = self.output(x)
    return x


def lenet5():
  model = LeNet5()
  return model


def lenet5_repeat():
  model = LeNet5Repeat()
  return model


if __name__ == "__main__":
  model = lenet5_repeat().cuda()
  batch = 1000
  dtype = "float32"
  img = np.random.uniform(-1, 1, [batch, 1, 32, 32]).astype(dtype)
  img_tensor = torch.tensor(img).cuda()
  number = 100
  for i in range(1):
    input_tensors = [img_tensor] * number
    beg = time.time()
    for j in range(number):
      model(input_tensors[j])
    end = time.time()
    print("Average time cost is %f ms" % ((end - beg) * 1e3 / number))
