from tvm.tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential
from tvm.tensor_graph.nn.functional import elementwise_add


class LeNet5(Layer):

  def __init__(self):
    super(LeNet5, self).__init__()
    self.conv1 = Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False)
    self.s2 = AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.conv3 = Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False)
    self.s4 = AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.conv5 = Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=False)
    self.global_pool = GlobalAvgPool2d(keep_dim=False)
    self.fc6 = Linear(120, 84, bias=False)
    self.output = Linear(84, 10, bias=False)
    self.relu = ReLU()

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.relu(x)
    x = self.s2(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.s4(x)
    x = self.conv5(x)
    x = self.relu(x)
    x = self.global_pool(x)
    x = self.fc6(x)
    x = self.relu(x)
    x = self.output(x)
    return x


class LeNet5Repeat(Layer):
  def __init__(self):
    super(LeNet5Repeat, self).__init__()
    self.conv1 = Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False)
    self.conv1_1 = Conv2d(6, 6, kernel_size=3, padding=1, bias=False)
    self.conv1_2 = Conv2d(6, 6, kernel_size=3, padding=1, bias=False)
    self.s2 = AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.conv3 = Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False)
    self.s4 = AvgPool2d(kernel_size=2, stride=2, padding=0)
    self.conv5 = Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=False)
    self.global_pool = GlobalAvgPool2d(keep_dim=False)
    self.fc6 = Linear(120, 84, bias=False)
    self.output = Linear(84, 10, bias=False)
    self.relu = ReLU()

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
    x = self.global_pool(x)
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
