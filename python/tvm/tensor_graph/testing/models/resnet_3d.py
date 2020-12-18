from tensor_graph.nn.layers import Layer, Conv3d, BatchNorm3d, ReLU, \
                                  GlobalAvgPool3d, Linear, Sequential
from tensor_graph.nn.functional import elementwise_add
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
from tensor_graph.core.transform import apply_layout_change
from tensor_graph.nn import CELoss, SGD

class ResidualUnit(Layer):
  """
  Parameters
  ----------
  num_filter : int
      Number of output channels

  stride : tuple
      Stride used in convolution

  dim_match : bool
      True means channel number between input and output is the same,
      otherwise means differ

  bottle_neck : bool
      Whether apply bottleneck transformation.
  """
  def __init__(self, in_feature, num_filter, stride, dim_match, bottle_neck=True):
    super(ResidualUnit, self).__init__()
    self.in_feature = in_feature
    self.num_filter = num_filter
    self.stride = stride
    self.dim_match = dim_match
    self.bottle_neck = bottle_neck

    if self.bottle_neck:
      self.bn1 = BatchNorm3d(in_feature, eps=2e-5)
      self.relu = ReLU()
      self.conv1 = Conv3d(in_feature, int(num_filter*0.25), 1, stride=stride, padding=0)
      self.bn2 = BatchNorm3d(int(num_filter*0.25), eps=2e-3)
      self.conv2 = Conv3d(int(num_filter*0.25), int(num_filter*0.25), 3, stride=1, padding=1)
      self.bn3 = BatchNorm3d(int(num_filter*0.25), eps=2e-3)
      self.conv3 = Conv3d(int(num_filter*0.25), num_filter, 1, stride=1, padding=0)
      self.shortcut = Conv3d(in_feature, num_filter, 1, stride=stride, padding=0)
    else:
      self.bn1 = BatchNorm3d(in_feature, eps=2e-5)
      self.relu = ReLU()
      self.conv1 = Conv3d(in_feature, num_filter, 3, stride=stride, padding=1)
      self.bn2 = BatchNorm3d(num_filter, eps=2e-3)
      self.conv2 = Conv3d(num_filter, num_filter, 3, stride=1, padding=1)
      self.shortcut = Conv3d(in_feature, num_filter, 1, stride=stride, padding=0)

  def forward(self, x):
    if self.bottle_neck:
      bn1 = self.bn1(x)
      act1 = self.relu(bn1)
      conv1 = self.conv1(act1)
      bn2 = self.bn2(conv1)
      act2 = self.relu(bn2)
      conv2 = self.conv2(act2)
      bn3 = self.bn3(conv2)
      act3 = self.relu(bn3)
      conv3 = self.conv3(act3)
      if self.dim_match:
        shortcut = x
      else:
        shortcut = self.shortcut(act1)
      return elementwise_add(conv3, shortcut)
    else:
      bn1 = self.bn1(x)
      act1 = self.relu(bn1)
      conv1 = self.conv1(act1)
      bn2 = self.bn2(conv1)
      act2 = self.relu(bn2)
      conv2 = self.conv2(act2)
      if self.dim_match:
        shortcut = x
      else:
        shortcut = self.shortcut(act1)
      return elementwise_add(conv2, shortcut)


class ResNet3D(Layer):
  """
  Parameters
  ----------
  depth: int
      depth of input
  
  units : list
      Number of units in each stage

  num_stages : int
      Number of stages

  filter_list : list
      Channel size of each stage

  num_classes : int
      Ouput size

  bottle_neck : bool
      Whether apply bottleneck transformation.
  """
  def __init__(self, dpeth, units, num_stages, filter_list, num_classes, bottle_neck=True):
    super(ResNet3D, self).__init__()
    self.dpeth = dpeth
    num_units = len(units)
    assert num_units == num_stages
    self.bn = BatchNorm3d(3, eps=2e-5)
    if self.dpeth <= 32:
      self.conv1 = Conv3d(3, filter_list[0], 3, stride=1, padding=1)
    else:
      self.conv1 = Conv3d(3, filter_list[0], (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
      self.bn1 = BatchNorm3d(filter_list[0], eps=2e-5)
    self.relu = ReLU()

    lst = []
    for i in range(num_stages):
      unit = ResidualUnit(filter_list[i], filter_list[i+1], 1 if i == 0 else 2, False, bottle_neck=bottle_neck)
      lst.append(unit)
      for j in range(units[i]-1):
        unit = ResidualUnit(filter_list[i+1], filter_list[i+1], 1, True, bottle_neck=bottle_neck)
        lst.append(unit)
    self.body = Sequential(*lst)
    self.bn2 = BatchNorm3d(filter_list[-1], eps=2e-5)
    self.pool = GlobalAvgPool3d(keep_dim=False)
    self.fc = Linear(filter_list[-1], num_classes)

  def forward(self, x):
    data = self.bn(x)
    if self.dpeth <= 32:  # cifar10
      body = self.conv1(data)
    else:
      body = self.conv1(data)
      body = self.bn1(body)
      body = self.relu(body)

    body = self.body(body)
    ret = self.bn2(body)
    ret = self.relu(ret)
    ret = self.pool(ret)
    ret = self.fc(ret)
    return ret


def resnet_3d(
  num_classes=10,
  num_layers=50,
  depth=16):
  if depth <= 28:
    num_stages = 3
    if (num_layers-2) % 9 == 0 and num_layers >= 164:
      per_unit = [(num_layers-2)//9]
      filter_list = [16, 64, 128, 256]
      bottle_neck = True
    elif (num_layers-2) % 6 == 0 and num_layers < 164:
      per_unit = [(num_layers-2)//6]
      filter_list = [16, 16, 32, 64]
      bottle_neck = False
    else:
      raise ValueError("no experiments done on num_layers {}".format(num_layers))
    units = per_unit * num_stages
  else:
    if num_layers >= 50:
      filter_list = [64, 256, 512, 1024, 2048]
      bottle_neck = True
    else:
      filter_list = [64, 64, 128, 256, 512]
      bottle_neck = False
    num_stages = 4
    if num_layers == 18:
      units = [2, 2, 2, 2]
    elif num_layers == 34:
      units = [3, 4, 6, 3]
    elif num_layers == 50:
      units = [3, 4, 6, 3]
    elif num_layers == 101:
      units = [3, 4, 23, 3]
    elif num_layers == 152:
      units = [3, 8, 36, 3]
    elif num_layers == 200:
      units = [3, 24, 36, 3]
    elif num_layers == 269:
      units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on num_layers {}".format(num_layers))
  return ResNet3D(
    depth, units=units, num_stages=num_stages,
    filter_list=filter_list, num_classes=num_classes, bottle_neck=bottle_neck)
  
if __name__ == "__main__":
  num_classes = 10
  depth = 16
  model = resnet_3d(num_classes=num_classes, depth=depth)
  batch = 1
  img_shape = [batch, 3, depth, 112, 112]
  label_shape = [batch, num_classes]
  dtype = "float32"
  img_tensor = GraphTensor(img_shape, dtype, name="data", requires_grad=False)
  label_tensor = GraphTensor(label_shape, dtype, name="label", requires_grad=False)
  
  fwd_graph = make_fwd_graph(model, [img_tensor])
  loss = CELoss(label_tensor)
  optimizer = SGD(lr=0.002)
  tir_graph = make_tir_graph(fwd_graph, loss=loss, optimizer=optimizer, inference=False)