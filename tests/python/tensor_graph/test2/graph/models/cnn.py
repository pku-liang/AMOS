import tvm.tensor_graph.core2.nn.functional as F

from tvm.tensor_graph.core2.nn.module import Module, Conv2d, BatchNorm2d, ReLU, AvgPool2d, Linear, Sequential

"""
here we use several small CNNs for tests
"""


class CNN1(Module):
    def __init__(self, channel=3, height=448, width=448, num_classes=1470):
        super(CNN1, self).__init__()

        self.conv1 = Conv2d(channel, 64, 7, stride=2, padding=3)
        self.pool1 = AvgPool2d(2, stride=2)

        self.conv2 = Conv2d(64, 192, 3, stride=1, padding=1)
        self.pool2 = AvgPool2d(2, stride=2)

        # self.conv3 = Conv2d(192, 128, 1)
        # self.conv4 = Conv2d(128, 512, 3, padding=1)

        # self.pool6 = AvgPool2d(2, stride=2)

        self.conv7 = Conv2d(192, 256, 1)

        self.fc1 = Linear(256 * 56 * 56, 4096)

        self.fc2 = Linear(4096, num_classes)

        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.pool1(x)

        x = self.relu(self.conv2(x))

        x = self.pool2(x)

        # x = self.relu(self.conv3(x))
        # x = self.relu(self.conv4(x))

        # x = self.pool6(x)

        x = self.relu(self.conv7(x))

        x = F.flatten(x, requires_grad=True)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        return x


def conv_bn(in_channel, out_channel, kernel_size, strides=1, padding=0, dilation=1, groups=1, use_bias=False):
    return Sequential(
        Conv2d(in_channel, out_channel, kernel_size, stride=strides, padding=padding, dilation=dilation, groups=groups, bias=use_bias),
        BatchNorm2d(out_channel),
        ReLU()
    )

class Block(Module):
    def __init__(self, in_channel, out_channel, strides):
        super(Block, self).__init__()
        self.strides = strides 
        assert strides in [1, 2]

        self.block = Sequential(
            conv_bn(in_channel, in_channel, 3, strides=strides, padding=1, groups=in_channel),
            conv_bn(in_channel, out_channel, 1, strides=1, padding=0)
        )

    def forward(self, inputs):
        return self.block(inputs)


class CNN2(Module):
    def __init__(self, n_class=1000):
        super(CNN2, self).__init__()
        block = Block
        self.last_channel = 1024
        block_setting = [
            # in_channel, out_channel, stride
            [   3, 1024, 2], 
            [1024, 1024, 1]
        ]

        self.features = []
        for config in block_setting:
            self.features.append(block(*config))
                
        self.pool = AvgPool2d(kernel_size=7, stride=7)
        self.features = Sequential(*self.features)

        self.classifier = Linear(self.last_channel, n_class)

    def forward(self, inputs):
        y = self.features(inputs)
        y = self.pool(y)
        y = F.flatten(y, requires_grad=self._train)
        y = self.classifier(y)
        return y


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = ReLU()
        self.downsample = downsample
        self.stride = stride

        self.add = F.add_no_broadcast

    def forward(self, x):
        out = self.relu(x)
        identity = out

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add(out, identity, requires_grad=x.requires_grad)
        out = self.relu(out)

        return out


def cnn1(num_classes=1470):
  model = CNN1(num_classes=num_classes)
  return model


def cnn2(num_classes=1000):
  model = CNN2(n_class=num_classes)
  return model


def bottle_neck(inplanes, planes):
    model = Bottleneck(inplanes, planes)
    return model
