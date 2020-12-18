from tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential
from tensor_graph.nn.functional import elementwise_add, batch_flatten
import tensor_graph.nn.functional as F
from tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
from tensor_graph.nn import CELoss, SGD
import math
import tvm

def conv_bn(in_channel, out_channel, kernel_size, strides=1, padding=0, dilation=1, groups=1, use_bias=False):
    return Sequential(
        Conv2d(in_channel, out_channel, kernel_size, stride=strides, padding=padding, dilation=dilation, groups=groups, bias=use_bias),
        BatchNorm2d(out_channel),
        ReLU()
    )

def shuffle_channels(inputs, groups):
    '''
    inputs: [batch, channel, height, width]
    groups: int
    return: [batch, channel * height * width]
    '''
    assert len(inputs.shape) == 4
    assert inputs.shape[1] % groups == 0
    batch, channel, height, width = inputs.shape
    channel_per_group = channel // groups

    def _inner_view(batch, groups, channel_per_group, height, width, inputs, requires_grad=True):
        return compute([batch, groups, channel_per_group, height, width], 
            lambda n, g, c, h, w: inputs[n, g * channel_per_group + c, h, w],
                name="view",
                requires_grad=requires_grad)

    inputs_view = GraphOp([batch, groups, channel_per_group, height, width], [], [inputs], _inner_view, name="view")

    def _inner_shuffle_channels(batch, channel, height, width, inputs_view, requires_grad=True):
        return compute([batch, channel, height, width], 
            lambda n, c, h, w: inputs_view[n, c % groups, c // groups, h, w],
                name="shuffle_channels",
                requires_grad=requires_grad)

    return GraphOp([batch, channel, height, width], [], [inputs_view], _inner_shuffle_channels, name="shuffle_channels")

def cat_dim1(A, B):
    assert len(A.shape) == 4 and len(B.shape) == 4
    assert A.shape[0] == B.shape[0] and A.shape[2:] == B.shape[2:]
    
    batch, channel_A, height, width = A.shape
    channel_B = B.shape[1]

    def _inner_cat_dim1(batch, channel, height, width, A, B, requires_grad=True):
        return compute([batch, channel, height, width], 
            lambda n, c, h, w: tvm.tir.Select(c < channel_A, A[n, c, h, w], B[n, c - channel_A, h, w]),
                name="cat_dim1",
                requires_grad=requires_grad)
    return GraphOp([batch, channel_A + channel_B, height, width], [], [A, B], _inner_cat_dim1, name="cat_dim1")

class ShuffleNetUnitA(Layer):
    """ShuffleNet unit for stride=1"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = Conv2d(in_channels, bottleneck_channels,
                                        1, groups=groups, stride=1)
        self.bn2 = BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = BatchNorm2d(bottleneck_channels)
        self.group_conv5 = Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.ReLU(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.ReLU(elementwise_add(x, out))
        return out

class ShuffleNetUnitB(Layer):
    """ShuffleNet unit for stride=2"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = BatchNorm2d(bottleneck_channels)
        self.depthwise_conv3 = Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=2,
                                         groups=bottleneck_channels)
        self.bn4 = BatchNorm2d(bottleneck_channels)
        self.group_conv5 = Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.ReLU(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avgpool2d(x, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2, padding=1)
        out = F.ReLU(cat_dim1(x, out)) # TODO: cat
        return out

class ShuffleNet(Layer):
    """ShuffleNet for groups=3"""
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()

        self.conv1 = Conv2d(in_channels, 24, 3, stride=2, padding=1)
        stage2_seq = [ShuffleNetUnitB(24, 240, groups=3)] + \
            [ShuffleNetUnitA(240, 240, groups=3) for i in range(3)]
        self.stage2 = Sequential(*stage2_seq)
        stage3_seq = [ShuffleNetUnitB(240, 480, groups=3)] + \
            [ShuffleNetUnitA(480, 480, groups=3) for i in range(7)]
        self.stage3 = Sequential(*stage3_seq)
        stage4_seq = [ShuffleNetUnitB(480, 960, groups=3)] + \
                     [ShuffleNetUnitA(960, 960, groups=3) for i in range(3)]
        self.stage4 = Sequential(*stage4_seq)
        self.fc = Linear(960, num_classes)

    def forward(self, x):
        net = self.conv1(x)
        # TODO: change to F.max_pool2d(net, 3, stride=2, padding=1)
        net = F.avgpool2d(net, kernel_h=3, kernel_w=3, stride_h=2, stride_w=2, padding=1)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4(net)
        net = F.avgpool2d(net, kernel_h=7, kernel_w=7, stride_h=7, stride_w=7)
        net = batch_flatten(net)
        net = self.fc(net)
        return net

if __name__ == "__main__":
    net = ShuffleNet()

    img_shape = [5, 3, 224, 224]
    dtype = "float32"
    img_tensor = GraphTensor(img_shape, dtype, name="data", requires_grad=False)
    output = net(img_tensor)

    print(output)

    label_shape = [5, 1000]
    label_tensor = GraphTensor(label_shape, dtype, name="label", requires_grad=False)
    loss = CELoss(label_tensor)

    optimizer = SGD(lr=0.002)

    fwd_graph = make_fwd_graph(net, [img_tensor])
    tir_graph = make_tir_graph(fwd_graph, inference=False, loss=loss, optimizer=optimizer, need_grad=False)