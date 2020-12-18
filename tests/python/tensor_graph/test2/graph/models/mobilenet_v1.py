import tensor_graph.core2.nn.functional as F

from tensor_graph.core2.nn.module import Module, Conv2d, BatchNorm2d, ReLU, AvgPool2d, Linear, Sequential

import math

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


class MobileNetv1(Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.0):
        super(MobileNetv1, self).__init__()
        block = Block
        self.last_channel = 1024
        block_setting = [
            # in_channel, out_channel, stride
            [   3,   32, 2], 
            [  32,   64, 1],
            [  64,  128, 2],
            [ 128,  128, 1],
            [ 128,  256, 2],
            [ 256,  256, 1],
            [ 256,  512, 2],
            [ 512,  512, 1],
            [ 512,  512, 1],
            [ 512,  512, 1],
            [ 512,  512, 1],
            [ 512,  512, 1],
            [ 512, 1024, 2],
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


def mobilenet_v1(num_classes=1000):
  return MobileNetv1(n_class=num_classes)