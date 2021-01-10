from tvm.tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential
from tvm.tensor_graph.nn.functional import elementwise_add, mean
from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
import math

def conv_bn(name, in_channel, out_channel, kernel_size, strides=1, padding=1, dilation=1, groups=1, use_bias=False, dtype = "float32", out_dtype="float32"):
    return Sequential(
        Conv2d(in_channel, out_channel, kernel_size, stride=strides, padding=padding, dilation=dilation, groups=groups, bias=use_bias, dtype = dtype, out_dtype = out_dtype),
        BatchNorm2d(out_channel, dtype = dtype, out_dtype = out_dtype),
        ReLU()
    )


def conv_3x3_bn(name, in_channel, out_channel, strides, dtype = "float32", out_dtype="float32"):
    return conv_bn(name, in_channel, out_channel, 3, strides, dtype = dtype, out_dtype = out_dtype)


def conv_1x1_bn(name, in_channel, out_channel, dtype = "float32", out_dtype="float32"):
    return conv_bn(name, in_channel, out_channel, 1, padding=0, dtype = dtype, out_dtype = out_dtype)


class InvertedResidual(Layer):
    def __init__(self, name, in_channel, out_channel, strides, expand_ratio, dtype = "float32", out_dtype="float32"):
        super(InvertedResidual, self).__init__()
        self.strides = strides 
        self.name = name
        assert strides in [1, 2]

        hidden_dim = round(in_channel * expand_ratio)
        self.use_res_connect = self.strides == 1 and in_channel == out_channel

        if expand_ratio == 1:
            self.conv = Sequential(
                # depthwise
                Conv2d(hidden_dim, hidden_dim, 3, stride=strides, padding=1, groups=hidden_dim, dtype = dtype, out_dtype = out_dtype),
                BatchNorm2d(hidden_dim, dtype = dtype, out_dtype = out_dtype),
                ReLU(),
                # pointwise
                Conv2d(hidden_dim, out_channel, 1, stride=1, padding=0, dtype = dtype, out_dtype = out_dtype),
                BatchNorm2d(out_channel, dtype = dtype, out_dtype = out_dtype)
            )
        else:
            self.conv = Sequential(
                # pointwise
                Conv2d(in_channel, hidden_dim, 1, stride=1, padding=0, dtype = dtype, out_dtype = out_dtype),
                BatchNorm2d(hidden_dim, dtype = dtype, out_dtype = out_dtype),
                ReLU(),
                # depthwise
                Conv2d(hidden_dim, hidden_dim, 3, stride=strides, padding=1, groups=hidden_dim, dtype = dtype, out_dtype = out_dtype),
                BatchNorm2d(hidden_dim, dtype = dtype, out_dtype = out_dtype),
                ReLU(),
                # pointwise
                Conv2d(hidden_dim, out_channel, 1, stride=1, padding=0, dtype = dtype, out_dtype = out_dtype),
                BatchNorm2d(out_channel, dtype = dtype, out_dtype = out_dtype)
            )

    def forward(self, inputs):
        if self.use_res_connect:
            return elementwise_add(inputs, self.conv(inputs))
        else:
            return self.conv(inputs)

def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class MobileNetV2(Layer):
    def __init__(self, name, n_class=1000, input_size=224, width_mult=1.0, dtype = "float32", out_dtype="float32"):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # expand_ratio, c, n, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        assert input_size % 32 == 0
        # input_channel = int(input_channel * width_mult)
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_3x3_bn("%s_first_conv3x3_bn" % name, 3, input_channel, 2, dtype = dtype, out_dtype = out_dtype)]

        for count, (t, c, n, s) in enumerate(interverted_residual_setting):
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block("%s_residual_%d_%d" % (name, count+1, i+1), 
                                    input_channel, output_channel, s, expand_ratio=t, dtype = dtype, out_dtype = out_dtype))
                else:
                    self.features.append(block("%s_residual_%d_%d" % (name, count+1, i+1), 
                                    input_channel, output_channel, 1, expand_ratio=t, dtype = dtype, out_dtype = out_dtype))
                input_channel = output_channel
                
        self.features.append(conv_1x1_bn("%s_last_conv1x1_bn" % name, input_channel, self.last_channel, dtype = dtype, out_dtype = out_dtype))
        self.features = Sequential(*self.features)

        self.pool = GlobalAvgPool2d(keep_dim=False)

        self.classifier = Sequential(
            Linear(self.last_channel, n_class, dtype = dtype, out_dtype = out_dtype)
        )

    def forward(self, inputs):
        y = self.features(inputs)
        y = self.pool(y)
        y = self.classifier(y)
        return y

if __name__ == "__main__":
    net = MobileNetV2("mobilenet-v2")

    img_shape = [5, 3, 224, 224]
    dtype = "float32"
    img_tensor = GraphTensor(img_shape, dtype, name="data")
    output = net(img_tensor)

    print(output)
