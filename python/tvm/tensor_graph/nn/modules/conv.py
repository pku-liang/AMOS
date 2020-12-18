from .module import Module

import tvm
from tvm import topi


class Conv2d(Module):

    def __init__(self, name, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__(name)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias

        weight_shape = [out_channels, in_channels, kernel_size, kernel_size]
        self.weight = tvm.te.placeholder(weight_shape, dtype='float64', name=f'{name}_weight')
        if bias:
            self.bias = tvm.te.placeholder([out_channels, ], dtype='float64', name=f'{name}_bias')
        else:
            self.bias = None

    def __call__(self, inputs):
        outputs = topi.nn.conv2d(inputs, self.weight, self.stride, self.padding, self.dilation)
        if self.bias:
            reshaped_bias = topi.reshape(self.bias, (self.in_channels, self.out_channels, 1, 1))
            outputs += reshaped_bias
        return outputs

    @property
    def weights(self):
        if self.bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]