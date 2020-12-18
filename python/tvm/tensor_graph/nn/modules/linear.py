from .module import Module

from tvm import topi
import tvm


class Linear(Module):

    def __init__(self, name, in_features, out_features, bias=True):
        super().__init__(name)
        self.in_features = in_features
        self.out_features = out_features
        # https://docs.tvm.ai/api/python/topi.html#topi.nn.dense
        # - weight (tvm.te.Tensor) – 2-D with shape [out_dim, in_dim]
        self.weight = tvm.te.placeholder([out_features, in_features], dtype='float64', name=f'{name}_weight')
        if bias:
            self.bias = tvm.te.placeholder([out_features, ], dtype='float64', name=f'{name}_bias')
        else:
            self.bias = None

    def __call__(self, inputs):
        return topi.nn.dense(inputs, self.weight, bias=self.bias)

    @property
    def weights(self):
        if self.bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]
