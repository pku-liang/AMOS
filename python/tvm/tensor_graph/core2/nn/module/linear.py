from tvm.tensor_graph.core2.graph.concrete import Tensor, FloatTensor, Compute
from tvm.tensor_graph.core2.nn import functional as F

from .base import StateTensor, Module


class Linear(Module):
    """Linear of float32 precision
    """
    def __init__(self, in_features, out_features, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = FloatTensor([out_features, in_features], target="llvm", device=0, name="linear_weight", requires_grad=True)
        if bias:
            self.bias = FloatTensor([out_features], target="llvm", device=0, name="linear_bias", requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x):
        requires_grad = x.requires_grad or self.requires_grad
        return F.linear(x, self.weight, self.bias, output_dtype="float32", requires_grad=requires_grad)