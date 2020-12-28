from tvm.tensor_graph.core2.graph.concrete import Tensor, FloatTensor, Compute
from tvm.tensor_graph.core2.nn import functional as F

from .base import StateTensor, Module


class Conv2d(Module):
    """Conv2d of float32 precision
    """
    def __init__(self, in_channel, out_channel, kernel_size,
        bias=False, stride=1, padding=0, dilation=1, groups=1):
        super(Conv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2
        stride = (stride, stride) if isinstance(stride, int) else stride
        assert isinstance(stride, (list, tuple)) and len(stride) == 2
        padding = (padding, padding) if isinstance(padding, int) else padding
        assert isinstance(padding, (tuple, list)) and len(padding) == 2
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        assert isinstance(dilation, (tuple, list)) and len(dilation) == 2
        assert isinstance(groups, int)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = FloatTensor(
            (out_channel, in_channel // groups, *kernel_size), target="llvm", device=0, name="conv2d_weight", requires_grad=True)
        if bias:
            self.bias = FloatTensor((out_channel,), target="llvm", device=0, name="conv2d_bias", requires_grad=True)
        else:
            self.bias = None

    def forward(self, inputs):
        requires_grad = inputs.requires_grad or self.requires_grad
        return F.conv2d_nchw(
            inputs, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups,
            output_dtype="float32", requires_grad=requires_grad)