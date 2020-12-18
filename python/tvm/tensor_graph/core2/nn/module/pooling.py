from tensor_graph.core2.graph.concrete import Tensor, FloatTensor, Compute
from tensor_graph.core2.nn import functional as F

from .base import StateTensor, Module


class AvgPool2d(Module):
    """Avgpool2d of float32 precision
    """
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(AvgPool2d, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        assert isinstance(self.kernel_size, (tuple, list)) and len(self.kernel_size) == 2
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        assert isinstance(self.stride, (list, tuple)) and len(self.stride) == 2
        self.padding = padding

    def forward(self, x):
        return F.avgpool2d(
            x,
            kernel_h=self.kernel_size[0], kernel_w=self.kernel_size[1],
            stride_h=self.stride[0], stride_w=self.stride[1],
            padding=self.padding,
            output_dtype="float32",
            requires_grad=x.requires_grad
        )


class GlobalAvgPool2d(Module):
    """GlobalAvgpool2d of float32 precision
    """
    def __init__(self, keep_dim=False):
        super(GlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim

    def forward(self, x):
        return F.global_avg_pool2d(
            x,
            keep_dim=self.keep_dim,
            requires_grad=x.requires_grad
        )