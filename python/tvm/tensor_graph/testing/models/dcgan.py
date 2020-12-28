from tvm.tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential
from tvm.tensor_graph.nn.functional import elementwise_add, batch_flatten, \
                                conv_transpose2d_nchw
from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph
import math
import tvm
# import torch.nn as nn

def tanh(x):
    """Take tanh of input x.

    Parameters
    ----------
    x : GraphNode
        Arbitrary dimension Input argument.

    Returns
    -------
    y : GraphOp
        The result.
    """
    def _inner_tanh(*args, requires_grad=True):
        assert len(args) > 1
        return compute(
            args[:-1], lambda *i: tvm.te.tanh(args[-1](*i)),
            name="tanh",
            requires_grad=requires_grad)
    return GraphOp(x.shape, [], [x], _inner_tanh, name="tanh")

class Tanh(Layer):
    r"""Applies the element-wise function:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input):
        return tanh(input)

class ConvTranspose2d(Layer):
  def __init__(self, in_channel, out_channel, kernel_size,
        bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, dtype="float32"):
    super(ConvTranspose2d, self).__init__()
    self.in_channel = in_channel
    self.out_channel = out_channel
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2
    stride = (stride, stride) if isinstance(stride, int) else stride
    assert isinstance(stride, (list, tuple)) and len(stride) == 2
    padding = (padding, padding) if isinstance(padding, int) else padding
    assert isinstance(padding, (tuple, list)) and len(padding) == 2
    output_padding = (output_padding, output_padding) if isinstance(output_padding, int) else output_padding
    assert isinstance(output_padding, (tuple, list)) and len(output_padding) == 2
    dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
    assert isinstance(dilation, (tuple, list)) and len(dilation) == 2
    assert isinstance(groups, int)

    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.output_padding = output_padding
    self.groups = groups
    self.dilation = dilation

    self.weight = GraphTensor(
      (in_channel, out_channel // groups, *kernel_size), dtype=dtype, name="conv_transpose_2d_weight", requires_grad=True)
    if bias:
      self.bias = GraphTensor((out_channel,), dtype=dtype, name="conv_transpose_2d_bias", requires_grad=True)
    else:
      self.bias = None

  def forward(self, inputs):
    return conv_transpose2d_nchw(
      inputs, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

def leaky_relu(x, alpha):
    """Take leaky_relu of input x.

    Parameters
    ----------
    x : GraphNode
        Arbitrary dimension Input argument.
    
    alpha : float
        The slope for the small gradient when x < 0

    Returns
    -------
    y : GraphOp
        The result.
    """
    def _inner_LeakyReLU(*args, requires_grad=True):
        assert len(args) > 1
        calpha = tvm.tir.const(alpha, args[-1].tvm_tensor.dtype)
        return compute(
            args[:-1], lambda *i: tvm.tir.Select(args[-1](*i) > 0, args[-1](*i), args[-1](*i) * calpha),
            name="leaky_relu",
            requires_grad=requires_grad)
    return GraphOp(x.shape, [], [x], _inner_LeakyReLU, name="leaky_relu")

class LeakyReLU(Layer):
  def __init__(self, negative_slope = 0.01):
    super(LeakyReLU, self).__init__()
    self.negative_slope = negative_slope
  
  def forward(self, inputs):
    return leaky_relu(inputs, self.negative_slope)

class Generator(Layer):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = Sequential(
            # input is Z, going into a convolution
            ConvTranspose2d(     nz, ngf * 8, 4, stride=1, padding=0, bias=False),
            BatchNorm2d(ngf * 8),
            ReLU(),
            # state size. (ngf*8) x 4 x 4
            ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1, bias=False),
            BatchNorm2d(ngf * 4),
            ReLU(),
            # state size. (ngf*4) x 8 x 8
            ConvTranspose2d(ngf * 4, ngf * 2, 4, stride=2, padding=1, bias=False),
            BatchNorm2d(ngf * 2),
            ReLU(),
            # state size. (ngf*2) x 16 x 16
            ConvTranspose2d(ngf * 2,     ngf, 4, stride=2, padding=1, bias=False),
            BatchNorm2d(ngf),
            ReLU(),
            # state size. (ngf) x 32 x 32
            ConvTranspose2d(    ngf,      nc, 4, stride=2, padding=1, bias=False),
            Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

class Discriminator(Layer):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.ndf = ndf
        self.main = Sequential(
            # input is (nc) x 64 x 64
            Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False),
            LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            BatchNorm2d(ndf * 2),
            LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            BatchNorm2d(ndf * 4),
            LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            BatchNorm2d(ndf * 8),
            LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            Conv2d(ndf * 8, 1, 4, stride=1, padding=0, bias=False),
        )

    def forward(self, input):
        output = self.main(input)
        return batch_flatten(output)


if __name__ == "__main__":
    net1 = Generator()
    net2 = Discriminator()

    batch_size = 1
    latent_vector_size = [batch_size, 100, 1, 1]
    dtype = "float32"
    latent_tensor = GraphTensor(latent_vector_size, dtype, name="data")
    img = net1(latent_tensor)

    print(img)

    output = net2(img)

    print(output)