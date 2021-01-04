import tvm
from tvm.tensor_graph.core import GraphTensor, GraphOp, compute
from tvm.tensor_graph.nn import functional as F


class Layer(object):
  def __init__(self):
    self.train = True

  def forward(self, *args, **kwargs):
    raise NotImplementedError()

  def weights(self):
    for k, v in self.__dict__.items():
      if isinstance(v, Layer):
        for w in v.weights():
          yield w
      if isinstance(v, GraphTensor) and v.requires_grad:
        yield v

  def eval(self):
    self.train = False
    for k, v in self.__dict__.items():
      if isinstance(v, Layer):
        v.eval()

  def __call__(self, *args, **kwargs):
    return self.forward(*args, **kwargs)


class Conv2d(Layer):
  def __init__(self, in_channel, out_channel, kernel_size,
        bias=False, stride=1, padding=0, dilation=1, groups=1,
        dtype="float32", out_dtype="float32"):
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
    self.dtype = dtype
    self.out_dtype = out_dtype

    self.weight = GraphTensor(
      (out_channel, in_channel // groups, *kernel_size), dtype=dtype, name="conv2d_weight", requires_grad=True)
    if bias:
      self.bias = GraphTensor((out_channel,), dtype=out_dtype, name="conv2d_bias", requires_grad=True)
    else:
      self.bias = None

  def forward(self, inputs):
    if self.groups == 1:
      return F.conv2d_nchw(
        inputs, self.weight, self.bias, self.stride, self.padding, self.dilation,
        out_dtype=self.out_dtype)
    else:
      return F.conv2d_nchw_grouped(
        inputs, self.weight, self.bias, self.stride, self.padding, self.dilation,
        self.groups, out_dtype=self.out_dtype
      )


class Conv3d(Layer):
  def __init__(self, in_channel, out_channel, kernel_size,
        bias=False, stride=1, padding=0, dilation=1, groups=1,
        dtype="float32", out_dtype="float32"):
    super(Conv3d, self).__init__()
    self.in_channel = in_channel
    self.out_channel = out_channel
    kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    assert isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 3
    stride = (stride, stride, stride) if isinstance(stride, int) else stride
    assert isinstance(stride, (list, tuple)) and len(stride) == 3
    padding = (padding, padding, padding) if isinstance(padding, int) else padding
    assert isinstance(padding, (tuple, list)) and len(padding) == 3
    dilation = (dilation, dilation, dilation) if isinstance(dilation, int) else dilation
    assert isinstance(dilation, (tuple, list)) and len(dilation) == 3
    assert isinstance(groups, int)

    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.dtype = dtype
    self.out_dtype = out_dtype

    self.weight = GraphTensor(
      (out_channel, in_channel, *kernel_size), dtype=dtype, name="conv3d_weight", requires_grad=True)
    if bias:
      self.bias = GraphTensor((out_channel,), dtype=out_dtype, name="conv3d_bias", requires_grad=True)
    else:
      self.bias = None

  def forward(self, inputs):
    if self.groups == 1:
      return F.conv3d_ncdhw(
        inputs, self.weight, self.bias,
        self.stride, self.padding, self.dilation,
        out_dtype=self.out_dtype)
    else:
      return F.conv3d_ncdhw_grouped(
        inputs, self.weight, self.bias,
        self.stride, self.padding, self.dilation, self.groups,
        out_dtype=self.out_dtype) 


class CapsuleConv2d(Layer):
  def __init__(self, in_channel, out_channel, kernel_size,
        bias=False, stride=1, padding=0, dilation=1, groups=1, num_caps=8,
        dtype="float32", out_dtype="float32"):
    super(CapsuleConv2d, self).__init__()
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
    assert isinstance(num_caps, int)

    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.num_caps = num_caps

    assert groups == 1

    self.dtype = dtype
    self.out_dtype = out_dtype

    self.weight = GraphTensor(
      (out_channel, in_channel, *kernel_size, num_caps), dtype=dtype, name="conv2d_weight", requires_grad=True)
    if bias:
      self.bias = GraphTensor((out_channel, num_caps), dtype=out_dtype, name="conv2d_bias", requires_grad=True)
    else:
      self.bias = None

  def forward(self, inputs):
    return F.conv2d_capsule(
      inputs, self.weight, self.bias, self.stride,
      self.padding, self.dilation, self.num_caps,
      out_dtype=self.out_dtype)


class BatchNorm2d(Layer):
  def __init__(self, num_features, eps=1e-5,
    dtype="float32", out_dtype="float32"):
    super(BatchNorm2d, self).__init__()
    self.alpha = GraphTensor(
      (num_features,), dtype=dtype, name="bn_alpha", requires_grad=True)
    self.beta = GraphTensor(
      (num_features,), dtype=dtype, name="bn_beta", requires_grad=True)
    self.eps = eps

    self.dtype = dtype
    self.out_dtype = out_dtype

  def forward(self, inputs):
    return F.batch_norm2d(
      inputs, self.alpha, self.beta, self.eps, not self.train)


class BatchNorm3d(Layer):
  def __init__(self, num_features, eps=1e-5,
    dtype="float32", out_dtype="float32"):
    super(BatchNorm3d, self).__init__()
    self.alpha = GraphTensor((num_features,), dtype=dtype, name="bn_alpha", requires_grad=True)
    self.beta = GraphTensor((num_features,), dtype=dtype, name="bn_beta", requires_grad=True)
    self.eps = eps

    self.dtype = dtype
    self.out_dtype = out_dtype

  def forward(self, inputs):
    return F.batch_norm3d(inputs, self.alpha, self.beta, self.eps, not self.train)


class ReLU(Layer):
  def __init__(self):
    super(ReLU, self).__init__()

  def forward(self, inputs):
    return F.ReLU(inputs)


class AvgPool2d(Layer):
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
      )


class GlobalAvgPool2d(Layer):
  def __init__(self, keep_dim=True):
    super(GlobalAvgPool2d, self).__init__()
    self.keep_dim = keep_dim

  def forward(self, x):
    return F.global_avg_pool2d(
      x,
      keep_dim=self.keep_dim
      )


class GlobalAvgPool3d(Layer):
  def __init__(self, keep_dim=True):
    super(GlobalAvgPool3d, self).__init__()
    self.keep_dim = keep_dim

  def forward(self, x):
    return F.global_avg_pool3d(
      x,
      keep_dim=self.keep_dim
      )


class Linear(Layer):
  def __init__(self, in_features, out_features, bias=False,
    dtype="float32", out_dtype="float32"):
    super(Linear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = GraphTensor(
      [out_features, in_features], dtype=dtype, name="linear_weight")
    if bias:
      self.bias = GraphTensor(
        [out_features], dtype=out_dtype, name="linear_bias", requires_grad=True)
    else:
      self.bias = None

    self.dtype = dtype
    self.out_dtype = out_dtype
    
  def forward(self, x):
    return F.dense(x, self.weight, self.bias, out_dtype=self.out_dtype)


class Sequential(Layer):
  def __init__(self, *args):
    super(Sequential, self).__init__()
    for i, arg in enumerate(args):
      setattr(self, "layer_" + str(i), arg)
    self.num_args = len(args)
  
  def forward(self, x):
    for i in range(self.num_args):
      x = getattr(self, "layer_" + str(i))(x)
    return x




