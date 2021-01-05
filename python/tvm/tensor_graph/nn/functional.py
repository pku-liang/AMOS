from tvm import topi
import tvm
from numbers import Integral
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, GraphNode
from functools import reduce


class TagGenerator(object):
    def __init__(self):
        # self.registry = {}
        pass

    def __call__(self, hint):
        # if hint not in self.registry:
        #     # self.registry[hint] counts the number, which will be returned as tag-suffix next time
        #     self.registry[hint] = 1
        #     new_tag = hint + "_0"
        # else:
        #     new_tag = hint + "_" + str(self.registry[hint])
        #     self.registry[hint] += 1
        # return new_tag
        # print(hint)
        return hint


tag_gen = TagGenerator()


# 3
# Functions
def elementwise_add(A, B, requires_grad=True, func_only=False):
    """elementwise_add for two GraphNodes with arbitrary same dimension 
    Args:
    -----------------------------
    A: GraphNode
        shape [N1, N2, ..., Nm]
    B: GraphNode
        shape [N1, N2, ..., Nm]

    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [N1, N2, ..., Nm]
    -----------------------------
    """
    assert A.shape == B.shape

    def _elementwise_add(*args, requires_grad=True):
        assert len(args) > 2
        return compute(
            args[:-2],
            lambda *indices: args[-2](*indices) + args[-1](*indices),
            name="elem_add",
            # tag=tag_gen("elem_add_" + "dim_" + str(len(args) - 2)),
            requires_grad=requires_grad)

    if func_only:
        return _elementwise_add
    else:
        return GraphOp(A.shape, [], [A, B], _elementwise_add, name="elem_add", requires_grad=requires_grad)


def l2_norm(A, B, scale=1.0):
    """l2-norm
    Args:
    -----------------------------
    A: GraphNode
        shape [batch, num_feature]
    B: GraphNode
        shape [batch, num_feature]
    scale: (optional:1.0) float
            scale = 1.0: sum all values
            scale = num_feature: mean on features and sum on batch
            scale = num_feature * batch: mean on both features and batch
            others: customized l2 norm
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [1]
    -----------------------------
    """
    assert abs(scale) > 1e-20
    batch, num_feature = A.shape
    assert A.shape == B.shape

    def _inner_l2_norm(one, batch, num_feature, A, B, requires_grad=True):
        rb = tvm.te.reduce_axis([0, batch])
        rf = tvm.te.reduce_axis([0, num_feature])
        return compute([one], lambda i: tvm.te.sum(tvm.tir.power(A[i+rb, rf] - B[i+rb, rf], 2) / scale, axis=[rb, rf]),
                       name="l2_norm",
                       # tag=tag_gen("l2_norm_scale" + str(scale)),
                       requires_grad=True)
    return GraphOp([1], [batch, num_feature], [A, B], _inner_l2_norm, name="l2_norm")


def batch_norm2d(inputs, alpha, beta, epsilon=1e-5, infer=False):
    """2D Batch Normalization for NCHW inputs

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    alpha   : GraphNode
        shape [channel]
    beta    : GraphNode
        shape [channel]
    epsilon : float
        optional
    infer   : bool
        whether for inference
    -----------------------------

    Returns:
    -----------------------------
    GraphNode
        shape [batch, channel, height, width]
    -----------------------------
    """
    assert isinstance(inputs, GraphNode)
    assert isinstance(alpha, GraphNode)
    assert isinstance(beta, GraphNode)
    N, C, H, W = inputs.shape
    prefix = inputs.name
    epsilon = tvm.tir.const(epsilon, inputs.dtype)

    assert (len(alpha.shape) == 1) and (alpha.shape[0] == C)
    assert (len(beta.shape) == 1) and (beta.shape[0] == C)

    if infer:
        def _inner_bn(N, C, H, W, inputs, alpha, beta, requires_grad=True):
            return compute([N, C, H, W],
                           lambda n, c, i, j: inputs[n, c,
                                                     i, j] * alpha[c] + beta[c],
                           name=prefix + "_bn2d",
                           # tag=tag_gen("bn2d"),
                           requires_grad=requires_grad)
        # return (mean, var), (inputs_p, mean_p, var_p), bn
        return GraphOp([N, C, H, W], [], [inputs, alpha, beta], _inner_bn, name=prefix+"batch_norm")
    else:
        def _inner_mean(C, N, H, W, inputs, requires_grad=False):
            rn1 = tvm.te.reduce_axis([0, N], name=prefix + "_rn1")
            rh1 = tvm.te.reduce_axis([0, H], name=prefix + "_rh1")
            rw1 = tvm.te.reduce_axis([0, W], name=prefix + "_rw1")
            return compute([C],
                           lambda c: tvm.te.sum(
                               inputs[rn1, c, rh1, rw1] / (N*H*W), axis=[rn1, rh1, rw1]),
                           name=prefix + "_bn2d_mean",
                           # tag=tag_gen("bn2d_mean"),
                           requires_grad=requires_grad)
        mean = GraphOp([C], [N, H, W], [inputs], _inner_mean,
                       name=prefix + "_mean", requires_grad=False)

        def _inner_square(C, N, H, W, inputs, requires_grad=False):
            rn2 = tvm.te.reduce_axis([0, N], name=prefix + "_rn2")
            rh2 = tvm.te.reduce_axis([0, H], name=prefix + "_rh2")
            rw2 = tvm.te.reduce_axis([0, W], name=prefix + "_rw2")
            return compute([N, C, H, W],
                           lambda c: tvm.te.sum(
                               (inputs[rn2, c, rh2, rw2] * inputs[rn2, c, rh2, rw2]) / (N*H*W), axis=[rn2, rh2, rw2]),
                           name=prefix + "_bn2d_square",
                           # tag=tag_gen("bn2d_mean"),
                           requires_grad=requires_grad)
        square = GraphOp([C], [N, H, W], [inputs], _inner_square,
                         name=prefix + "_square", requires_grad=False)

        def _inner_var(C, square, mean, requires_grad=False):
            return compute([C],
                           lambda c: square[c] - mean[c] * mean[c],
                           name=prefix + "_bn2d_var",
                           # tag=tag_gen("bn2d_var"),
                           requires_grad=requires_grad)
        var = GraphOp([C], [], [square, mean], _inner_var,
                      name=prefix + "_var", requires_grad=False)

        def _inner_bn(N, C, H, W, inputs, mean, var, alpha, beta, requires_grad=True):
            return compute([N, C, H, W],
                           lambda n, c, i, j: (
                               inputs[n, c, i, j] - mean[c]) / tvm.te.sqrt(var[c] + epsilon) * alpha[c] + beta[c],
                           name=prefix + "_bn2d",
                           # tag=tag_gen("bn2d"),
                           requires_grad=requires_grad)
        # return (mean, var), (inputs_p, mean_p, var_p), bn
        return GraphOp([N, C, H, W], [], [inputs, mean, var, alpha, beta], _inner_bn, name=prefix+"batch_norm")


def batch_norm3d(inputs, alpha, beta, epsilon=1e-5, infer=False):
    """3D Batch Normalization for NCDHW inputs

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, depth, height, width]
    alpha   : GraphNode
        shape [channel]
    beta    : GraphNode
        shape [channel]
    epsilon : float
        optional
    infer   : bool
        whether for inference
    -----------------------------

    Returns:
    -----------------------------
    GraphNode
        shape [batch, channel, depth, height, width]
    -----------------------------
    """
    assert isinstance(inputs, GraphNode)
    assert isinstance(alpha, GraphNode)
    assert isinstance(beta, GraphNode)
    N, C, D, H, W = inputs.shape
    prefix = inputs.name
    epsilon = tvm.tir.const(epsilon, inputs.dtype)
    assert (len(alpha.shape) == 1) and (alpha.shape[0] == C)
    assert (len(beta.shape) == 1) and (beta.shape[0] == C)

    if infer:
        def _inner_bn(N, C, D, H, W, inputs, alpha, beta, requires_grad=True):
            return compute([N, C, D, H, W],
                           lambda n, c, d, i, j: inputs[n, c,
                                                        d, i, j] * alpha[c] + beta[c],
                           name=prefix + "_bn3d",
                           # tag=tag_gen("bn2d"),
                           requires_grad=requires_grad)
        # return (mean, var), (inputs_p, mean_p, var_p), bn
        return GraphOp([N, C, D, H, W], [], [inputs, alpha, beta], _inner_bn, name=prefix+"batch_norm")
    else:
        def _inner_mean(C, N, D, H, W, inputs, requires_grad=False):
            rn1 = tvm.te.reduce_axis([0, N], name=prefix + "_rn1")
            rd1 = tvm.te.reduce_axis([0, D], name=prefix + "_rd1")
            rh1 = tvm.te.reduce_axis([0, H], name=prefix + "_rh1")
            rw1 = tvm.te.reduce_axis([0, W], name=prefix + "_rw1")
            return compute([C],
                           lambda c: tvm.te.sum(
                               inputs[rn1, c, rd1, rh1, rw1] / (N*D*H*W), axis=[rn1, rd1, rh1, rw1]),
                           name=prefix + "_bn3d_mean",
                           # tag=tag_gen("bn2d_mean"),
                           requires_grad=requires_grad)
        mean = GraphOp([C], [N, D, H, W], [inputs], _inner_mean,
                       name=prefix + "_mean", requires_grad=False)

        def _inner_square(C, N, D, H, W, inputs, requires_grad=False):
            rn2 = tvm.te.reduce_axis([0, N], name=prefix + "_rn2")
            rd2 = tvm.te.reduce_axis([0, D], name=prefix + "_rd2")
            rh2 = tvm.te.reduce_axis([0, H], name=prefix + "_rh2")
            rw2 = tvm.te.reduce_axis([0, W], name=prefix + "_rw2")
            return compute([N, C, D, H, W],
                           lambda c: tvm.te.sum(
                (inputs[rn2, c, rd2, rh2, rw2] * inputs[rn2, c, rd2, rh2, rw2]) / (N*H*W), axis=[rn2, rd2, rh2, rw2]),
                name=prefix + "_bn3d_square",
                # tag=tag_gen("bn2d_mean"),
                requires_grad=requires_grad)
        square = GraphOp([C], [N, D, H, W], [inputs], _inner_square,
                         name=prefix + "_square", requires_grad=False)

        def _inner_var(C, square, mean, requires_grad=False):
            return compute([C],
                           lambda c: square[c] - mean[c] * mean[c],
                           name=prefix + "_bn3d_var",
                           # tag=tag_gen("bn2d_var"),
                           requires_grad=requires_grad)
        var = GraphOp([C], [], [square, mean], _inner_var,
                      name=prefix + "_var", requires_grad=False)

        def _inner_bn(N, C, D, H, W, inputs, mean, var, alpha, beta, requires_grad=True):
            return compute([N, C, D, H, W],
                           lambda n, c, d, i, j: (
                               inputs[n, c, d, i, j] - mean[c]) / tvm.te.sqrt(var[c] + epsilon) * alpha[c] + beta[c],
                           name=prefix + "_bn3d",
                           # tag=tag_gen("bn2d"),
                           requires_grad=requires_grad)
        # return (mean, var), (inputs_p, mean_p, var_p), bn
        return GraphOp([N, C, D, H, W], [], [inputs, mean, var, alpha, beta], _inner_bn, name=prefix+"batch_norm")


def zero_pad1d(inputs, padding=0):
    """Zero padding for 1d tensor
    Args:
    -----------------------------
    inputs : GraphNode
        shape [batch, channel, length]
    padding: (optional:0) int or tuple
    -----------------------------
    Returns:
    -----------------------------
    GraphOp
        shape [batch, channel, padded_length]
    -----------------------------
    """
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    assert(isinstance(padding, tuple))
    assert(len(padding) == 2)

    padding_zero = tvm.tir.expr.const(0, inputs.tvm_tensor.dtype)

    batch_size, in_channel, in_len = inputs.shape

    def _inner_zero_pad1d(batch_size, in_channel, out_len, inputs, requires_grad=True):
        return compute((batch_size, in_channel, out_len),
                       lambda b, c, l: tvm.te.if_then_else(
            tvm.te.all(l >= padding[0], l < in_len + padding[0]),
            inputs[b, c, l - padding[0]],
            padding_zero
        ),
            name="zero_pad1d",
            # tag=tag_gen("zero_pad1d" + str(padding)),
            requires_grad=requires_grad)
    return GraphOp([batch_size, in_channel, in_len + padding[0] + padding[1]], [],
                   [inputs], _inner_zero_pad1d, name="zero_pad1d")


def zero_expand1d(inputs, stride=1):
    """Expand the inputs by zeros
    explain the expand operation:
    given stride = 2
    [1, 2, 3, 4, 5] --> expand [1, 0, 2, 0, 3, 0, 4, 0, 5]
    Args:
    -----------------------------
    inputs : GraphNode
        shape [batch, channel, length]
    stride: (optional:0) int or tuple
    -----------------------------

    Returns:
    -----------------------------
    GraphNode
        shape [batch, channel, (length - 1) * stride + 1]
    -----------------------------
    """
    stride = stride[0] if isinstance(stride, tuple) else stride
    assert(isinstance(stride, (int, tvm.tir.IntImm)))

    expand_zero = tvm.tir.expr.const(0, inputs.tvm_tensor.dtype)

    batch_size, in_channel, in_len = inputs.shape
    out_len = (in_len - 1) * stride + 1

    def _inner_zero_expand1d(batch_size, in_channel, out_len, inputs, requires_grad=True):
        return compute([batch_size, in_channel, out_len],
                       lambda b, c, l: tvm.te.if_then_else(
            l % stride == 0,
            inputs[b, c, l // stride],
            expand_zero),
            name="zero_expand1d",
            # tag=tag_gen("zero_expand1d" + str(stride)),
            requires_grad=requires_grad)
    return GraphOp([batch_size, in_channel, out_len], [], [inputs],
                   _inner_zero_expand1d, name="zero_expand1d")


def zero_pad2d(inputs, padding=0):
    """
    Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : GraphNode
        shape [batch, channel, height, width]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, channel, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert (len(padding) == 4)
    batch_size, in_channel, height, width = inputs.shape
    padded_shape = (batch_size, in_channel, height +
                    padding[0] + padding[1], width + padding[2] + padding[3])

    def _inner_zero_pad2d(batch_size, in_channel, h, w, inputs, requires_grad=True):
        # Warning, we use "float32" as type of 0
        padding_zero = tvm.tir.expr.const(0, inputs.tvm_tensor.dtype)
        return compute(padded_shape,
                       lambda b, c, h, w: tvm.te.if_then_else(
                           tvm.te.all(
                               h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                           inputs[b, c, h - padding[0], w - padding[2]],
                           padding_zero
                       ),
                       name="zero_pad2d",
                       # tag=tag_gen("zero_pad2d" + str(padding)),
                       requires_grad=requires_grad)
    return GraphOp(padded_shape, [], [inputs], _inner_zero_pad2d, name="zero_pad2d")


def zero_pad2d_nchwc(inputs, padding=0):
    """Zero padding for 2d tensor of NCHWc layout
    Args:
    -----------------------------
    inputs : GraphNode
        shape [batch, channel_chunk, height, width, channel_block]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    GraphNode
        shape [batch, channel_chunk, padded_height, padded_width, channel_block]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple)
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert(len(padding) == 4)

    padding_zero = tvm.tir.expr.const(0, inputs.tvm_tensor.dtype)

    batch_size, in_channel_chunk, height, width, in_channel_block = inputs.shape
    def _inner_zero_pad2d_nchwc(batch_size, in_channel_chunk, padded_height, padded_width, in_channel_block,
                                inputs, requires_grad=True):
        return compute([batch_size, in_channel_chunk, padded_height, padded_width, in_channel_block],
                       lambda b, c_c, h, w, c_b: tvm.te.if_then_else(
            tvm.te.all(h >= padding[0], h < height + padding[0],
                       w >= padding[2], w < width + padding[2]),
            inputs[b, c_c, h - padding[0], w - padding[2], c_b],
            padding_zero),
            name="zero_pad2d_nchwc",
            # tag=tag_gen("zero_pad2d_nchwc" + str(padding)),
            requires_grad=requires_grad)
    return GraphOp([batch_size, in_channel_chunk, height + padding[0] + padding[1], width + padding[2] + padding[3], in_channel_block],
                   [], [inputs], _inner_zero_pad2d_nchwc, name="zero_pad2d_nchwc")


def zero_pad3d(inputs, padding=0):
    """Zero padding for 3d GraphNode
    Args:
    -----------------------------
    inputs : GraphNode
        shape [batch, channel, depth, height, width]
    padding: (optional:0) int or tuple
        expected: (d_pad_up, d_pad_down, h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, channel, padded_depth, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding, padding, padding) \
        if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert(isinstance(padding, tuple))
    if len(padding) == 3:
        padding = (padding[0], padding[0], padding[1],
                   padding[1], padding[2], padding[2])
    assert(len(padding) == 6)

    padding_zero = tvm.tir.expr.const(0, inputs.tvm_tensor.dtype)

    batch_size, in_channel, depth, height, width = inputs.shape
    def _inner_zero_pad3d(batch_size, in_channel, padded_depth, padded_height, padded_width,
                          inputs, requires_grad=True):
        return compute([batch_size, in_channel, padded_depth, padded_height, padded_width],
                       lambda b, c, d, h, w: tvm.te.if_then_else(
            tvm.te.all(d >= padding[0], d < depth + padding[0], h >= padding[2],
                       h < height + padding[2], w >= padding[4], w < width + padding[4]),
            inputs[b, c, d - padding[0], h - padding[2], w - padding[4]],
            padding_zero),
            name="zero_pad3d",
            # tag=tag_gen("zero_pad3d" + str(padding)),
            requires_grad=requires_grad)
    return GraphOp([batch_size, in_channel, depth + padding[0] + padding[1], height + padding[2] + padding[3], width + padding[4] + padding[5]],
                   [], [inputs], _inner_zero_pad3d, name="zero_pad3d")


def zero_expand2d(inputs, stride=1):
    """Expand the inputs by zeros
    explain the expand operation:
    given stride = 2
    [[1, 2]      [[1, 0, 2]
     [3, 4]] -->  [0, 0, 0]
                  [3, 0, 4]]
    Args:
    -----------------------------
    inputs : GraphNode
        shape [batch, channel, height, width]
    stride: (optional:0) int or tuple
        expected: (h_stride, w_stride)
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, channel, (height - 1) * h_stride + 1, (width - 1) * w_stride + 1]
    -----------------------------
    """
    stride = (stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    assert(isinstance(stride, tuple))
    assert(len(stride) == 2)

    expand_zero = tvm.tir.expr.const(0, inputs.tvm_tensor.dtype)

    batch_size, in_channel, height, width = inputs.shape
    out_height = (height - 1) * stride[0] + 1
    out_width = (width - 1) * stride[1] + 1

    def _inner_zero_expand2d(batch_size, in_channel, out_height, out_width, inputs, requires_grad=True):
        return compute([batch_size, in_channel, out_height, out_width],
                       lambda b, c, h, w: tvm.te.if_then_else(
            tvm.te.all(
                h % stride[0] == 0,
                w % stride[1] == 0
            ),
            inputs[b, c, h // stride[0], w // stride[1]],
            expand_zero),
            name="zero_expand2d",
            # tag=tag_gen("zero_expand2d" + str(stride)),
            requires_grad=requires_grad)
    return GraphOp([batch_size, in_channel, out_height, out_width], [], [inputs],
                   _inner_zero_expand2d, name="zero_expand2d")


def zero_expand3d(inputs, stride=1):
    """Expand the inputs by zeros
    explain the expand operation:
    given stride = 2
    [[[1, 2] --> [[[1, 0, 2]
      [3, 4]]      [0, 0, 0]                    
                   [3, 0, 4]]
     [[5, 6]        
      [7, 8]]]    [[0, 0, 0] 
                   [0, 0, 0] 
                   [0, 0, 0]]
                  [[5, 0, 6]
                   [0, 0, 0]
                   [7, 0, 8]]]
    Args:
    -----------------------------
    inputs : GraphNode
        shape [batch, channel, depth, height, width]
    stride: (optional:0) int or tuple
        expected: (d_stride, h_stride, w_stride)
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, channel, (depth - 1) * d_stride + 1, (height - 1) * h_stride + 1, (width - 1) * w_stride + 1]
    -----------------------------
    """
    stride = (stride, stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    assert(isinstance(stride, tuple))
    assert(len(stride) == 3)

    expand_zero = tvm.tir.expr.const(0, inputs.tvm_tensor.dtype)

    batch_size, in_channel, depth, height, width = inputs.shape
    out_depth = (depth - 1) * stride[0] + 1
    out_height = (height - 1) * stride[1] + 1
    out_width = (width - 1) * stride[2] + 1

    def _inner_zero_expand3d(batch_size, in_channel, out_depth, out_height, out_width, inputs, requires_grad=True):
        return compute((batch_size, in_channel, out_depth, out_height, out_width),
                       lambda b, c, d, h, w: tvm.te.if_then_else(
            tvm.te.all(
                d % stride[0] == 0,
                h % stride[1] == 0,
                w % stride[2] == 0
            ),
            inputs[b, c, d // stride[0], h // stride[1], w // stride[2]],
            expand_zero),
            name="zero_expand3d",
            # tag=tag_gen("zero_expand3d" + str(stride)),
            requires_grad=requires_grad)
    return GraphOp([batch_size, in_channel, out_depth, out_height, out_width], [], [inputs],
                   _inner_zero_expand3d, name="zero_expand3d")


def conv1d(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out_dtype="float32"):
    """Convolution 1d
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, length]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_length]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    dilation: (optional:1) int
    groups  : (optional:1) int

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_length]
    -----------------------------
    """
    batch_size, in_channel, in_len = inputs.shape
    out_channel, channel_per_group, k_len = weight.shape
    assert(isinstance(groups, (int, tvm.tir.IntImm)))
    assert((channel_per_group * groups) == in_channel)
    assert((out_channel % groups) == 0)
    out_channel_per_group = out_channel // groups

    stride = stride[0] if isinstance(stride, tuple) else stride
    padding = padding[0] if isinstance(padding, tuple) else padding
    assert(isinstance(stride, (int, tvm.tir.IntImm)))
    assert(isinstance(padding, (int, tvm.tir.IntImm)))
    assert(isinstance(dilation, (int, tvm.tir.IntImm)))

    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1

    padded = zero_pad1d(inputs, padding=padding)

    def _inner_conv1d(batch_size, out_channel, out_len, channel_per_group, k_len, padded, requires_grad=True):
        rc = tvm.te.reduce_axis((0, channel_per_group))
        rl = tvm.te.reduce_axis((0, k_len))
        return compute([batch_size, out_channel, out_len],
                       lambda b, c, l: tvm.te.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, l * stride + rl * dilation] *
             weight[c, rc, rl]).astype(out_dtype),
            axis=[rc, rl]),
            name="conv1d",
            # tag=tag_gen("conv1d" + str(stride) + str(dilation) + str(groups)),
            requires_grad=requires_grad)
    conved = GraphOp([batch_size, out_channel, out_len], [channel_per_group, k_len],
                     [padded], _inner_conv1d, name="conv1d")
    if bias is not None:
        out_channel2 = bias.shape
        assert bias.shape == 1 and out_channel2 == out_channel

        def _inner_bias_add(batch_size, out_channel, out_len, conved, bias, requires_grad=True):
            return compute([batch_size, out_channel, out_len],
                           lambda b, c, l: conved[b, c, l] + bias[c],
                           name="conv1d_bias",
                           # tag=tag_gen("conv1d_bias"),
                           requires_grad=requires_grad)
        conved_bias = GraphOp([batch_size, out_channel, out_len], [], [
                              conved, bias], _inner_bias_add, name="bias")
        return conved_bias
    return conved


def conv_transpose1d(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, out_dtype="float32"):
    """Convolution transpose 1d
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, length]
    weight  : GraphNode
        shape [channel, out_channel // groups, kernel_length]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    output_padding : (optional:0) int or tuple
    groups  : (optional:1) int
    dilation: (optional:1) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, out_channel, output_length]
    -----------------------------
    """
    batch_size, input_channel, length = inputs.shape
    input_channel_w, channel_per_group, k_len = weight.shape
    assert(input_channel == input_channel_w)
    in_channel_per_group = input_channel // groups
    assert(in_channel_per_group * groups == input_channel)
    output_channel = channel_per_group * groups

    stride = stride[0] if isinstance(stride, tuple) else stride
    padding = padding[0] if isinstance(padding, tuple) else padding
    output_padding = output_padding[0] if isinstance(
        output_padding, tuple) else output_padding
    assert(isinstance(stride, (int, tvm.tir.IntImm)))
    assert(isinstance(padding, (int, tvm.tir.IntImm)))
    assert(isinstance(output_padding, (int, tvm.tir.IntImm)))
    assert(isinstance(groups, (int, tvm.tir.IntImm)))
    assert(isinstance(dilation, (int, tvm.tir.IntImm)))

    kernel_size = (k_len - 1) * dilation + 1
    output_len = (length - 1) * stride - 2 * \
        padding + kernel_size + output_padding

    expanded = zero_expand1d(inputs, stride=stride)
    padded = zero_pad1d(expanded, padding=(kernel_size - 1 - padding,
                                           kernel_size - 1 - padding + output_padding))

    def _inner_conv_transpose1d(batch_size, output_channel, output_len, in_channel_per_group, k_len,
                                padded, weight, requires_grad=True):
        rc = tvm.te.reduce_axis((0, in_channel_per_group))
        rl = tvm.te.reduce_axis((0, k_len))
        return compute([batch_size, output_channel, output_len],
                       lambda b, c, l: tvm.te.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, l + rl * dilation] *
             weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_len - rl - 1]).astype(out_dtype),
            axis=[rc, rl]),
            name="conv_transpose1d",
            # tag=tag_gen("conv_transpose1d" + str(stride) + str(dilation) + str(groups)),
            requires_grad=requires_grad)
    convtrans = GraphOp([batch_size, output_channel, output_len], [], [padded, weight],
                        _inner_conv_transpose1d, name="conv_transpose1d")
    if bias is not None:
        def _inner_bias_add(batch_size, output_channel, output_len, output, bias, requires_grad=True):
            return compute([batch_size, output_channel, output_len],
                           lambda b, c, l: output[b, c, l] + bias[c],
                           name="conv_transpose1d_bias",
                           # tag=tag_gen("conv_transpose1d_bias"),
                           requires_grad=requires_grad)
        return GraphOp([batch_size, output_channel, output_len], [], [convtrans, bias],
                       _inner_bias_add, name="conv_transpose1d_bias")
    return convtrans


def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0,
    dilation=1, out_dtype="float32"):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape

    assert in_channel == channel_per_group

    stride = (stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0]
             * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1]
             * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)
    conv_out_shape = (batch_size, out_channel, out_h, out_w)
    def _inner_conv2d_nchw(batch_size, out_channel, out_h, out_w, channel_per_group,
                           k_w, k_h, padded, weight, requires_grad=True):
        rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
        rw = tvm.te.reduce_axis((0, k_w), name="rw")
        rh = tvm.te.reduce_axis((0, k_h), name="rh")
        return compute(conv_out_shape,
                       lambda b, c, h, w: tvm.te.sum(
                           (padded[b, rc,
                                   h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
                            * weight[c, rc, rh, rw]).astype(out_dtype),
                           axis=[rc, rw, rh]),
                       name="conv2d_nchw",
                       # tag=tag_gen("conv2d_nchw" + str(stride) + str(dilation) + str(groups)),
                       requires_grad=requires_grad
                       )
    conv_out = GraphOp(
        conv_out_shape,
        [channel_per_group, k_w, k_h],
        [padded, weight],
        _inner_conv2d_nchw,
        name="conv2d_nchw")

    def _inner_bias(batch_size, out_channel, out_h, out_w, conv_out, bias, requires_grad=True):
        return compute(
                (batch_size, out_channel, out_h, out_w),
                lambda b, c, h, w: conv_out[b, c, h, w] + bias[c],
                name="conv2d_nchw_bias",
                requires_grad=requires_grad)
    if bias is not None:
        return GraphOp(conv_out_shape, [], [conv_out, bias], _inner_bias, name="conv2d_bias")
    return conv_out


def conv2d_nchw_grouped(
    inputs, weight, bias=None, stride=1, padding=0, dilation=1,
    groups=1, out_dtype="float32"):
    """Convolution 2d NCHW layout, grouped

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert channel_per_group * \
        groups == in_channel, "%d vs. %d" % (
            channel_per_group * groups, in_channel)
    out_channel_per_group = out_channel // groups
    assert out_channel_per_group * groups == out_channel

    stride = (stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0]
             * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1]
             * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)

    def _inner_reshape_data(N, G, C, H, W, data, requires_grad=True):
        return compute([N, G, C, H, W],
                       lambda n, g, c, h, w: data[n, g *
                                                  channel_per_group + c, h, w],
                       name="reshape_data",
                       requires_grad=requires_grad)

    data_reshape = GraphOp(
        [batch_size, groups, channel_per_group, in_h +
            2 * padding[0], in_w + 2 * padding[1]],
        [],
        [padded],
        _inner_reshape_data,
        name="reshape_data"
    )

    def _inner_reshape_kernel(G, K, C, R, S, kernel, requires_grad=True):
        return compute([G, K, C, R, S],
                       lambda g, k, c, r, s: kernel[g *
                                                    out_channel_per_group + k, c, r, s],
                       name="reshape_kernel",
                       requires_grad=requires_grad)

    kernel_reshape = GraphOp(
        [groups, out_channel_per_group, channel_per_group, k_h, k_w],
        [],
        [weight],
        _inner_reshape_kernel,
        name="reshape_kernel"
    )

    conv_out_shape = (batch_size, groups, out_channel_per_group, out_h, out_w)

    def _inner_grouped_nchw(batch_size, groups, out_channel_per_group, out_h, out_w, channel_per_group,
                            k_w, k_h, data, kernel, requires_grad=True):
        rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
        rw = tvm.te.reduce_axis((0, k_w), name="rw")
        rh = tvm.te.reduce_axis((0, k_h), name="rh")
        return compute(conv_out_shape,
                       lambda b, g, k, h, w: tvm.te.sum(
                           (data[b, g, rc,
                                 h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
                            * kernel[g, k, rc, rh, rw]).astype(out_dtype),
                           axis=[rc, rw, rh]),
                       name="conv2d_nchw_grouped",
                       requires_grad=requires_grad
                       )
    conv_out = GraphOp(
        conv_out_shape,
        [channel_per_group, k_w, k_h],
        [data_reshape, kernel_reshape],
        _inner_grouped_nchw,
        name="conv2d_nchw_grouped")

    def _inner_reshape_output(batch_size, out_channel, out_h, out_w,
                              conv_out, requires_grad=True):
        return compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: conv_out[b, c //
                                        out_channel_per_group, c % out_channel_per_group, h, w],
            name="reshape_output",
            requires_grad=requires_grad
        )

    output = GraphOp(
        [batch_size, out_channel, out_h, out_w],
        [],
        [conv_out],
        _inner_reshape_output,
        name="reshape_output")

    def _inner_bias(batch_size, out_channel, out_h, out_w, output, bias, requires_grad=True):
        return compute(
                (batch_size, out_channel, out_h, out_w),
                lambda b, c, h, w: output[b, c, h, w] + bias[c],
                name="conv2d_nchw_bias",
                requires_grad=requires_grad)
    if bias is not None:
        return GraphOp(
            [batch_size, out_channel, out_h, out_w],
            [], [output, bias], _inner_bias, name="conv2d_grouped_bias")

    return output


def conv2d_capsule(inputs, weight, bias=None,
    stride=1, padding=0, dilation=1, num_caps=8, out_dtype="float32"):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width, num_caps]
    bias    : (optional:None) GraphNode
        shape [out_channel, num_caps]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    num_caps : (optional:8) int
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_channel, output_height, output_width, num_caps]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w, num_caps_ = weight.shape
    assert channel_per_group == in_channel
    assert num_caps_ == num_caps

    stride = (stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0]
             * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1]
             * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)
    conv_out_shape = (batch_size, out_channel, out_h, out_w, num_caps)

    def _inner_conv2d_nchw(batch_size, out_channel, out_h, out_w, num_caps, channel_per_group,
                           k_w, k_h, padded, weight, requires_grad=True):
        rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
        rw = tvm.te.reduce_axis((0, k_w), name="rw")
        rh = tvm.te.reduce_axis((0, k_h), name="rh")
        return compute(conv_out_shape,
                       lambda b, c, h, w, s: tvm.te.sum(
                           (padded[b, rc,
                                   h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
                            * weight[c, rc, rh, rw, s]).astype(out_dtype),
                           axis=[rc, rw, rh]),
                       name="conv2d_capsule",
                       # tag=tag_gen("conv2d_nchw" + str(stride) + str(dilation) + str(groups)),
                       requires_grad=requires_grad
                       )
    conv_out = GraphOp(
        conv_out_shape,
        [channel_per_group, k_w, k_h],
        [padded, weight],
        _inner_conv2d_nchw,
        name="conv2d_capsule")

    def _inner_bias(batch_size, out_channel, out_h, out_w, num_caps, conv_out, bias, requires_grad=True):
        return compute(
            (batch_size, out_channel, out_h, out_w, num_caps),
            lambda b, c, h, w, s: conv_out[b, c, h, w, s] + bias[c, s],
            name="conv2d_capsule_bias",
            # tag=tag_gen("conv2d_capsule_bias"),
            requires_grad=requires_grad
        )
    if bias is not None:
        return GraphOp(conv_out_shape, [], [conv_out, bias], _inner_bias, name="conv2d_capsule_bias")
    return conv_out


def conv2d_nchwc(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out_dtype="float32"):
    """Convolution 2d NCHWc layout
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel // vlen1, height, width, vlen1]
    weight  : GraphNode
        shape [out_channel // vlen2, channel // vlen1 // groups, kernel_height, kernel_width, vlen1, vlen2]
    bias    : (optional:None) GraphNode
        shape [out_channel // vlen2, vlen2]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    dilation: (optional:1) int
    groups  : (optional:1) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, out_channel // vlen2, output_height, output_width, vlen2]
    -----------------------------
    """
    batch_size, in_channel_chunk, in_h, in_w, in_channel_block = inputs.shape
    out_channel_chunk, channel_per_group_chunk, k_h, k_w, _in_channel_block, out_channel_block = weight.shape
    assert (channel_per_group_chunk * groups == in_channel_chunk)
    assert _in_channel_block == in_channel_block
    out_channel_per_group = out_channel_chunk // groups
    assert (out_channel_per_group * groups == out_channel_chunk)

    stride = (stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert (isinstance(stride, tuple) and len(stride) == 2)
    assert (isinstance(padding, tuple) and len(padding) == 2)
    assert (isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0]
             * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1]
             * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d_nchwc(inputs, padding=padding)

    def _inner_conv2d_nchwc(batch_size, out_channel_chunk, out_h, out_w, out_channel_block,
                            channel_per_group_chunk, in_channel_block, k_w, k_h, padded, weight, requires_grad=True):
        rc_chunk = tvm.te.reduce_axis(
            (0, channel_per_group_chunk), name="rc_chunk")
        rc_block = tvm.te.reduce_axis((0, in_channel_block), name="rc_block")
        rw = tvm.te.reduce_axis((0, k_w))
        rh = tvm.te.reduce_axis((0, k_h))
        return compute([batch_size, out_channel_chunk, out_h, out_w, out_channel_block],
                       lambda b, c_c, h, w, c_b: tvm.te.sum(
            (padded[b, c_c // out_channel_per_group * channel_per_group_chunk + rc_chunk,
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1], rc_block]
             * weight[c_c, rc_chunk, rh, rw, rc_block, c_b]).astype(out_dtype),
            axis=[rc_chunk, rc_block, rw, rh]),
            name="conv2d_nchwc",
            # tag=tag_gen("conv2d_nchwc" + str(stride) + str(dilation) + str(groups)),
            requires_grad=requires_grad)
    conv_result = GraphOp([batch_size, out_channel_chunk, out_h, out_w, out_channel_block],
                          [channel_per_group_chunk, in_channel_block,
                              k_w, k_h], [padded, weight],
                          _inner_conv2d_nchwc, name="conv2d_nchwc")
    if bias is not None:
        def _inner_bias_add(batch_size, out_channel_chunk, out_h, out_w, out_channel_block, output, bias, requires_grad=True):
            return compute([batch_size, out_channel_chunk, out_h, out_w, out_channel_block],
                           lambda b, c_c, h, w, c_b: output[b,
                                                            c_c, h, w, c_b] + bias[c_c, c_b],
                           name="conv2d_nchwc_bias",
                           # tag=tag_gen("conv2d_nchwc_bias"),
                           requires_grad=requires_grad)
        return GraphOp([batch_size, out_channel_chunk, out_h, out_w, out_channel_block], [],
                       [conv_result, bias], _inner_bias_add, name="conv2d_nchwc_bias")
    return conv_result


def conv_transpose2d_nchw(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, out_dtype="float32"):
    """Convolution transpose 2d NCHW layout
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : tGraphNode
        shape [channel, out_channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    output_padding : (optional:0) int or tuple
    groups  : (optional:1) int
    dilation: (optional:1) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, input_channel, in_h, in_w = inputs.shape
    input_channel_w, channel_per_group, k_h, k_w = weight.shape
    assert(input_channel == input_channel_w)
    in_channel_per_group = input_channel // groups
    assert(in_channel_per_group * groups == input_channel)
    output_channel = channel_per_group * groups

    stride = (stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    output_padding = ((output_padding, output_padding)
                      if isinstance(output_padding, (int, tvm.tir.IntImm)) else output_padding)
    dilation = (dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert(isinstance(stride, tuple) and len(stride) == 2)
    assert(isinstance(padding, tuple) and len(padding) == 2)
    assert(isinstance(output_padding, tuple) and len(output_padding) == 2)
    assert(isinstance(groups, (int, tvm.tir.IntImm)))
    assert(isinstance(dilation, tuple) and len(dilation) == 2)

    kernel_h = (k_h - 1) * dilation[0] + 1
    kernel_w = (k_w - 1) * dilation[1] + 1
    out_h = (in_h - 1) * stride[0] - 2 * \
        padding[0] + kernel_h + output_padding[0]
    out_w = (in_w - 1) * stride[1] - 2 * \
        padding[1] + kernel_w + output_padding[1]

    expanded = zero_expand2d(inputs, stride=stride)
    padded = zero_pad2d(expanded, padding=(
        kernel_h - 1 - padding[0],
        kernel_h - 1 - padding[0] + output_padding[0],
        kernel_w - 1 - padding[1],
        kernel_w - 1 - padding[1] + output_padding[1]))

    def _inner_conv_transpose2d_nchw(batch_size, output_channel, out_h, out_w,
                                     in_channel_per_group, k_h, k_w, padded, weight, requires_grad=True):
        rc = tvm.te.reduce_axis((0, in_channel_per_group))
        rh = tvm.te.reduce_axis((0, k_h))
        rw = tvm.te.reduce_axis((0, k_w))
        return compute([batch_size, output_channel, out_h, out_w],
                       lambda b, c, h, w: tvm.te.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, h + rh * dilation[0], w + rw * dilation[1]] *
             weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_h - rh - 1, k_w - rw - 1]).astype(out_dtype),
            axis=[rc, rw, rh]),
            name="conv_transpose2d_nchw",
            # tag=tag_gen("conv_transpose2d_nchw" + str(stride) + str(dilation) + str(groups)),
            requires_grad=requires_grad)
    conv_result = GraphOp([batch_size, output_channel, out_h, out_w], [in_channel_per_group, k_h, k_w],
                          [padded, weight], _inner_conv_transpose2d_nchw, name="conv_transpose2d_nchw")
    if bias is not None:
        def _inner_bias_add(batch_size, output_channel, out_h, out_w, output, bias, requires_grad=True):
            return compute([batch_size, output_channel, out_h, out_w],
                           lambda b, c, h, w: output[b, c, h, w] + bias[c],
                           name="conv_transpose2d_nchw_bias",
                           # tag=tag_gen("conv_transpose2d_nchw_bias"),
                           requires_grad=requires_grad)
        return GraphOp([batch_size, output_channel, out_h, out_w], [],
                       [conv_result, bias], _inner_bias_add, name="conv_transpose2d_nchw_bias")
    return conv_result


def depthwise_conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, out_dtype="float32"):
    """Depthwise convolution 2d NCHW layout
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, height, width]
    weight  : GraphNode
        shape [in_channel, factor, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    dilation: (optional:1) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    _in_channel, factor, k_h, k_w = weight.shape
    assert(_in_channel == in_channel)
    out_channel = in_channel * factor

    stride = (stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert(isinstance(stride, tuple) and len(stride) == 2)
    assert(isinstance(padding, tuple) and len(padding) == 2)
    assert(isinstance(dilation, tuple) and len(dilation) == 2)

    out_h = (in_h + 2 * padding[0] - dilation[0]
             * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1]
             * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)

    def _inner_depthwise_conv2d_nchw(batch_size, out_channel, out_h, out_w, k_h, k_w,
                                     padded, weight, requires_grad=True):
        rh = tvm.te.reduce_axis((0, k_h))
        rw = tvm.te.reduce_axis((0, k_w))
        return compute([batch_size, out_channel, out_h, out_w],
                       lambda b, c, h, w: tvm.te.sum(
            (padded[b, c//factor,
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
             * weight[c//factor, c % factor, rh, rw]).astype(out_dtype),
            axis=[rh, rw]),
            name="depthwise_conv2d_nchw",
            # tag=tag_gen("depthwise_conv2d_nchw" + str(stride) + str(dilation) + str(factor)),
            requires_grad=requires_grad)
    conv_result = GraphOp([batch_size, out_channel, out_h, out_w], [k_h, k_w], [padded, weight],
                          _inner_depthwise_conv2d_nchw, name="depthwise_conv2d_nchw")
    if bias is not None:
        def _inner_bias_add(batch_size, out_channel, out_h, out_w, output, bias, requires_grad=True):
            return compute((batch_size, out_channel, out_h, out_w),
                           lambda b, c, h, w: output[b, c, h, w] + bias[c],
                           name="depthwise_conv2d_nchw_bias",
                           # tag=tag_gen("depthwise_conv2d_nchw_bias"),
                           requires_grad=requires_grad)
        return GraphOp([batch_size, out_channel, out_h, out_w], [], [conv_result, bias],
                       _inner_bias_add, name="depthwise_conv2d_nchw_bias")
    return conv_result


def conv3d_ncdhw(inputs, weight, bias=None,
    stride=1, padding=0, dilation=1, out_dtype="float32"):
    """Convolution 3d NCDHW layout
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, depth, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_depth, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    dilation: (optional:1) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, out_channel, output_depth, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_d, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_d, k_h, k_w = weight.shape
    assert in_channel == channel_per_group

    stride = (stride, stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert(isinstance(stride, tuple) and len(stride) == 3)
    assert(isinstance(padding, tuple) and len(padding) == 3)
    assert(isinstance(dilation, tuple) and len(dilation) == 3)

    out_d = (in_d + 2 * padding[0] - dilation[0]
             * (k_d - 1) - 1) // stride[0] + 1
    out_h = (in_h + 2 * padding[1] - dilation[1]
             * (k_h - 1) - 1) // stride[1] + 1
    out_w = (in_w + 2 * padding[2] - dilation[2]
             * (k_w - 1) - 1) // stride[2] + 1

    padded = zero_pad3d(inputs, padding=padding)

    def _inner_conv3d_ncdhw(batch_size, out_channel, out_d, out_h, out_w,
                            channel_per_group, k_d, k_h, k_w, padded, weight, requires_grad=True):
        rc = tvm.te.reduce_axis((0, channel_per_group))
        rd = tvm.te.reduce_axis((0, k_d))
        rh = tvm.te.reduce_axis((0, k_h))
        rw = tvm.te.reduce_axis((0, k_w))
        return compute([batch_size, out_channel, out_d, out_h, out_w],
                       lambda b, c, d, h, w: tvm.te.sum(
            (padded[b, rc,
                    d * stride[0] + rd * dilation[0], h * stride[1] + rh * dilation[1], w * stride[2] + rw * dilation[2]]
             * weight[c, rc, rd, rh, rw]).astype(out_dtype),
            axis=[rc, rd, rh, rw]),
            name="conv3d_ncdhw",
            # tag=tag_gen("conv3d_ncdhw" + str(stride) + str(dilation) + str(groups)),
            requires_grad=requires_grad)
    conv_result = GraphOp([batch_size, out_channel, out_d, out_h, out_w], [channel_per_group, k_d, k_h, k_w],
                          [padded, weight], _inner_conv3d_ncdhw, name="conv3d_ncdhw")
    if bias is not None:
        def _inner_bias_add(batch_size, out_channel, out_d, out_h, out_w, output, bias, requires_grad=True):
            return compute([batch_size, out_channel, out_d, out_h, out_w],
                           lambda b, c, d, h, w: output[b,
                                                        c, d, h, w] + bias[c],
                           name="conv3d_ncdhw_bias",
                           # tag=tag_gen("conv3d_ncdhw_bias"),
                           requires_grad=requires_grad)
        return GraphOp([batch_size, out_channel, out_d, out_h, out_w], [],
                       [conv_result, bias], _inner_bias_add, name="conv3d_ncdhw_bias")
    return conv_result


def conv3d_ncdhw_grouped(inputs, weight, bias=None,
    stride=1, padding=0, dilation=1, groups=1, out_dtype="float32"):
    """Convolution 3d NCDHW layout
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, depth, height, width]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_depth, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    dilation: (optional:1) int
    groups  : (optional:1) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, out_channel, output_depth, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_d, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_d, k_h, k_w = weight.shape
    assert(channel_per_group * groups == in_channel)
    out_channel_per_group = out_channel // groups
    assert(out_channel_per_group * groups == out_channel)

    stride = (stride, stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert(isinstance(stride, tuple) and len(stride) == 3)
    assert(isinstance(padding, tuple) and len(padding) == 3)
    assert(isinstance(dilation, tuple) and len(dilation) == 3)

    out_d = (in_d + 2 * padding[0] - dilation[0]
             * (k_d - 1) - 1) // stride[0] + 1
    out_h = (in_h + 2 * padding[1] - dilation[1]
             * (k_h - 1) - 1) // stride[1] + 1
    out_w = (in_w + 2 * padding[2] - dilation[2]
             * (k_w - 1) - 1) // stride[2] + 1

    padded = zero_pad3d(inputs, padding=padding)

    def _inner_reshape_data(N, G, C, D, H, W, data, requires_grad=True):
        return compute([N, G, D, C, H, W],
                       lambda n, g, c, d, h, w: data[n, g *
                                                  channel_per_group + c, d, h, w],
                       name="reshape_data",
                       requires_grad=requires_grad)

    data_reshape = GraphOp(
        [batch_size, groups, channel_per_group, in_d + 2 * padding[0], 
        in_h + 2 * padding[1], in_w + 2 * padding[2]],
        [],
        [padded],
        _inner_reshape_data,
        name="reshape_data"
    )

    def _inner_reshape_kernel(G, K, C, T, R, S, kernel, requires_grad=True):
        return compute([G, K, C, T, R, S],
                       lambda g, k, c, t, r, s: kernel[g *
                                                    out_channel_per_group + k, c, t, r, s],
                       name="reshape_kernel",
                       requires_grad=requires_grad)

    kernel_reshape = GraphOp(
        [groups, out_channel_per_group, channel_per_group, k_d, k_h, k_w],
        [],
        [weight],
        _inner_reshape_kernel,
        name="reshape_kernel"
    )

    def _inner_conv3d_ncdhw(batch_size, out_channel, out_d, out_h, out_w,
                            channel_per_group, k_d, k_h, k_w, data, kernel, requires_grad=True):
        rc = tvm.te.reduce_axis((0, channel_per_group))
        rd = tvm.te.reduce_axis((0, k_d))
        rh = tvm.te.reduce_axis((0, k_h))
        rw = tvm.te.reduce_axis((0, k_w))
        return compute([batch_size, groups, out_channel, out_d, out_h, out_w],
                       lambda b, g, k, d, h, w: tvm.te.sum(
            (data[b, g, rc,
                    d * stride[0] + rd * dilation[0], h * stride[1] + rh * dilation[1], w * stride[2] + rw * dilation[2]]
             * kernel[g, k, rc, rd, rh, rw]).astype(out_dtype),
            axis=[rc, rd, rh, rw]),
            name="conv3d_ncdhw",
            requires_grad=requires_grad)
    conv_result = GraphOp(
        [batch_size, groups, out_channel_per_group, out_d, out_h, out_w],
        [channel_per_group, k_d, k_h, k_w],
        [data_reshape, kernel_reshape],
        _inner_conv3d_ncdhw, name="conv3d_ncdhw")

    def _inner_reshape_output(batch_size, groups, out_channel, out_d, out_h, out_w,
                              conv_out, requires_grad=True):
        return compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, d, h, w: conv_out[b, c //
                                        out_channel_per_group, c % out_channel_per_group, d, h, w],
            name="reshape_output",
            requires_grad=requires_grad
        )

    output = GraphOp(
        [batch_size, out_channel, out_d, out_h, out_w],
        [],
        [conv_result],
        _inner_reshape_output,
        name="reshape_output")

    if bias is not None:
        def _inner_bias_add(batch_size, out_channel, out_d, out_h, out_w, output, bias, requires_grad=True):
            return compute([batch_size, out_channel, out_d, out_h, out_w],
                           lambda b, c, d, h, w: output[b,
                                                        c, d, h, w] + bias[c],
                           name="conv3d_ncdhw_bias",
                           # tag=tag_gen("conv3d_ncdhw_bias"),
                           requires_grad=requires_grad)
        return GraphOp([batch_size, out_channel, out_d, out_h, out_w], [],
                       [output, bias], _inner_bias_add, name="conv3d_ncdhw_bias")
    return output


def conv_transpose3d_ncdhw(inputs, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1, out_dtype="float32"):
    """Convolution transpose 3d NCDHW layout
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, channel, depth, height, width]
    weight  : GraphNode
        shape [channel, out_channel // groups, kernel_depth, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    output_padding : (optional:0) int or tuple
    groups  : (optional:1) int
    dilation: (optional:1) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, out_channel, out_depth, output_height, output_width]
    -----------------------------
    """
    batch_size, input_channel, in_d, in_h, in_w = inputs.shape
    input_channel_w, channel_per_group, k_d, k_h, k_w = weight.shape
    assert(input_channel == input_channel_w)
    in_channel_per_group = input_channel // groups
    assert(in_channel_per_group * groups == input_channel)
    output_channel = channel_per_group * groups

    stride = (stride, stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    output_padding = ((output_padding, output_padding, output_padding)
                      if isinstance(output_padding, (int, tvm.tir.IntImm)) else output_padding)
    dilation = (dilation, dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert(isinstance(stride, tuple) and len(stride) == 3)
    assert(isinstance(padding, tuple) and len(padding) == 3)
    assert(isinstance(output_padding, tuple) and len(output_padding) == 3)
    assert(isinstance(groups, (int, tvm.tir.IntImm)))
    assert(isinstance(dilation, tuple) and len(dilation) == 3)

    kernel_d = (k_d - 1) * dilation[0] + 1
    kernel_h = (k_h - 1) * dilation[1] + 1
    kernel_w = (k_w - 1) * dilation[2] + 1
    out_d = (in_d - 1) * stride[0] - 2 * \
        padding[0] + kernel_d + output_padding[0]
    out_h = (in_h - 1) * stride[1] - 2 * \
        padding[1] + kernel_h + output_padding[1]
    out_w = (in_w - 1) * stride[2] - 2 * \
        padding[2] + kernel_w + output_padding[2]

    expanded = zero_expand3d(inputs, stride=stride)
    padded = zero_pad3d(expanded, padding=(
        kernel_d - 1 - padding[0],
        kernel_d - 1 - padding[0] + output_padding[0],
        kernel_h - 1 - padding[1],
        kernel_h - 1 - padding[1] + output_padding[1],
        kernel_w - 1 - padding[2],
        kernel_w - 1 - padding[2] + output_padding[2]))

    def _inner_conv_transpose3d_ncdhw(batch_size, output_channel, out_d, out_h, out_w,
                                      in_channel_per_group, k_d, k_h, k_w, padded, weight, requires_grad=True):
        rc = tvm.te.reduce_axis((0, in_channel_per_group))
        rd = tvm.te.reduce_axis((0, k_d))
        rh = tvm.te.reduce_axis((0, k_h))
        rw = tvm.te.reduce_axis((0, k_w))
        return compute([batch_size, output_channel, out_d, out_h, out_w],
                       lambda b, c, d, h, w: tvm.te.sum(
            (padded[b, c // channel_per_group * in_channel_per_group + rc, d + rd * dilation[0], h + rh * dilation[1], w + rw * dilation[2]] *
             weight[c // channel_per_group * in_channel_per_group + rc, c % channel_per_group, k_d - rd - 1, k_h - rh - 1, k_w - rw - 1]).astype(out_dtype),
            axis=[rc, rd, rh, rw]),
            name="conv_transpose3d_ncdhw",
            # tag=tag_gen("conv_transpose3d_ncdhw" + str(groups)),
            requires_grad=requires_grad)
    conv_result = GraphOp([batch_size, output_channel, out_d, out_h, out_w], [in_channel_per_group, k_d, k_h, k_w],
                          [padded, weight], _inner_conv_transpose3d_ncdhw, name="conv_transpose3d_ncdhw")
    if bias is not None:
        def _inner_bias_add(batch_size, output_channel, out_d, out_h, out_w, output, bias, requires_grad=True):
            return compute([batch_size, output_channel, out_d, out_h, out_w],
                           lambda b, c, d, h, w: output[b,
                                                        c, d, h, w] + bias[c],
                           name="conv_transpose3d_ncdhw_bias",
                           # tag=tag_gen("conv_transpose3d_ncdhw_bias"),
                           requires_grad=requires_grad)
        return GraphOp([batch_size, output_channel, out_d, out_h, out_w], [],
                       [conv_result, bias], _inner_bias_add, name="conv_transpose3d_ncdhw_bias")
    return conv_result


def conv2d_nhwc(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out_dtype="float32"):
    """Convolution 2d NHWC layout
    Args:
    -----------------------------
    inputs  : GraphNode
        shape [batch, height, width, channel]
    weight  : GraphNode
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) GraphNode
        shape [out_channel]
    stride  : (optional:1) int or tuple
    padding : (optional:0) int or tuple
    dilation: (optional:1) int
    groups  : (optional:1) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, output_height, output_width, out_channel]
    -----------------------------
    """
    batch_size, in_h, in_w, in_channel = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert(channel_per_group * groups == in_channel)
    out_channel_per_group = out_channel // groups
    assert(out_channel_per_group * groups == out_channel)

    stride = (stride, stride) if isinstance(
        stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(
        dilation, (int, tvm.tir.IntImm)) else dilation
    assert(isinstance(stride, tuple) and len(stride) == 2)
    assert(isinstance(padding, tuple) and len(padding) == 2)
    assert(isinstance(dilation, tuple) and len(dilation) == 2)
    out_h = (in_h + 2 * padding[0] - dilation[0]
             * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1]
             * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding)

    def _inner_conv2d_nhwc(batch_size, out_h, out_w, out_channel, channel_per_group, k_h, k_w,
                           padded, weight, requires_grad=True):
        rc = tvm.te.reduce_axis((0, channel_per_group))
        rh = tvm.te.reduce_axis((0, k_h))
        rw = tvm.te.reduce_axis((0, k_w))
        return compute([batch_size, out_h, out_w, out_channel],
                       lambda b, h, w, c: tvm.te.sum(
            (padded[b, h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1],
                    c // out_channel_per_group * channel_per_group + rc]
             * weight[c, rc, rh, rw]).astype(out_dtype),
            axis=[rc, rh, rw]),
            name="conv2d_nhwc",
            # tag=tag_gen("conv2d_nhwc" + str(stride) + str(dilation) + str(groups)),
            requires_grad=requires_grad)
    conv_result = GraphOp([batch_size, out_h, out_w, out_channel], [channel_per_group, k_h, k_w],
                          [padded, weight], _inner_conv2d_nhwc, name="conv2d_nhwc")

    if bias is not None:
        def _inner_bias_add(batch_size, out_h, out_w, out_channel, output, bias, requires_grad=True):
            return compute([batch_size, out_h, out_w, out_channel],
                           lambda b, h, w, c: output[b, h, w, c] + bias[c],
                           name="conv2d_nhwc_bias",
                           # tag=tag_gen("conv2d_nhwc_bias"),
                           requires_grad=requires_grad)
        return GraphOp([batch_size, out_h, out_w, out_channel], [], [conv_result, bias],
                       _inner_bias_add, name="conv2d_nhwc_bias")
    return conv_result


def gemv(A, vector, transposeA=False, out_dtype="float32"):
    """Matrix multiplies vector
    Args:
    -----------------------------
    A: GraphNode
        shape [height, width]
    vector: GraphNode
        shape [width]
    transposeA: (optional:False) bool
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [height]
    -----------------------------
    """
    k = tvm.te.reduce_axis((0, vector.shape[0]))
    if transposeA:
        assert(A.shape[0] == vector.shape[0])

        def _inner_gemv1(A_shape1, vector_shape0, A, vector, requires_grad=True):
            return compute([A_shape1],
                           lambda i: tvm.te.sum((A[k, i] * vector[k]).astype(out_dtype), axis=k),
                           name="gemv1",
                           # tag=tag_gen("gemv1"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[1]], [vector.shape[0]], [A, vector], _inner_gemv1, name="gemv1")
    else:
        assert(A.shape[1] == vector.shape[0])

        def _inner_gemv2(A_shape0, vector_shape0, A, vector, requires_grad=True):
            return compute([A_shape0],
                           lambda i: tvm.te.sum((A[i, k] * vector[k]).astype(out_dtype), axis=k),
                           name="gemv2",
                           # tag=tag_gen("gemv2"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[0]], [vector.shape[0]], [A, vector], _inner_gemv2, name="gemv2")


def gemm(A, B, transposeA=False, transposeB=False, out_dtype="float32"):
    """Matrix multiplies matrix
    Args:
    -----------------------------
    A: GraphNode
        shape [height, width]
    B: GraphNode
        shape [width, length]
    transposeA: (optional:False) bool
    transposeB: (optional:False) bool
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [height, length]
    -----------------------------
    """
    if transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert(A.shape[0] == B.shape[1])

        def _inner_gemm1(A_shape1, B_shape0, B_shape1, A, B, requires_grad=True):
            return compute([A_shape1, B_shape0],
                           lambda i, j: tvm.te.sum((A[k, i] * B[j, k]).astype(out_dtype), axis=k),
                           name="gemm1",
                           # tag=tag_gen("gemm1"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[1], B.shape[0]], [B.shape[1]], [A, B], _inner_gemm1, name="gemm1")
    elif transposeA and not transposeB:
        k = tvm.te.reduce_axis((0, B.shape[0]))
        assert(A.shape[0] == B.shape[0])

        def _inner_gemm2(A_shape1, B_shape1, B_shape0, A, B, requires_grad=True):
            return compute([A_shape1, B_shape1],
                           lambda i, j: tvm.te.sum((A[k, i] * B[k, j]).astype(out_dtype), axis=k),
                           name="gemm2",
                           # tag=tag_gen("gemm2"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[1], B.shape[1]], [B.shape[0]], [A, B], _inner_gemm2, name="gemm2")
    elif not transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert(A.shape[1] == B.shape[1])

        def _inner_gemm3(A_shape0, B_shape0, B_shape1, A, B, requires_grad=True):
            return compute([A_shape0, B_shape0],
                           lambda i, j: tvm.te.sum((A[i, k] * B[j, k]).astype(out_dtype), axis=k),
                           name="gemm3",
                           # tag=tag_gen("gemm3"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[0], B.shape[0]], [B.shape[1]], [A, B], _inner_gemm3, name="gemm3")
    else:
        k = tvm.te.reduce_axis((0, B.shape[0]))
        assert(A.shape[1] == B.shape[0])

        def _inner_gemm4(A_shape0, B_shape1, B_shape0, A, B, requires_grad=True):
            return compute([A_shape0, B_shape1],
                           lambda i, j: tvm.te.sum((A[i, k] * B[k, j]).astype(out_dtype), axis=k),
                           name="gemm4",
                           # tag=tag_gen("gemm4"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[0], B.shape[1]], [B.shape[0]], [A, B], _inner_gemm4, name="gemm4")


def batch_gemm(A, B, transposeA=False, transposeB=False, out_dtype="float32"):
    """Batched matrix multiplies matrix
    Args:
    -----------------------------
    A: GraphNode
        shape [batch, height, width]
    B: GraphNode
        shape [batch, width, length]
    transposeA: (optional:False) bool
    transposeB: (optional:False) bool
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, height, length]
    -----------------------------
    """
    assert(A.shape[0] == B.shape[0])

    if transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[2]))
        assert(A.shape[1] == B.shape[2])

        def _inner_batch_gemm1(A_shape0, A_shape2, B_shape1, B_shape2, A, B, requires_grad=True):
            return compute([A_shape0, A_shape2, B_shape1],
                           lambda b, i, j: tvm.te.sum(
                               (A[b, k, i] * B[b, j, k]).astype(out_dtype), axis=k),
                           name="batch_gemm1",
                           # tag=tag_gen("batch_gemm1"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[0], A.shape[2], B.shape[1]], [B.shape[2]], [A, B], _inner_batch_gemm1, name="batch_gemm1")

    elif transposeA and not transposeB:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert(A.shape[1] == B.shape[1])

        def _inner_batch_gemm2(A_shape0, A_shape2, B_shape2, B_shape1, A, B, requires_grad=True):
            return compute([A_shape0, A_shape2, B_shape2],
                           lambda b, i, j: tvm.te.sum(
                               (A[b, k, i] * B[b, k, j]).astype(out_dtype), axis=k),
                           name="batch_gemm2",
                           # tag=tag_gen("batch_gemm2"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[0], A.shape[2], B.shape[2]], [B.shape[1]], [A, B], _inner_batch_gemm2, name="batch_gemm2")

    elif not transposeA and transposeB:
        k = tvm.te.reduce_axis((0, B.shape[2]))
        assert(A.shape[2] == B.shape[2])

        def _inner_batch_gemm3(A_shape0, A_shape1, B_shape1, B_shape2, A, B, requires_grad=True):
            return compute([A_shape0, A_shape1, B_shape1],
                           lambda b, i, j: tvm.te.sum(
                               (A[b, i, k] * B[b, j, k]).astype(out_dtype), axis=k),
                           name="batch_gemm3",
                           # tag=tag_gen("batch_gemm3"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[0], A.shape[1], B.shape[1]], [B.shape[2]], [A, B], _inner_batch_gemm3, name="batch_gemm3")

    else:
        k = tvm.te.reduce_axis((0, B.shape[1]))
        assert(A.shape[2] == B.shape[1])

        def _inner_batch_gemm4(A_shape0, A_shape1, B_shape2, B_shape1, A, B, requires_grad=True):
            return compute([A_shape0, A_shape1, B_shape2],
                           lambda b, i, j: tvm.te.sum(
                               (A[b, i, k] * B[b, k, j]).astype(out_dtype), axis=k),
                           name="batch_gemm4",
                           # tag=tag_gen("batch_gemm4"),
                           requires_grad=requires_grad)
        return GraphOp([A.shape[0], A.shape[1], B.shape[2]], [B.shape[1]], [A, B], _inner_batch_gemm4, name="batch_gemm4")


def linear(inputs, weight, bias=None, out_dtype="float32"):
    """Linear function
    Args:
    -----------------------------
    inputs: GraphNode
        shape [batch, ..., in_feature]
    weight: GraphNode
        shape [out_feature, in_feature]
    bias  : GraphNode
        shape [out_feature]
    -----------------------------
    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, ..., out_feature]
    -----------------------------
    """
    assert(inputs.shape[-1] == weight.shape[1])
    k = tvm.te.reduce_axis((0, inputs.shape[-1]))

    # Expected args:
    # input-dim0, ... input-dim(N-1), weight-out_feature, weight-in_feature, inputs,  weight
    # ..................args[:-4],    args[-4](args[:-3]),   args[-3],      args[-2], args[-1]
    def _inner_linear(*args, requires_grad=True):
        def _inner(*indices):
            return tvm.te.sum((inputs[(*indices[:-1], k)] * weight[indices[-1], k]).astype(out_dtype), axis=k)
        return compute(args[:-3],
                       _inner,
                       name="linear",
                       # tag=tag_gen("linear_dim_" + str(len(args) - 4)),
                       requires_grad=requires_grad)
    linear_result = GraphOp([*inputs.shape[:-1], weight.shape[0]], [inputs.shape[-1]], [inputs, weight],
                            _inner_linear, name="linear")

    if bias is not None:
        assert(bias.shape[0] == weight.shape[0])
        # Expected args:
        # output-dim0, ..., output-dimN, linear_result, bias
        # .............................,    args[-2],  args[-1]
        # Todo: Check whether it is correct

        def _inner_bias_add(*args, requires_grad=True):
            return compute(args[-2].shape,
                           lambda *indice: args[-2][indice] +
                           args[-1](indice[-1]),
                           name="linear_bias",
                           # tag=tag_gen("linear_bias_dim" + str(len(args) - 2)),
                           requires_grad=requires_grad)
        return GraphOp(linear_result.shape, [], [linear_result, bias], _inner_bias_add, name="linear_bias")

    return linear_result


def bilinear(inputs1, inputs2, weight, bias=None, out_dtype="float32"):
    """Bilinear function
    Args:
    -----------------------------
    inputs1: GraphNode
        shape [batch, ..., in_feature1]
    inputs2: GraphNode
        shape [batch, ..., in_feature2]
    weight: GraphNode
        shape [out_feature, in_feature1, in_feature2]
    bias  : GraphNode
        shape [out_feature]
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [batch, ..., out_feature]
    -----------------------------
    """
    assert(inputs1.shape[-1] == weight.shape[-2])
    assert(inputs2.shape[-1] == weight.shape[-1])
    k1 = tvm.te.reduce_axis((0, inputs1.shape[-1]))
    k2 = tvm.te.reduce_axis((0, inputs2.shape[-1]))
    for i in range(len(inputs1.shape) - 1):
        assert(inputs1.shape[i] == inputs2.shape[i])

    def _inner_bilinear(*args, requires_grad=True):
        def _inner(*indices):
            return tvm.te.sum(
                (inputs1[(*indices[:-1], k1)] * weight[indices[-1],
                                                      k1, k2] * inputs2[(*indices[:-1], k2)]).astype(out_dtype),
                axis=[k1, k2]
            )
        return compute([*inputs1.shape[:-1], weight.shape[0]],
                       _inner,
                       name="bilinear",
                       # tag=tag_gen("bilinear_dim" + str(len(inputs1.shape)) + "_dim" + str(len(inputs2.shape))),
                       requires_grad=requires_grad)

    bilinear_result = GraphOp([*inputs1.shape[:-1], weight.shape[0]], [inputs1.shape[-1], inputs2.shape[-1]],
                              [inputs1, inputs2, weight], _inner_bilinear, name="bilinear")
    if bias is not None:
        assert(bias.shape[0] == weight.shape[0])
        # Expected args:
        # dim0, ..., dimN, biliear_result, bias
        # ....            , args[-2],     args[-1]
        # Todo: Check whether it is correct

        def _inner_bias_add(*args, requires_grad=True):
            def _add(*indices):
                return args[-2][indices] + args[-1][indices[-1]]
            return compute(bilinear_result.shape,
                           _add,
                           name="bilinear_bias",
                           # tag=tag_gen("bilinear_bias_dim" + str(len(bilinear_result.shape)))
                           )
        return GraphOp(bilinear_result.shape, [], [bilinear_result, bias],
                       _inner_bias_add, name="bilinear_bias")
    return bilinear_result


def MTTKRP3d(A, B, C, out_dtype="float32"):
    """Dense MTTKRP 3D
    Args:
    -----------------------------
    A: GraphNode
        shape [out_height, h1, h2]
    B: GraphNode
        shape [h1, out_width]
    C: GraphNode
        shape [h2, out_width]
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [out_height, out_width]
    -----------------------------
    """
    assert(A.shape[1] == B.shape[0])
    assert(A.shape[2] == C.shape[0])
    assert(B.shape[1] == C.shape[1])
    k = tvm.te.reduce_axis((0, B.shape[0]))
    l = tvm.te.reduce_axis((0, C.shape[0]))

    def _inner_MTTKRP3d(A_shape0, B_shape1, B_shape0, C_shape0, A, B, C, requires_grad=True):
        return compute([A_shape0, B_shape1],
                       lambda i, j: tvm.te.sum(
                           (A[i, k, l] * B[k, j] * C[l, j]).astype(out_dtype), axis=[k, l]),
                       name="MTTKRP3d",
                       # tag=tag_gen("MTTKRP3d"),
                       requires_grad=requires_grad)
    return GraphOp([A.shape[0], B.shape[1]], [B.shape[0], C.shape[0]], [A, B, C],
                   _inner_MTTKRP3d, name="MTTKRP3d")


def pointwise_multiply(A, B):
    """Pointwise multiply
    Args:
    -----------------------------
    A: GraphNode
        shape [...]
    B: GraphNode
        shape same as A
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape same as A
    -----------------------------
    """
    assert(len(A.shape) == len(B.shape))
    for i in range(len(A.shape)):
        assert(A.shape[i] == B.shape[i])

    def _inner_pointwise_multiply(A, B, requires_grad=True):
        def _mul(*indices):
            return A[indices] * B[indices]
        return compute(A.shape,
                       _mul,
                       name="pointwise_multiply",
                       # tag=tag_gen("pointwise_multiply_dim" + str(len(A.shape))),
                       requires_grad=requires_grad)

    return GraphOp(A.shape, [], [A, B], _inner_pointwise_multiply, name="pointwise_multiply")


def mean(inputs, dim=0):
    """Mean
    Args:
    -----------------------------
    A: GraphNode
        shape [...]
    dim: (optional:0) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [...]
    -----------------------------
    """
    assert(dim >= 0 and dim < len(inputs.shape))
    output_shape = []
    for i in range(len(inputs.shape)):
        if i == dim:
            continue
        output_shape.append(inputs.shape[i])

    k = tvm.te.reduce_axis((0, inputs.shape[dim]))

    # Expected args:
    # dim0, dim1, ..., dimN, reduction_dim, inputs
    def _inner_mean(*args, requires_grad=True):
        def _inner(*indices):
            return tvm.te.sum(inputs[(*indices[:dim], k, *indices[dim:])] / inputs.shape[dim], axis=k)
        return compute(output_shape,
                       _inner,
                       name="mean",
                       # tag=tag_gen("mean_dim" + str(inputs.shape)),
                       requires_grad=requires_grad)

    return GraphOp(output_shape, [inputs.shape[dim]], [inputs], _inner_mean, name="mean")


def variance(inputs, mean_val=None, dim=0):
    """Variance
    Args:
    -----------------------------
    A: GraphNode
        shape [...]
    dim: (optional:0) int
    -----------------------------
    Returns:
    -----------------------------
    GraphNode
        shape [...]
    -----------------------------
    """
    assert(dim >= 0 and dim < len(inputs.shape))
    assert(inputs.shape[dim] > 1)
    output_shape = []
    for i in range(len(inputs.shape)):
        if i == dim:
            continue
        output_shape.append(inputs.shape[i])

    k = tvm.te.reduce_axis((0, inputs.shape[dim]))
    mean_val = mean_val if mean_val is not None else mean(inputs, dim)

    def _inner_variance(*args, requires_grad=True):
        def _inner(*indices):
            return tvm.te.sum((inputs[(*indices[:dim], k, *indices[dim:])] - mean_val[indices]) *
                              (inputs[(*indices[:dim], k, *indices[dim:])] - mean_val[indices]) / (inputs.shape[dim] - 1), axis=k)
        return compute(output_shape,
                       _inner,
                       name="variance",
                       # tag=tag_gen("variance_dim" + str(len(inputs.shape))),
                       requires_grad=requires_grad)
    return GraphOp(output_shape, [inputs.shape[dim]], [inputs, mean_val], _inner_variance, name="variance")


def LSTMCell(inputs, hs, cs, weights, bias=None):
    assert inputs.shape[0] == hs.shape[0]
    assert hs.shape[0] == cs.shape[0]
    assert weights.shape[0] == 4
    assert weights.shape[2] == inputs.shape[1] + hs.shape[1]
    k1 = tvm.te.reduce_axis((0, inputs.shape[1]))
    k2 = tvm.te.reduce_axis((0, hs.shape[1]))

    def _inner_LSTMCell_A(inputs0, weights0, weight1, inputs1, inputs, weights, requires_grad=True):
        return compute([inputs0, weights0, weight1],
                       lambda b, i, j: tvm.te.sum(
                           inputs[b, k1] * weights[i, j, k1], axis=k1),
                       name="LSTMCellA",
                       # tag=tag_gen("LSTMCellA"),
                       requires_grad=requires_grad)
    A = GraphOp([inputs.shape[0], weights.shape[0], weights.shape[1]], [inputs.shape[1]],
                [inputs, weights], _inner_LSTMCell_A, name="LSTMCellA")

    def _inner_LSTMCell_B(hs0, weights0, weights1, hs1, hs, weights, requires_grad=True):
        return compute([hs0, weights0, weights1],
                       lambda b, i, j: tvm.te.sum(
                           hs[b, k2] * weights[i, j, k2 + inputs.shape[1]], axis=k2),
                       name="LSTMCellB",
                       # tag=tag_gen("LSTMCellB"),
                       requires_grad=requires_grad)
    B = GraphOp([hs.shape[0], weights.shape[0], weights.shape[1]], [hs.shape[1]], [hs, weights],
                _inner_LSTMCell_B, name="LSTMCellB")

    if bias is not None:
        def _inner_bias(inputs0, weights0, weights1, A, B, bias, requires_grad=True):
            return compute([inputs0, weights0, weights1],
                           lambda b, i, j: A[b, i, j] +
                           B[b, i, j] + bias[b, i, j],
                           name="LSTM_bias_add",
                           # tag=tag_gen("LSTM_bias_add"),
                           requires_grad=requires_grad)
        C = GraphOp([inputs.shape[0], weights.shape[0], weights.shape[1]], [], [A, B, bias], _inner_bias,
                    name="LSTM_bias_add")
    else:
        def _inner_no_bias(inputs0, weights0, weights1, A, B, requires_grad=True):
            return compute([inputs0, weights0, weights1],
                           lambda b, i, j: A[b, i, j] + B[b, i, j],
                           name="LSTM_no_bias_add",
                           # tag=tag_gen("LSTM_no_bias_add"),
                           requires_grad=requires_grad)
        C = GraphOp([inputs.shape[0], weights.shape[0], weights.shape[1]], [], [A, B],
                    _inner_no_bias, name="LSTM_no_bias_add")

    def _inner_next_cs(cs_0, weight_1, C, cs, requires_grad=True):
        return compute([cs_0, weight_1],
                       lambda b, i: tvm.te.sigmoid(
                           C[b, 1, i]) * cs[b, i] + tvm.te.sigmoid(C[b, 0, i]) * tvm.te.tanh(C[b, 3, i]),
                       name="next_cs",
                       # tag=tag_gen("next_cs"),
                       requires_grad=requires_grad)
    next_cs = GraphOp([cs.shape[0], weights.shape[1]], [], [
                      C, cs], _inner_next_cs, name="next_cs")

    def _inner_next_hs(hs_0, weight_1, C, next_cs, requires_grad=True):
        return compute([hs_0, weight_1],
                       lambda b, i: tvm.te.sigmoid(
                           C[b, 2, i]) * tvm.te.tanh(next_cs[b, i]),
                       name="next_hs",
                       # tag=tag_gen("next_hs"),
                       requires_grad=requires_grad)
    next_hs = GraphOp([hs.shape[0], weights.shape[1]], [], [
                      C, next_cs], _inner_next_hs, name="next_hs")

    return next_hs, next_cs


def block_circulant_matrix(Input, factor):
    ROW, COL = Input.shape
    FFT = factor

    k = tvm.te.reduce_axis((0, FFT))

    def _inner_compress(shape0, shape1, FFT, Input, requires_grad=True):
        return compute([shape0, shape1],
                       lambda i, j: (
            tvm.te.sum(
                Input[i * FFT + k, (j // FFT) *
                      FFT + (j % FFT + k) % FFT] / FFT,
                axis=k)
        ),
            name="block_circulant_matrix_compress",
            # tag=tag_gen("block_circulant_matrix_compress" + str(factor)),
            requires_grad=requires_grad)
    Compress = GraphOp([ROW // FFT, (COL // FFT) * FFT], [FFT], [Input],
                       _inner_compress, name="block_circulant_matrix_compress")

    def _inner_output(shape0, shape1, Compress, requires_grad=True):
        return compute([shape0, shape1],
                       lambda i, j: (
            tvm.te.if_then_else(
                tvm.te.all(i < (ROW // FFT) * FFT,
                           j < (COL // FFT) * FFT),
                Compress[i // FFT, (j // FFT) * FFT +
                         ((j % FFT) + FFT - (i % FFT)) % FFT],
                tvm.tir.const(0, Input.dtype)
            )
        ),
            name="block_circulant_matrix_output",
            # tag=tag_gen("block_circulant_matrix_output" + str(factor)),
            requires_grad=requires_grad)

    Output = GraphOp([ROW, COL], [], [Compress], _inner_output,
                     name="block_circulant_matrix_output")

    return Output


def MaxUnpooling1d(Input, Indices, kernel_size, stride, padding):
    """
    Max Unpooling 1d Operator
    Parameters
    ----------
    Input: GraphNode
        3-D with shape [batch_size, channels, in_lengths]
    Indices: GraphNode
        3-D with shape [batch_size, channels, out_lengths]
    kernel_size: int
    stride: int
    Returns
    -------
    Output: GraphNode
        3-D with shape [batch_size, channels, out_lengths]
    """

    batch_size, channels, in_lengths = Input.shape
    batch_size, channels, in_lengths = Indices.shape

    out_lengths = (in_lengths - 1) * stride - 2 * padding + kernel_size

    iterK = tvm.te.reduce_axis((0, in_lengths), name='k')

    def _inner_MaxUnpooling1d(batch_size, channels, out_lengths, in_lengths, Indices, Input, requires_grad=True):
        return compute([batch_size, channels, out_lengths],
                       lambda b, c, l:
                       tvm.te.max(
            tvm.te.if_then_else(l == Indices[b, c, iterK],
                                Input[b, c, iterK],
                                tvm.tir.expr.const(0, Input.tvm_tensor.dtype)),
            axis=iterK),
            name="MaxUnpooling1d",
            # tag=tag_gen("MaxUnpooling1d"),
            requires_grad=requires_grad)
    Output = GraphOp([batch_size, channels, out_lengths], [in_lengths], [Indices, Input],
                     _inner_MaxUnpooling1d, name="MaxUnpooling1d")

    return Output


def MaxUnpooling2d(Input, Indices, kernel_size, stride, padding, output_size=None):
    """
    Max Unpooling 2d Operator
    Parameters
    ----------
    Input: GraphNode
        4-D with shape [batch_size, channels, in_height, in_width]
    Indices: GraphNode
        4-D with shape [batch_size, channels, in_height, in_width]
    kernel_size: int or tuple
    stride: int or tuple
    Returns
    -------
    Output: GraphNode
        4-D with shape [batch_size, channels, out_height, out_width]
    """

    batch_size, channels, in_height, in_width = Input.shape
    batch_size, channels, in_height, in_width = Indices.shape

    if type(kernel_size) == int:
        kernel_size = (kernel_size, kernel_size)
    if type(stride) == int:
        stride = (stride, stride)
    if type(padding) == int:
        padding = (padding, padding)

    out_height = (in_height - 1) * stride[0] - 2 * padding[0] + kernel_size[0]
    out_width = (in_width - 1) * stride[1] - 2 * padding[1] + kernel_size[1]

    iterH = tvm.te.reduce_axis((0, in_height), name='h')
    iterW = tvm.te.reduce_axis((0, in_width), name='ws')

    def _inner_MaxUnpooling2d(batch_size, channels, out_height, out_width,
                              in_height, in_width, Indices, Input, requires_grad=True):
        return compute([batch_size, channels, out_height, out_width],
                       lambda b, c, h, w:
                       tvm.te.max(
            tvm.te.if_then_else(h * out_width + w == Indices[b, c, iterH, iterW],
                                Input[b, c, iterH, iterW],
                                tvm.tir.expr.const(0, Input.tvm_tensor.dtype)),
            axis=[iterH, iterW]),
            name="MaxUnpooling2d",
            # tag=tag_gen("MaxUnpooling2d"),
            requires_grad=requires_grad)
    Output = GraphOp([batch_size, channels, out_height, out_width], [in_height, in_width],
                     [Indices, Input], _inner_MaxUnpooling2d, name="MaxUnpooling2d")

    return Output


def PixelCNN(Input, Kernel, mask_type, bias=None, dilation=1, stride=1, padding=0):
    """
    Pixel CNN Operator
    Parameters
    ----------
    Input: GraphNode
        4-D with shape [batch_size, input_height, input_width, in_channels]
    Kernel: GraphNode
        4-D with shape [out_channels, in_channels, kernel_height, kernel_width]
    mask_type: str 'A' or 'B'
    dilation: int or tuple
    stride: int or tuple
    padding: int or tuple
    Returns
    -------
    Output: GraphNode
        4-D with shape [batch_size, out_height, out_width, channels]
    """

    batch, inputHeight, inputWidth, in_channels = Input.shape
    batch, out_channels, kernelHeight, kernelWidth = Kernel.shape

    assert mask_type in ['A', 'B']

    if mask_type == 'A':
        def _inner_PixelCNN_A(batch, out_c, kernel_h, kernel_w, Kernel, requires_grad=True):
            return compute([batch, out_c, kernel_h, kernel_w],
                           lambda b, o, h, w: tvm.te.if_then_else(
                               tvm.tir.Or(
                                   tvm.tir.And(
                                        h == kernelHeight // 2,
                                        w >= kernelWidth // 2),
                                        h > kernelHeight // 2),
                                        tvm.tir.expr.const(0, Input.tvm_tensor.dtype),
                                        Kernel[b, o, h, w]),
                           name="PixelCNN_A",
                           # tag=tag_gen("PixelCNN_A"),
                           requires_grad=requires_grad)
        Mask = GraphOp(Kernel.shape, [], [Kernel],
                       _inner_PixelCNN_A, name="PixelCNN_A")
    else:
        def _inner_PixelCNN_B(batch, out_c, kernel_h, kernel_w, Kernel, requires_grad=True):
            return compute([batch, out_c, kernel_h, kernel_w],
                           lambda b, o, h, w: tvm.te.if_then_else(
                               tvm.tir.Or(
                                   tvm.tir.And(
                                      h == kernelHeight // 2,
                                      w > kernelWidth // 2),
                                    h > kernelHeight // 2),
                                tvm.tir.expr.const(0, Input.tvm_tensor.dtype),
                                Kernel[b, o, h, w]),
                           name="PixelCNN_B",
                           # tag=tag_gen("PixelCNN_B"),
                           requires_grad=requires_grad)
        Mask = GraphOp(Kernel.shape, [], [Kernel],
                       _inner_PixelCNN_B, name="PixelCNN_B")

    Output = conv2d_nhwc(Input, Mask, bias, stride=stride,
                         padding=padding, dilation=dilation)

    return Mask, Output


def GatedPixelCNN(Input, KernelV, KernelV2H, KernelH, KernelHOut, ClassVector=None, bias=None, dilation=1, stride=1, padding=0):
    """
    Gated Pixel CNN Operator
    Parameters
    ----------
    Input: GraphNode
        4-D with shape [batch_size, input_height, input_width, in_channels]
    KernelV: GraphNode
        Vertical Kernel
        4-D with shape [2 * out_channels, in_channels, kernel_size, kernel_size]
    KernelV2H: GraphNode
        Combine output from vertical to horizontal
        4-D with shape [2 * out_channels, 2 * out_channels, 1, 1]
    KernelH: GraphNode
        Horizontal Kernel
        4-D with shape [2 * out_channels, in_channels, 1, kernel_size]
    KernelHOut: GraphNode
        Horizontal Output Kernel
        4-D with shape [out_channels, out_channels, 1, 1]
    ClassVector: GraphNode
        4-D with shape [batch_size, 2 * out_channels, 1, 1]
    dilation: int
    stride: int
    padding: int
    Returns
    -------
    GateV: GraphNode
        4-D with shape [batch_szie, out_height, out_width, out_channels]
    Output: GraphNode
        4-D with shape [batch_size, out_height, out_width, out_channels]
    """
    batch, inputHeight, inputWidth, in_channels = Input.shape
    out_channels, in_channels, kernelHeight, kernelWidth = KernelV.shape
    out_channels /= 2

    assert kernelHeight == kernelWidth

    ConvV = PixelCNN(Input, KernelV, mask_type='B', bias=bias, dilation=(
        dilation, dilation), stride=(stride, stride), padding=(padding, padding))[-1]
    Vertical2HorizonTal = conv2d_nhwc(
        ConvV, KernelV2H, bias=bias, stride=(1, 1), padding=(0, 0), dilation=(1, 1))
    ConvH = PixelCNN(Input, KernelH, mask_type='B', bias=bias, dilation=(
        dilation, dilation), stride=(stride, stride), padding=(0, padding))[-1]

    def _inner_CombineFeature(ConvH_0, ConvH_1, ConvH_2, ConvH_3, ConvH, Vertical2HorizonTal, requires_grad=True):
        return compute([ConvH_0, ConvH_1, ConvH_2, ConvH_3],
                       lambda b, h, w, c: ConvH[b, h, w, c] +
                       Vertical2HorizonTal[b, h, w, c],
                       name="GatedPixelCNN_CombineFeature",
                       # tag=tag_gen("GatedPixelCNN_CombineFeature"),
                       requires_grad=requires_grad)
    CombineFeature = GraphOp(ConvH.shape, [], [ConvH, Vertical2HorizonTal],
                             _inner_CombineFeature, name="GatedPixelCNN_CombineFeature")

    if ClassVector == None:
        def _inner_ActivationV_1(ConvV_0, ConvV_1, ConvV_2, ConvV_3, ConvV, requires_grad=True):
            return compute([ConvV_0, ConvV_1, ConvV_2, ConvV_3],
                           lambda b, h, w, o: tvm.te.if_then_else(o < out_channels,
                                                                  tvm.te.tanh(
                                                                      ConvV[b, h, w, o]),
                                                                  tvm.te.sigmoid(ConvV[b, h, w, o])),
                           name="GatedPixelCNN_ActivationV_1",
                           # tag=tag_gen("GatedPixelCNN_ActivationV_1"),
                           requires_grad=requires_grad)
        ActivationV = GraphOp(ConvV.shape, [], [
                              ConvV], _inner_ActivationV_1, name="GatedPixelCNN_ActivationV_1")
    else:
        def _inner_ActivationV_2(ConvV_0, ConvV_1, ConvV_2, ConvV_3, ConvV, ClassVector, requires_grad=True):
            return compute([ConvV_0, ConvV_1, ConvV_2, ConvV_3],
                           lambda b, h, w, o: tvm.te.if_then_else(o < out_channels,
                                                                  tvm.te.tanh(
                                                                      ConvV[b, h, w, o] + ClassVector[b, 0, 0, o]),
                                                                  tvm.te.sigmoid(ConvV[b, h, w, o] + ClassVector[b, 0, 0, o])),
                           name="GatedPixelCNN_ActivationV_2",
                           # tag=tag_gen("GatedPixelCNN_ActivationV_2"),
                           requires_grad=requires_grad)
        ActivationV = GraphOp(ConvV.shape, [], [
                              ConvV, ClassVector], _inner_ActivationV_2, name="GatedPixelCNN_ActivationV_2")

    def _inner_GatedPixelCNN_GateV(batch, Act_1, Act_2, out_channels, ActivationV, requires_grad=True):
        return compute([batch, Act_1, Act_2, out_channels],
                       lambda b, h, w, c: ActivationV[b, h, w, c] *
                       ActivationV[b, h, w, c + out_channels],
                       name="GatedPixelCNN_GateV",
                       # tag=tag_gen("GatedPixelCNN_GateV"),
                       requires_grad=requires_grad)
    GateV = GraphOp([batch, ActivationV.shape[1], ActivationV.shape[2], out_channels], [],
                    [ActivationV], _inner_GatedPixelCNN_GateV, name="GatedPixelCNN_GateV")

    def _inner_ActivationH(combine_0, combine_1, combine_2, combine_3, CombineFeature, requires_grad=True):
        return compute([combine_0, combine_1, combine_2, combine_3],
                       lambda b, h, w, o: tvm.te.if_then_else(o < out_channels,
                                                              tvm.te.tanh(
                                                                  CombineFeature[b, h, w, o]),
                                                              tvm.te.sigmoid(CombineFeature[b, h, w, o])),
                       name="ActivationH",
                       # tag=tag_gen("ActivationH"),
                       requires_grad=requires_grad)
    ActivationH = GraphOp(CombineFeature.shape, [], [
                          CombineFeature], _inner_ActivationH, name="ActivationH")

    def _inner_GateH(batch, act_1, act_2, out_channels, ActivationH, requires_grad=True):
        return compute([batch, act_1, act_2, out_channels],
                       lambda b, h, w, c: ActivationH[b, h, w, c] *
                       ActivationH[b, h, w, c + out_channels],
                       name="GateH",
                       # tag=tag_gen("GateH"),
                       requires_grad=requires_grad)
    GateH = GraphOp([batch, ActivationH.shape[1], ActivationH.shape[2], out_channels], [],
                    [ActivationH], _inner_GateH, name="GateH")

    ConvGateH = conv2d_nhwc(GateH, KernelHOut, bias=bias, dilation=(
        dilation, dilation), stride=(stride, stride), padding=(padding, padding))

    def _inner_Output(shape0, shape1, shape2, shape3, ConvGateH, Input, requires_grad=True):
        return compute([shape0, shape1, shape2, shape3],
                       lambda b, h, w, o: ConvGateH[b,
                                                    h, w, o] + Input[b, h, w, o],
                       name="GatedPixelCNN_output",
                       # tag=tag_gen("GatedPixelCNN_output"),
                       requires_grad=requires_grad)
    Output = GraphOp(ConvGateH.shape, [], [ConvGateH, Input],
                     _inner_Output, name="GatedPixelCNN_output")

    return GateV, Output


def ReLU(x):
    """Take relu of input x.

    Parameters
    ----------
    x : GraphNode
        Arbitrary dimension Input argument.

    Returns
    -------
    y : GraphOp
        The result.
    """
    def _inner_ReLU(*args, requires_grad=True):
        assert len(args) > 1
        # Here we assume that the input is a NamedDimTensor with tvm_tensor as its attribute!
        return compute(
            args[:-1], lambda *i: tvm.te.max(args[-1](*i),
                                             tvm.tir.const(0, args[-1].tvm_tensor.dtype)),
            name="relu",
            # tag=tag_gen("relu_dim" + str(len(args) - 1)),
            requires_grad=requires_grad)
    return GraphOp(x.shape, [], [x], _inner_ReLU, name="relu")


def GELU(x):
    def _inner_GELU(*args, requires_grad=True):
        assert len(args) > 1
        # Here we assume that the input is a NamedDimTensor with tvm_tensor as its attribute!
        def _c(i): 
            return tvm.tir.const(i, args[-1].tvm_tensor.dtype)
            
        def _gelu(*i):
            x = args[-1](*i)
            y = x * _c(0.7978845608) * (_c(1.0) + _c(0.044715) * x * x)
            y = _c(0.5) * x * (_c(1.0) + tvm.te.tanh(y))
            return y

        return compute(
            args[:-1], _gelu,
            name="gelu",
            requires_grad=requires_grad)
    return GraphOp(x.shape, [], [x], _inner_GELU, name="gelu")


def batch_flatten(inputs):
    '''
    inputs: [batch, channel, height, width]
    return: [batch, channel * height * width]
    '''
    assert len(inputs.shape) == 4
    batch, channel, height, width = inputs.shape

    def _inner_batch_flatten(batch, chw, inputs, requires_grad=True):
        return compute([batch, chw],
                       lambda i, j: inputs[i, j//(height * width), (j % (height * width)) // width,
                                           j % width],
                       name="batch_flatten",
                       # tag=tag_gen("batch_flatten"),
                       requires_grad=requires_grad)
    return GraphOp([batch, channel * height * width], [], [inputs], _inner_batch_flatten, name="batch_flatten")


def dense(inputs, weight, bias=None, out_dtype="float32"):
    """Linear function, only for 2-dim inputs and weight

    Args:
    -----------------------------
    inputs: GraphNode
        shape [batch, in_feature]
    weight: GraphNode
        shape [out_feature, in_feature]
    bias  : GraphNode
        shape [out_feature]
    -----------------------------

    Returns:
    -----------------------------
    GraphOp
        shape [batch, out_feature]
    -----------------------------
    """
    assert len(inputs.shape) == 2 and len(weight.shape) == 2
    batch, in_feature = inputs.shape
    out_feature, in_feature2 = weight.shape
    assert in_feature == in_feature2, "%d vs. %d" % (in_feature, in_feature2)

    def _inner_dense(batch, out_feature, in_feature, inputs, weight, requires_grad=True):
        k = tvm.te.reduce_axis((0, in_feature))
        return compute(
            [batch, out_feature],
            lambda i, j: tvm.te.sum(
                (inputs[i, k] * weight[j, k]).astype(out_dtype), axis=[k]),
            name="dense",
            # tag=tag_gen("dense"),
            requires_grad=requires_grad)

    withoutBias = GraphOp([batch, out_feature], [in_feature], [
                          inputs, weight], _inner_dense, name="linear")

    if bias is not None:
        assert(bias.shape[0] == weight.shape[0])

        def _inner_bias_add(batch, out_feature, withoutBias, bias, requires_grad=True):
            return compute(
                [batch, out_feature],
                lambda i, j: withoutBias[i, j] + bias[j],
                name="dense_bias",
                # tag=tag_gen("dense_bias"),
                requires_grad=requires_grad)
        return GraphOp([batch, out_feature], [], [withoutBias, bias], _inner_bias_add, name="bias_add")

    return withoutBias


def avgpool2d(inputs, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, padding=0):
    """Average Pooling for GraphOp

    Args:
    --------------------------
    inputs: GraphNode
        shape: [batch, channel, height, width]
    Stride, padding are also supported

    out_height = (height + 2*padding - kernel_h) // stride_h + 1

    Returns:
    --------------------------
    GraphOp
        shape: [batch, channel, out_height, out_width]
    """
    padding = (padding, padding) if isinstance(
        padding, (int, tvm.tir.IntImm)) else padding
    padded_inputs = zero_pad2d(inputs, padding=padding)

    batch, channel, h_in, w_in = padded_inputs.shape

    h_out = (h_in - kernel_h) // stride_h + 1
    w_out = (w_in - kernel_w) // stride_w + 1

    config = [kernel_h, kernel_w, stride_h, stride_w]

    def _inner_avgpool2d(batch, channel, h_out, w_out, kernel_h, kernel_w, padded_inputs, requires_grad=True):
        r = tvm.te.reduce_axis([0, kernel_h], name="r")
        s = tvm.te.reduce_axis([0, kernel_w], name="s")
        return compute(
            [batch, channel, h_out, w_out],
            lambda n, c, i, j:
                tvm.te.sum(padded_inputs[n, c, i * stride_h + r, j *
                                         stride_w + s]/(kernel_h*kernel_w), axis=[r, s]),
            name="avgpool2d",
            # tag=tag_gen("avgpool2d" + str(config)),
            requires_grad=requires_grad)
    return GraphOp([batch, channel, h_out, w_out], [kernel_h, kernel_w], [padded_inputs], _inner_avgpool2d, name="avgpool2d")


def cross_entropy_loss(inputs, targets):
    """Cross entropy loss for GraphOp

    Args:
    --------------------------
    inputs: GraphNode
        shape: [batch, num_class]
    targets: GraphNode
        shape: [batch, num_class]

    Returns:
    --------------------------
    GraphOp
        shape: [1]
    """
    assert isinstance(inputs, GraphNode) and isinstance(targets, GraphNode)
    N, C = inputs.shape
    assert N == targets.shape[0] and C == targets.shape[1]
    # First compute the maximum for each batch

    def _inner_max_val(N, C, inputs, requires_grad=True):
        k1 = tvm.te.reduce_axis([0, C], name="k1")
        return compute(
            [N],
            lambda n: tvm.te.max(inputs[n, k1], axis=[k1]),
            name="ce_max",
            # tag=tag_gen("ce_max"),
            requires_grad=requires_grad)
    max_val = GraphOp([N], [C], [inputs], _inner_max_val, name="max_val")

    # Use the log_softmax trick to avoid overflow
    def _inner_sum(N, C, inputs, max_val, requires_grad=True):
        c = tvm.te.reduce_axis([0, C], "c")
        return compute(
            [N],
            lambda i: tvm.te.sum(tvm.tir.exp(
                inputs[i, c] - max_val[i]), axis=[c]),
            name="ce_sum",
            # tag=tag_gen("ce_sum"),
            requires_grad=requires_grad)
    sum_val = GraphOp([N], [C], [inputs, max_val], _inner_sum, name="sum_val")

    def _inner_CEloss(one, N, C, sum_val, max_val, inputs, targets, requires_grad=True):
        rrn = tvm.te.reduce_axis([0, N], "rrn")
        rrc = tvm.te.reduce_axis([0, C], "rrc")
        return compute([1], lambda i: tvm.te.sum(
            targets[i + rrn, rrc] * ((tvm.tir.log(sum_val[i + rrn]) + max_val[i + rrn]) - inputs[i + rrn, rrc] * targets[
                i + rrn, rrc]) / N,
            axis=[rrn, rrc]),
            name="cross_entropy",
            # tag=tag_gen("cross_entropy"),
            requires_grad=True)
    return GraphOp([1], [N, C], [sum_val, max_val, inputs, targets], _inner_CEloss, name="cross_entropy")


def add_4d(inputs1, inputs2):
    """4-dimensional addition for GraphOp

    Args:
    --------------------------
    inputs1 & input2: GraphNode
        shape: [batch, channel, height, width]

    Returns:
    --------------------------
    GraphOp
        shape: [batch, channel, height, width]
    """
    N, C, H, W = inputs1.shape
    N2, C2, H2, W2 = inputs2.shape
    assert N == N2 and C == C2 and H == H2 and W == W2

    def _inner_add_4d(N, C, H, W, inputs1, inputs2, requires_grad=True):
        return compute(
            [N, C, H, W],
            lambda n, c, h, w: inputs1[n, c, h, w] + inputs2[n, c, h, w],
            name="add_4d",
            # tag=tag_gen("add_4d"),
            requires_grad=requires_grad)
    return GraphOp([N, C, H, W], [], [inputs1, inputs2], _inner_add_4d, name="add_4d")


def global_avg_pool2d(inputs, keep_dim=True):
    """Global Average Pooling for GraphOp

    Args:
    --------------------------
    inputs: GraphNode
        shape: [batch, channel, height, width]

        keep_dim: bool

    Returns:
    --------------------------
    GraphOp
        shape: [batch, channel, 1, 1] if keep dim is True
        else [batch, channel]
    """
    N, C, H, W = inputs.shape

    def _innner_global_avg_pool2d_keep(N, C, one1, one2, H, W, inputs, requires_grad=True):
        h = tvm.te.reduce_axis([0, H], name="h")
        w = tvm.te.reduce_axis([0, W], name="w")
        return compute([N, C, one1, one1],
                       lambda n, c, i, j:
                       tvm.te.sum(inputs[n, c, i + h, j + w] /
                                  (H*W), axis=[h, w]),
                       name="global_avg_pool2d_keep",
                       # tag=tag_gen("global_avg_pool2d_keep"),
                       requires_grad=requires_grad)

    def _innner_global_avg_pool2d_nokeep(N, C, H, W, inputs, requires_grad=True):
        h = tvm.te.reduce_axis([0, H], name="h")
        w = tvm.te.reduce_axis([0, W], name="w")
        return compute([N, C],
                       lambda n, c:
                       tvm.te.sum(inputs[n, c, h, w]/(H*W), axis=[h, w]),
                       name="global_avg_pool2d_nokeep",
                       # tag=tag_gen("global_avg_pool2d_nokeep"),
                       requires_grad=requires_grad)
    if keep_dim:
        return GraphOp([N, C, 1, 1], [H, W], [inputs], _innner_global_avg_pool2d_keep, name="global_avg_pool2d")
    else:
        return GraphOp([N, C], [H, W], [inputs], _innner_global_avg_pool2d_nokeep, name="global_avg_pool2d")


def global_avg_pool3d(inputs, keep_dim=True):
    """Global Average Pooling for GraphOp

    Args:
    --------------------------
    inputs: GraphNode
        shape: [batch, channel, depth, height, width]

        keep_dim: bool

    Returns:
    --------------------------
    GraphOp
        shape: [batch, channel, 1, 1, 1] if keep dim is True
        else [batch, channel]
    """
    N, C, D, H, W = inputs.shape

    def _innner_global_avg_pool3d_keep(N, C, one1, one2, one3, D, H, W, inputs, requires_grad=True):
        d = tvm.te.reduce_axis([0, D], name="d")
        h = tvm.te.reduce_axis([0, H], name="h")
        w = tvm.te.reduce_axis([0, W], name="w")
        return compute([N, C, one1, one2, one3],
                       lambda n, c, l, i, j:
                       tvm.te.sum(inputs[n, c, l + d, i + h,
                                         j + w]/(H*D*W), axis=[d, h, w]),
                       name="global_avg_pool3d_keep",
                       requires_grad=requires_grad)

    def _innner_global_avg_pool3d_nokeep(N, C, D, H, W, inputs, requires_grad=True):
        d = tvm.te.reduce_axis([0, D], name="d")
        h = tvm.te.reduce_axis([0, H], name="h")
        w = tvm.te.reduce_axis([0, W], name="w")
        return compute([N, C],
                       lambda n, c:
                       tvm.te.sum(inputs[n, c, d, h, w] /
                                  (H*D*W), axis=[d, h, w]),
                       name="global_avg_pool3d_nokeep",
                       requires_grad=requires_grad)
    if keep_dim:
        return GraphOp([N, C, 1, 1, 1], [D, H, W], [inputs], _innner_global_avg_pool3d_keep, name="global_avg_pool3d")
    else:
        return GraphOp([N, C], [D, H, W], [inputs], _innner_global_avg_pool3d_nokeep, name="global_avg_pool3d")


def equal_const_int(expr, value):
    """Returns if expr equals value.

    Parameters
    ----------
    expr : tvm.Expr
        The input expression.

    Returns
    -------
    equal : bool
        Whether they equals.
    """
    if isinstance(expr, Integral):
        return expr == value
    if not isinstance(expr, tvm.tir.IntImm):
        expr = tvm.tir.ir_pass.Simplify(expr)
    if not isinstance(expr, tvm.tir.IntImm):
        return False
    return expr.value == value


def pad(data, pad_before, pad_after=None, pad_value=0.0, name="PadInput"):
    """Pad Input with zeros.

    Parameters
    ----------
    data : GraphNode
        n-D input, can be any layout.

    pad_before : list / tuple of n ints
        Pad width on each dimension to pad the before the axis begin.

    pad_after : list / tuple of n ints, optional
        Pad width each dimension to pad the after the axis end.

    pad_value : float, optional
        The value to be padded.

    name : str, optional
        The name prefix operators generated

    Returns
    -------
    Output : GraphNode
        n-D, the same layout as Input.
    """
    n = len(data.shape)
    pad_after = pad_after if pad_after else pad_before
    if len(pad_before) != n:
        raise ValueError("Input dimension and pad_before dismatch : %d vs %d" % (
            n, len(pad_before)))
    if len(pad_after) != n:
        raise ValueError("Input dimension and pad_after dismatch : %d vs %d" % (
            n, len(pad_before)))
    out_shape = tuple(
        tvm.tir.ir_pass.Simplify(
            (data.shape[i] + pad_before[i] + pad_after[i])) for i in range(n))
    pad_value = (pad_value if isinstance(pad_value, tvm.tir.PrimExpr)
                 else tvm.tir.const(pad_value, data.dtype))

    def _inner_pad(*args, requires_grad=True):
        def _pad(*indices):
            not_zero = []
            index_tuple = []
            for i in range(n):
                if equal_const_int(pad_before[i], 0) and equal_const_int(pad_after[i], 0):
                    index_tuple.append(indices[i])
                else:
                    index_tuple.append(indices[i] - pad_before[i])
                    not_zero.append(indices[i] >= pad_before[i])
                    not_zero.append(indices[i] < data.shape[i] + pad_before[i])
            if not_zero:
                not_zero = tvm.tir.all(*not_zero)
                return tvm.tir.if_then_else(not_zero, data(*index_tuple), pad_value)
            return data(*index_tuple)
        return compute(out_shape,
                       _pad,
                       name="pad",
                       # tag=tag_gen("pad_dim" + str(len(data.shape))),
                       requires_grad=requires_grad)
    return GraphOp(out_shape, [], [data], _inner_pad, name="pad")


def ShiftConv2d_nhwc(Input, KernelIndex, KernelShape, dilation, stride):
    """
    Shift Convolution Operator
    Parameters
    ----------
    Input: GraphNode
        4-D with shape [batch_size, input_height, input_width, channels]
    KernelIndex: GraphNode
        1-D with shape [channels] integers ranging in [0, kernel_height * kernel_width)
    KernelShape: int or tuple, specify kernel height and width
    dilation: int or tuple
    stride: int or tuple
    Returns
    -------
    Output: GraphNode
        4-D with shape [batch_size, out_height, out_width, channels]
    """

    batch, inputHeight, inputWidth, channels = Input.shape
    # channels_, kernelHeight, kernelWidth = Kernel.shape
    channels_ = KernelIndex.shape[0]
    if isinstance(KernelShape, int):
        kernelHeight, kernelWidth = KernelShape, KernelShape
    else:
        assert isinstance(KernelShape, tuple) and len(
            KernelShape) == 2 and isinstance(KernelShape[0], int)
        kernelHeight, kernelWidth = KernelShape

    assert channels == channels_

    if type(dilation) == int:
        dilation = (dilation, dilation)
    if type(stride) == int:
        stride = (stride, stride)

    assert len(dilation) == 2
    assert len(stride) == 2

    padding = [((stride[0] - 1) * inputHeight - stride[0] + dilation[0] * (kernelHeight - 1) + 1) / 2,
               ((stride[1] - 1) * inputWidth - stride[1] + dilation[1] * (kernelWidth - 1) + 1) / 2]

    outHeight = (inputHeight + 2 *
                 padding[0] - dilation[0] * (kernelHeight - 1) - 1) // stride[0] + 1
    outWidth = (inputWidth + 2 * padding[1] - dilation[1]
                * (kernelWidth - 1) - 1) // stride[1] + 1

    # This pad should be equivalent to topi.nn.pad
    PInput = pad(Input, (0, padding[0], padding[1], 0),
                 (0, padding[0], padding[1], 0), name="PInput")

    # argmax(data, axis=None, keepdims=False): topi argmax function
    # kernelIndex = topi.argmax(Kernel, axis=(1, 2))

    def _inner_ShiftConv2d_nhwc(batch, out_h, out_w, channels, PInput, KernelIndex, requires_grad=True):
        return compute([batch, out_h, out_w, channels],
                       lambda n, h, w, o: PInput[n, h * stride[0] + (KernelIndex[o] // kernelHeight) * dilation[0],
                                                 w *
                                                 stride[1] + (KernelIndex[o] %
                                                              kernelWidth) * dilation[1],
                                                 o],
                       name="ShiftConv2d_nhwc",
                       # tag=tag_gen("ShiftConv2d_nhwc" + str(stride) + str(dilation) + str(KernelShape)),
                       requires_grad=requires_grad)
    Output = GraphOp([batch, outHeight, outWidth, channels], [], [PInput, KernelIndex],
                     _inner_ShiftConv2d_nhwc, name="ShiftConv2d_nhwc")
    # return PInput, kernelIndex, Output
    return Output


def reshape(A, new_shape):
    """Reshape of given tensor

    Parameters
    ----------
    A: GraphNode

    new_shape: list of tuple

    Returns
    -------
    Output: GraphNode
    """
    org_shape = A.shape
    num_ele = reduce(lambda x, y: x * y, org_shape, 1)
    num_ele_ = reduce(lambda x, y: x * y, new_shape, 1)
    assert num_ele == num_ele_

    def _inner_scatter(*args, requires_grad=True):
        Input = args[-1]
        shape = args[:-1]
        dim = len(shape)

        def scatter(*indices):
            flatten = indices[0]
            for i in range(1, dim):
                flatten = flatten * shape[i] + indices[i]
            return Input(flatten)
        return compute(shape, scatter, name="scatter", requires_grad=requires_grad)

    def _inner_gather(length, Input, requires_grad=True):
        dim = len(Input.shape)
        factors = []
        cur_factor = 1
        for i in range(0, dim):
            factors.append(cur_factor)
            cur_factor = cur_factor * Input.shape[dim - 1 - i]

        def gather(ind):
            indices = []
            cur = ind
            for i in range(dim):
                indices.append(cur // factors[dim - i - 1])
                cur = cur % factors[dim - i - 1]
            return Input(*indices)
        return compute([length], gather, name="gather", requires_grad=requires_grad)

    x = GraphOp([num_ele], [], [A], _inner_gather, name="reshape_gather")
    x = GraphOp(new_shape, [], [x], _inner_scatter, name="reshape_scatter")

    return x
