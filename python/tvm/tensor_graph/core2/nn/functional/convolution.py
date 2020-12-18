import tvm

from tensor_graph.core2.graph.concrete import Compute, Tensor

from .padding import zero_pad2d


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
    output_dtype="float32", requires_grad=False):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : Tensor
        shape [batch, channel, height, width]
    weight  : Tensor
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert channel_per_group * groups == in_channel, "%d vs. %d" % (channel_per_group * groups, in_channel)
    out_channel_per_group = out_channel // groups
    assert out_channel_per_group * groups == out_channel

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

    padded = zero_pad2d(inputs, padding=padding, output_dtype=output_dtype, requires_grad=requires_grad)

    conv_out_shape = (batch_size, out_channel, out_h, out_w)

    if bias is not None:
        if groups > 1:
            def _inner_conv2d_nchw(padded, weight, bias):
                def _for_spatial(b, c, h, w):
                    def _for_reduce(rc, rw, rh):
                        return (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
                                * weight[c, rc, rh, rw]) + bias[c] / (channel_per_group*k_w*k_h)
                    return _for_reduce, [channel_per_group, k_w, k_h], "sum"
                return _for_spatial

            conv_out = Compute(conv_out_shape, output_dtype, padded, weight, bias,
                fhint=_inner_conv2d_nchw, name="conv2d_nchw", requires_grad=requires_grad)
            return conv_out
        else:
            def _inner_conv2d_nchw(padded, weight, bias):
                def _for_spatial(b, c, h, w):
                    def _for_reduce(rc, rw, rh):
                        return (padded[b, rc, 
                                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
                                * weight[c, rc, rh, rw]) + bias[c] / (channel_per_group*k_w*k_h)
                    return _for_reduce, [channel_per_group, k_w, k_h], "sum"
                return _for_spatial

            conv_out = Compute(conv_out_shape, output_dtype, padded, weight, bias,
                fhint=_inner_conv2d_nchw, name="conv2d_nchw", requires_grad=requires_grad)
            return conv_out
    else:
        if groups > 1:
            def _inner_conv2d_nchw(padded, weight):
                def _for_spatial(b, c, h, w):
                    def _for_reduce(rc, rw, rh):
                        return (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                                h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
                                * weight[c, rc, rh, rw])
                    return _for_reduce, [channel_per_group, k_w, k_h], "sum"
                return _for_spatial

            conv_out = Compute(conv_out_shape, output_dtype, padded, weight,
                fhint=_inner_conv2d_nchw, name="conv2d_nchw", requires_grad=requires_grad)
            return conv_out
        else:
            def _inner_conv2d_nchw(padded, weight):
                def _for_spatial(b, c, h, w):
                    def _for_reduce(rc, rw, rh):
                        return (padded[b, rc, 
                                h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
                                * weight[c, rc, rh, rw])
                    return _for_reduce, [channel_per_group, k_w, k_h], "sum"
                return _for_spatial

            conv_out = Compute(conv_out_shape, output_dtype, padded, weight,
                fhint=_inner_conv2d_nchw, name="conv2d_nchw", requires_grad=requires_grad)
            return conv_out