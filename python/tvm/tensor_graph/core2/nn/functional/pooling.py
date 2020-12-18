import tvm

from tensor_graph.core2.graph.concrete import Compute, Tensor

from .padding import zero_pad2d


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def avgpool2d(inputs, kernel_h=2, kernel_w=2, stride_h=2, stride_w=2, padding=0, output_dtype="float32", requires_grad=False):
    """Average Pooling 4D Tensor with the last two dim [height, width]
    
    Args:
    --------------------------
    inputs: Tensor
        shape: [batch, channel, height, width]
    Stride, padding are also supported

    out_height = (height + 2*padding - kernel_h) // stride_h + 1

    output_dtype : str
    requires_grad : bool

    Returns:
    --------------------------
    Tensor
        shape: [batch, channel, out_height, out_width]
    """
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    padded_inputs = zero_pad2d(inputs, padding=padding, requires_grad=requires_grad)

    batch, channel, h_in, w_in = padded_inputs.shape

    h_out = (h_in - kernel_h) // stride_h + 1
    w_out = (w_in - kernel_w) // stride_w + 1

    def _inner_avgpool2d(padded_inputs):
        def _for_spatial(n, c, i, j):
            def _for_reduce(r, s):
                return padded_inputs[n, c, i * stride_h + r, j * stride_w + s]/(kernel_h*kernel_w)
            return _for_reduce, [kernel_h, kernel_w], "sum"
        return _for_spatial
    return Compute([batch, channel, h_out, w_out], output_dtype, padded_inputs,
        fhint=_inner_avgpool2d, name="avgpool2d", requires_grad=requires_grad)


def global_avg_pool2d(inputs, keep_dim=True, requires_grad=False):
    """Global Average Pooling for GraphOp

    Args:
    --------------------------
    inputs: Tensor
        shape: [batch, channel, height, width]

    keep_dim: bool
    
    Returns:
    --------------------------
    Tensor
        shape: [batch, channel, 1, 1] if keep dim is True
        else [batch, channel]
    """
    N, C, H, W = inputs.shape
    def _innner_global_avg_pool2d_keep(inputs):
        def _for_spatial(n, c, i, j):
            def _for_reduce(h, w):
                return inputs[n, c, i + h, j + w]/(H*W)
            return _for_reduce, [H, W], "sum"
        return _for_spatial

    def _innner_global_avg_pool2d_nokeep(inputs):
        def _for_spatial(n, c):
            def _for_reduce(h, w):
                return inputs[n, c, h, w]/(H*W)
            return _for_reduce, [H, W], "sum"
        return _for_spatial
    
    if keep_dim:
        return Compute([N, C, 1, 1], inputs.dtype, inputs, fhint=_innner_global_avg_pool2d_keep, name="global_avg_pool2d", requires_grad=requires_grad)
    else:
        return Compute([N, C], inputs.dtype, inputs, fhint=_innner_global_avg_pool2d_nokeep, name="global_avg_pool2d", requires_grad=requires_grad)
