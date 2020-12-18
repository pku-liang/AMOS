import tvm

from tensor_graph.core2.graph.concrete import Compute, Tensor


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def zero_pad2d(inputs, padding=0, output_dtype="float32", requires_grad=False):
    """
    Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : Tensor
        shape [batch, channel, height, width]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    output_dtype : str
    requires_grad : bool
    -----------------------------
    
    Returns:
    -----------------------------
    Tensor
        shape [batch, channel, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert (len(padding) == 4)

    if all([padding[i] == 0 for i in range(len(padding))]):
        return inputs

    batch_size, in_channel, height, width = inputs.shape
    padded_shape = (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3])
    padding_value = tvm.tir.expr.const(0, output_dtype)
    def _inner_zero_pad2d(inputs):
        def _for_spatial(b, c, h, w):
            def _for_reduce():
                return tvm.te.if_then_else(
                            tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c, h - padding[0], w - padding[2]],
                            padding_value
                            )
            return _for_reduce, [], "none"
        return _for_spatial
    
    return Compute(padded_shape, output_dtype , inputs, fhint=_inner_zero_pad2d, name="zero_pad2d", requires_grad=requires_grad)
