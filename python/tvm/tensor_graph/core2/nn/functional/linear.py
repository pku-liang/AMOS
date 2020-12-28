import tvm

from tvm.tensor_graph.core2.graph.concrete import Compute, Tensor


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def linear(inputs, weight, bias=None, output_dtype="float32", requires_grad=False):
    """Linear function
    Args:
    -----------------------------
    inputs: Tensor
        shape [batch, ..., in_feature]
    weight: Tensor
        shape [out_feature, in_feature]
    bias  : Tensor
        shape [out_feature]
    output_dtype : str
    requires_grad : bool
    -----------------------------
    Returns:
    -----------------------------
    Tensor
        shape [batch, ..., out_feature]
    -----------------------------
    """
    assert(inputs.shape[-1] == weight.shape[1])

    K = inputs.shape[-1]

    if bias is None:
        def _inner_linear(inputs, weight):
            def _for_spatial(*spatial_indices):
                def _for_reduce(k):
                    return inputs[(*spatial_indices[:-1], k)] * weight[spatial_indices[-1], k]
                return _for_reduce, [K], "sum"
            return _for_spatial

        return Compute([*inputs.shape[:-1], weight.shape[0]], output_dtype, inputs, weight,
                fhint=_inner_linear, name="linear", requires_grad=requires_grad)
    else:
        def _inner_linear(inputs, weight, bias):
            def _for_spatial(*spatial_indices):
                def _for_reduce(k):
                    return inputs[(*spatial_indices[:-1], k)] * weight[spatial_indices[-1], k] + bias[spatial_indices[-1]]/K
                return _for_reduce, [K], "sum"
            return _for_spatial

        return Compute([*inputs.shape[:-1], weight.shape[0]], output_dtype, inputs, weight,
                fhint=_inner_linear, name="linear", requires_grad=requires_grad)

