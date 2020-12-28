import tvm

from tvm.tensor_graph.core2.graph.concrete import Compute, Tensor


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def mse_loss(outputs, labels, output_dtype="float32", requires_grad=False):
    """MSE Loss function
    
    Args:
    -----------------------------
    outputs  : Tensor
        shape [batch, length]
    
    labels   : Tensor
        shape [batch, length]

    output_dtype : str

    requires_grad : bool
    -----------------------------

    Returns:
    -----------------------------
    Tensor
        shape [1]
    -----------------------------
    """
    assert len(outputs.shape) == len(labels.shape) and outputs.shape[0] == labels.shape[0] and outputs.shape[1] == labels.shape[1]
    
    def _inner_mse(_out, _label):
        def _for_spatial(i):
            def _for_reduce(b, l):
                return tvm.tir.power(_label[i+b, l] - _out[i+b, l], 2) / outputs.shape[1]
            return _for_reduce, [*outputs.shape], "sum"
        return _for_spatial
    
    return Compute([1], output_dtype, outputs, labels, fhint=_inner_mse, name="mse", requires_grad=requires_grad)