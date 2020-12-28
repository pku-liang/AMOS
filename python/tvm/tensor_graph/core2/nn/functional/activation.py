import tvm

from tvm.tensor_graph.core2.graph.concrete import Compute, Tensor


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def ReLU(x, output_dtype="float32", requires_grad=False):
    """Take relu of input x.

    Parameters
    ----------
    x : Tensor
        Arbitrary dimension Input argument.
    output_dtype : str
    requires_grad : bool

    Returns
    -------
    Tensor
        The result.
    """
    def _inner_ReLU(tensor):
        def _for_spatial(*args):
            def _for_reduce():
                return tvm.te.max(tensor(*args), tvm.tir.const(0, output_dtype))
            return _for_reduce, [], "none"
        return _for_spatial
    return Compute([*x.shape], output_dtype, x, fhint=_inner_ReLU, name="ReLU", requires_grad=requires_grad)