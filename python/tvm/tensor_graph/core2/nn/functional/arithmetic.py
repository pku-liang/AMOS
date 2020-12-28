import tvm

from tvm.tensor_graph.core2.graph.concrete import Compute, Tensor


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def add_no_broadcast(x, y, requires_grad=False):
    """Add x and y without broadcast.

    Parameters
    ----------
    x : Tensor
        Arbitrary dimension Input argument.

    x : Tensor
        Arbitrary dimension Input argument.

    requires_grad : bool

    Returns
    -------
    Tensor
        The result.
    """
    assert x.tensor_type == y.tensor_type, str(x.tensor_type) + " vs. " + str(y.tensor_type)
    def _inner_add(a, b):
        def _for_spatial(*args):
            def _for_reduce():
                return a(*args) + b(*args)
            return _for_reduce, [], "none"
        return _for_spatial
    return Compute([*x.shape], x.dtype, x, y, fhint=_inner_add, name="add_no_broadcast", requires_grad=requires_grad)