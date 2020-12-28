import tvm

from tvm.tensor_graph.core2.graph.concrete import Compute, Tensor


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def batch_norm2d(
    inputs, alpha, beta, epsilon=1e-5, momentum=0.1,
    track_running_stats=True, tracking_mean=None, tracking_var=None,
    output_dtype="float32", requires_grad=False):
    """2D Batch Normalization for NCHW inputs
    
    Args:
    -----------------------------
    inputs  : Tensor
        shape [batch, channel, height, width]
    alpha   : Tensor
        shape [channel]
    beta    : Tensor
        shape [channel]
    epsilon : float
        optional
    momentum : float
        optional
    track_running_stats : bool
        whether for track mean and variance

    tracking_mean : optional, Tensor
        the tracking mean, shape [channel]
    
    tracking_var : optional, Tensor
        the tracking variance, shape [channel]

    output_dtype : str

    requires_grad : bool
    -----------------------------

    Returns:
    -----------------------------
    Tensor
        shape [batch, channel, height, width]
    if tracking_running_stats also return new mean and var
        of shape [channel]
    -----------------------------
    """
    assert isinstance(inputs, Tensor)
    assert isinstance(alpha, Tensor)
    assert isinstance(beta, Tensor)
    N, C, H, W = inputs.shape
    epsilon = tvm.tir.const(epsilon, inputs.dtype)

    assert (len(alpha.shape) == 1) and (alpha.shape[0] == C)
    assert (len(beta.shape) == 1) and (beta.shape[0] == C)

    def _inner_mean(inputs):
        def _for_spatial(c):
            def _for_reduce(rn, rh, rw):
                return inputs[rn, c, rh, rw] / (N*H*W)
            return _for_reduce, [N, H, W], "sum"
        return _for_spatial

    mean = Compute([C], output_dtype, inputs, fhint=_inner_mean, name="bn_mean", requires_grad=requires_grad)

    def _inner_square(inputs):
        def _for_spatial(c):
            def _for_reduce(rn, rh, rw):
                return tvm.tir.power(inputs[rn, c, rh, rw], 2) / (N*H*W)
            return _for_reduce, [N, H, W], "sum"
        return _for_spatial
    
    square = Compute([C], output_dtype, inputs, fhint=_inner_square, name="bn_sqr_mean", requires_grad=requires_grad)

    def _inner_var(square, mean):
        def _for_spatial(c):
            def _for_reduce():
                return square[c] - tvm.tir.power(mean[c], 2)
            return _for_reduce, [], "none"
        return _for_spatial

    var = Compute([C], output_dtype, square, mean, fhint=_inner_var, name="bn_var", requires_grad=requires_grad)

    def _inner_bn(inputs, mean, var, alpha, beta):
        def _for_spatial(n, c, i, j):
            def _for_reduce():
                return (inputs[n, c, i, j] - mean[c]) / tvm.te.sqrt(var[c] + epsilon) * alpha[c] + beta[c]
            return _for_reduce, [], "none"
        return _for_spatial

    ret = Compute([N, C, H, W], output_dtype, inputs, mean, var, alpha, beta,
        fhint=_inner_bn, name="bn", requires_grad=requires_grad)

    if track_running_stats:
        assert tracking_mean is not None and tracking_var is not None

        def _momentum_update(original, update):
            def _for_spatial(c):
                def _for_reduce():
                    return (1 - momentum) * original[c] + momentum * update[c]
                return _for_reduce, [], "none"
            return _for_spatial
        new_mean = Compute([C], output_dtype, tracking_mean, mean, fhint=_momentum_update, name="bn_update_mean", requires_grad=False)
        new_var = Compute([C], output_dtype, tracking_var, var, fhint=_momentum_update, name="bn_update_var", requires_grad=False)
        return ret, new_mean, new_var
    else:
        return ret