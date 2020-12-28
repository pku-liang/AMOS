import tvm

from tvm.tensor_graph.core2.graph.concrete import Compute, Tensor


######################################################################
# for functional, all states are inputs, data from inside functionals
# can only be constants
######################################################################


def flatten(inputs, output_dtype="float32", requires_grad=False):
    '''
    inputs: [batch, channel, height, width]
    return: [batch, channel * height * width]
    '''
    assert len(inputs.shape) == 4
    batch, channel, height, width = inputs.shape
    def _inner_batch_flatten(inputs):
        def _for_spatial(i, j):
            def _for_reduce():
                return inputs[i, j//(height * width), (j%(height * width)) // width, j % width]
            return _for_reduce, [], "none"
        return _for_spatial

    return Compute(
        [batch, channel * height * width], output_dtype, inputs,
        fhint=_inner_batch_flatten, name="flatten", requires_grad=requires_grad)