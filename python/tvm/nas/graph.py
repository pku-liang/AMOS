import tvm._ffi
import tvm
from . import _ffi_api
from tvm.runtime import Object


def layer_tensor(shape, name="layer_tensor", dtype="float32"):
    t = tvm.te.placeholder(shape, name=name, dtype=dtype)
    return _ffi_api.LayerTensor(name, None, t, 0)


@tvm._ffi.register_object("nas.LayerTensor")
class LayerTensor(Object):
    """LayerTensor object"""


@tvm._ffi.register_object("nas.Layer")
class Layer(Object):
    """Layer object"""

    def __call__(self, *inputs):
        ninp = self.num_inputs
        if len(inputs) != ninp:
            raise ValueError(
                f"Need to provide {ninp} inputs, but get {len(inputs)}.")
        for inp in inputs:
            assert isinstance(inp, LayerTensor)
        ret = _ffi_api.ProduceOutputs(self, inputs)
        if len(ret) == 1:
            return ret[0]

    # def __getitem__(self, indices):
    #     return TensorSlice(self, indices)

    # def __hash__(self):
    #     return _ffi_api.TensorHash(self)

    # def __eq__(self, other):
    #     if not isinstance(other, Tensor):
    #         if isinstance(other, _expr.ExprOp):
    #             return _expr.EqualOp(self, other)
    #         return False
    #     if self.ndim == 0 and other.ndim == 0:
    #         raise ValueError(
    #             "Equal == comparison among rank-0 tensor is ambiguous, "
    #             "use Tensor.equal for content expression equvalence, "
    #             "use Tensor.same_as for exact reference comparison"
    #         )
    #     return _ffi_api.TensorEqual(self, other)

    @property
    def num_inputs(self):
        """Number of inputs required by this layer."""
        return len(self.inputs)

    # @property
    # def axis(self):
    #     """Axis of the tensor."""
    #     return self.__getattr__("axis")

    # @property
    # def op(self):
    #     """The corressponding :py:class:`Operation`."""
    #     return self.__getattr__("op")

    # @property
    # def value_index(self):
    #     """The output value index the tensor corresponds to."""
    #     return self.__getattr__("value_index")

    # @property
    # def shape(self):
    #     """The output shape of the tensor."""
    #     return self.__getattr__("shape")

    # @property
    # def requires_grad(self):
    #     """The requires_grad of the tensor."""
    #     return self.__getattr__("requires_grad")

    # @property
    # def name(self):
    #     op = self.op
    #     if op.num_outputs == 1:
    #         return op.name
    #     return "%s.v%d" % (op.name, self.value_index)


def layer(ops, inputs=None, weights=None, const_scalars=None,
              const_tensors=None, gradients=None, requires_grad=False, name="layer"):
    """Make a network layer through IR.

    Parameters
    ----------
    ops : te.Operation or List[te.Operation] (not empty)
        The output operations that make this layer.

    inputs : optional List[te.Tensor]
        The list of input tensors.

    weights : optional List[te.Tensor]
        The list of weights.

    const_scalars : optional List[PrimExpr]
        The list of constant scalar values.

    const_tensors : optional List[te.Tensor]
        The list of constant tensors.

    gradients : optional List[te.Tensor]
        The list of gradients.

    requires_grad : optional bool
        If gradients is None and requires_grad is set True, autodiff is used
        to calculate the gradients for this layer.

    name: str

    Returns
    -------
    tensors: nast.Layer
        The result layer

    Example
    -------
    .. code-block:: python

        x = tvm.te.placeholder((32, 3, 28, 28), name='x')
        w1 = tvm.te.placeholder((10, 3, 3, 3), name='w1')
        w2 = tvm.te.placeholder((10, 10, 3, 3), name='w2')
        z1 = topi.nn.conv2d(x, w1, 1, 1, 1)
        z2 = topi.nn.conv2d(z1, w2, 1, 1, 1)
        y = topi.sum(z2)

        # make a layer
        layer = MakeLayer(y.op, inputs=[x], weights=[w1, w2])

    """
    if not isinstance(ops, list):
        ops = [ops]
    if inputs is None:
        inputs = []
    if weights is None:
        weights = []
    if const_scalars is None:
        const_scalars = []
    if const_tensors is None:
        const_tensors = []
    if gradients is None:
        if requires_grad:
            # TODO: integrate autodiff into this function
            raise RuntimeError("Currently not support autodiff in MakeLayer")
        else:
            gradients = []
    return _ffi_api.MakeLayer(name, ops, inputs, weights,
                              const_scalars, const_tensors, gradients)
    
    
@tvm._ffi.register_object("nas.Graph")
class Graph(Object):
    """Graph object"""
    
    
def graph(out_tensors, name="graph"):
    if not isinstance(out_tensors, list):
        out_tensors = [out_tensors]
    return _ffi_api.Graph(name, out_tensors)