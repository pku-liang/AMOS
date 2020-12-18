from collections import Callable


class TensorType(object):
    """ Type of a Tensor <shape, dtype>
        shape is a dict {name:extent}, dtype is a str
    Parameters:
        shape: list/tuple

        dtype: str, optional
            e.g., int32, float32
    """
    supported_dtypes = ["float32", "int32"]
    def __init__(self, shape, dtype="float32"):
        shape_dict = {}
        shape_order = []
        if isinstance(shape, (list, tuple)):
            for i, s in enumerate(shape):
                assert isinstance(s, int), "Expect shape composed of integers."
                shape_dict[str(i)] = s
                shape_order.append(str(i))
        # elif isinstance(shape, dict):
        #     for k, v in shape.items():
        #         assert isinstance(k, str), "Expect string name for dimensions of shape."
        #         assert isinstance(v, int), "Expect shape composed of integers."
        #         shape_dict[k] = v
        else:
            raise ValueError("Don't know how to handle shape of type %s." % str(type(shape)))
        self.shape = shape_dict
        self.shape_order = shape_order

        assert dtype in self.supported_dtypes, "Can't support dtype %s." % dtype
        self.dtype = dtype

    def match(self, another):
        if not isinstance(another, self.__class__):
            return False
        if len(self.shape_order) != len(another.shape_order):
            return False
        for a, b in zip(self.shape_order, another.shape_order):
            if a != b:
                return False
        if len(self.shape) != len(another.shape):
            return False
        for k in self.shape_order:
            if k not in another.shape:
                return False
            if self.shape[k] != another.shape[k]:
                return False
        return self.dtype == another.dtype

    def get_shape(self):
        impl_shape = [self.shape[x] for x in self.shape_order]
        return impl_shape

    def __eq__(self, another):
        return self.match(another)

    def __str__(self):
        ret = "<"
        ret += str(self.get_shape())
        ret += ", "
        ret += str(self.dtype)
        ret += ">"
        return ret

    def __repr__(self):
        return self.__str__()


class OperatorType(object):
    """ Type of an Operator in_tensor_types->out_tensor_types

    Parameters:
        in_tensor_types: list/tuple of input tensor types
            the order matters

        out_tensor_types: list/tuple of output tensor types
            the order matters
    """
    def __init__(self, in_tensor_types, out_tensor_types):
        for to_check in [in_tensor_types, out_tensor_types]:
            if isinstance(to_check, (list, tuple)):
                for v in to_check:
                    assert isinstance(v, TensorType), "Expect TensorType."
            else:
                raise ValueError("Expect list of tuple as inputs.")
        self.in_tensor_types = in_tensor_types
        self.out_tensor_types = out_tensor_types

    def match(self, another):
        if not isinstance(another, self.__class__):
            return False
        if len(self.in_tensor_types) != len(another.in_tensor_types):
            return False
        for a, b in zip(self.in_tensor_types, another.in_tensor_types):
            if not a.match(b):
                return False
        if len(self.out_tensor_types) != len(another.out_tensor_types):
            return False
        for a, b in zip(self.out_tensor_types, another.out_tensor_types):
            if not a.match(b):
                return False
        return True

    def __eq__(self, another):
        return self.match(another)
    
    def __str__(self):
        ret = ""
        ret += str(self.in_tensor_types)
        ret += "->"
        ret += str(self.out_tensor_types)
        return ret

    def __repr__(self):
        return self.__str__()


class ExpressionContext(object):
    def __init__(self, inputs, expression, axis, reduce_axis):
        self.inputs = inputs
        self.expression = expression
        self.axis = axis
        self.reduce_axis = reduce_axis

    def __str__(self):
        ret = "ExpressionContext(\n"
        ret += "inputs=" + str(self.inputs) + "\n"
        ret += "expr=" + str(self.expression) + "\n"
        ret += "axis=" + str(self.axis) + "\n"
        ret += "reduce_axis=" + str(self.reduce_axis) + "\n"
        return ret


class Operator(object):
    """ Abstract Operator is stateless, stand-alone
        A function from tensors to tensors
    
    Parameters:
        op_type: OperatorType

        fhint: Callable, optional
            that can produce concrete compute expressions

        name: optional, str

        requires_grad: bool, optional
    """
    def __init__(self, op_type, fhint=None, name="", requires_grad=False):
        assert isinstance(op_type, OperatorType)
        if fhint is not None:
            assert isinstance(fhint, (Callable, ExpressionContext))
        assert isinstance(requires_grad, (bool, int)), type(requires_grad)
        requires_grad = bool(requires_grad)

        self.op_type = op_type
        self.fhint = fhint
        self.name = name
        self.requires_grad = requires_grad

        self.inputs = []
        self.outputs = []

    def requires_grad_(self):
        self.requires_grad = True

    def _validate_types(self, types, tensors):
        if len(types) != len(tensors):
            return False
        for t, tensor in zip(types, tensors):
            assert isinstance(tensor, Tensor)
            if not t.match(tensor.tensor_type):
                return False
        return True

    def set_inputs(self, inputs):
        assert isinstance(inputs, (list, tuple))
        assert self._validate_types(self.op_type.in_tensor_types, inputs), \
            "Can't match expected tensor types %s and %s." % (str(self.op_type.in_tensor_types), str(inputs))
        self.inputs = inputs
    
    def set_outputs(self, outputs):
        assert isinstance(outputs, (list, tuple))
        assert self._validate_types(self.op_type.out_tensor_types, outputs), \
            "Can't match expected tensor types %s and %s." % (str(self.op_type.out_tensor_types), str(outputs))
        self.outputs = outputs

    def __repr__(self):
        ret = "Abstract Op("
        ret += self.name
        ret += ", "
        ret += str(self.op_type)
        ret += ", "
        ret += "requires_grad=" + str(self.requires_grad)
        ret += ")"
        return ret
    
    def __str__(self):
        ret = "Abstract Op("
        ret += self.name
        ret += ", "
        ret += str(self.op_type)
        ret += ", "
        ret += "requires_grad=" + str(self.requires_grad)
        ret += ")"
        return ret


class Tensor(object):
    """ Abstract Tensor
        only shape and dtype, no access support
    
    Parameters:
        tensor_type: TensorType

        name: optional, str

        requires_grad: bool, optional
    """
    def __init__(self, tensor_type, name="", requires_grad=False):
        assert isinstance(tensor_type, TensorType)
        assert isinstance(requires_grad, (bool, int)), type(requires_grad)
        requires_grad = bool(requires_grad)

        self.tensor_type = tensor_type
        self.name = name
        self.requires_grad = requires_grad

        self.producer_op = None
        self.producer_index = 0

    @property
    def shape(self):
        return self.tensor_type.get_shape()

    @property
    def dtype(self):
        return self.tensor_type.dtype

    def requires_grad_(self):
        self.requires_grad = True

    def set_producer_op(self, op, index):
        assert isinstance(op, Operator)
        out_types = op.op_type.out_tensor_types
        assert index >= 0 and index < len(out_types)
        target_type = out_types[index]
        if not target_type.match(self.tensor_type):
            raise ValueError(
                "Can't math producer output type %s with %s" % (str(target_type), str(self.tensor_type)))
        self.producer_op = op
        self.producer_index = index

    def __repr__(self):
        ret = "Abstract Tensor("
        ret += self.name
        ret += ", "
        ret += str(self.tensor_type)
        ret += ", "
        ret += "requires_grad=" + str(self.requires_grad)
        ret += ")"
        return ret
    
    def __str__(self):
        ret = "Abstract Tensor("
        ret += self.name
        ret += ", "
        ret += str(self.tensor_type)
        ret += ", "
        ret += "requires_grad=" + str(self.requires_grad)
        ret += ")"
        return ret


def FloatTensor(shape, name="", requires_grad=False):
    """ Get Tensor of dtype float32
    Parameters:
        shape: list/tuple

        name: optional, str

        requires_grad: bool
    
    Returns:
        Tensor
    """
    tensor_type = TensorType(shape, dtype="float32")
    return Tensor(tensor_type, name=name, requires_grad=requires_grad)


def Compute(out_shape, out_dtype, *inputs, fhint=None, name="", requires_grad=False):
    """ Make an operator that computes one tensor
        this API links tensors and operators automatically
    Parameters:
        out_shape: list/tuple

        out_dtype: str

        inputs: Tensor

        fhint: Callable

        name: optional, str

        requires_grad: bool
    
    Returns:
        Tensor
    """
    for inp in inputs:
        assert isinstance(inp, Tensor), "Expect Tensor as inputs, but get %s." % (str(type(inp)))
    tensor_type = TensorType(out_shape, out_dtype)
    out_tensor = Tensor(tensor_type, requires_grad=requires_grad)
    in_tensor_types = [inp.tensor_type for inp in inputs]
    op_type = OperatorType(in_tensor_types, [out_tensor.tensor_type])
    op = Operator(op_type, fhint=fhint, name=name, requires_grad=requires_grad)
    op.set_inputs(inputs)
    op.set_outputs([out_tensor])
    out_tensor.set_producer_op(op, 0)
    return out_tensor

