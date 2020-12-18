import inspect
import tvm
import numpy as np

from collections import Callable

from .abstract import TensorType, OperatorType, ExpressionContext
from .abstract import Operator as _Operator
from .abstract import Tensor as _Tensor
from ..utils import to_tuple


class Operator(_Operator):
    """ Concrete Operator
    
    Parameters:
        op_type: OperatorType

        fhint: Callable, optional
            that can produce concrete compute expressions

        name: optional, str

        requires_grad: bool, optional
    """
    supported_reduce_op = {"sum": tvm.te.sum, "max": tvm.te.max, "min": tvm.te.min, "none": lambda x, *args, **kwargs: x}
    def __init__(self, op_type, fhint=None, name="", requires_grad=False):
        super(Operator, self).__init__(op_type, fhint, name, requires_grad)
        assert self.fhint is not None, "Concrete Operator should have fhint."

        # inner expression context
        self.expression = None

    def make_expr(self):
        if self.expression is None:
            if isinstance(self.fhint, ExpressionContext):
                self.expression = self.fhint
            elif isinstance(self.fhint, Callable):
                try:
                    inputs = [x.placeholder for x in self.inputs]
                    for_spatial = self.fhint(*inputs)
                    spatial_shape = self.outputs[0].tensor_type.shape
                    if inspect.getfullargspec(for_spatial).varargs is not None:
                        varnames = ["iv" + str(i) for i in range(len(spatial_shape))]
                    else:
                        varnames = for_spatial.__code__.co_varnames
                    assert len(self.outputs) == 1, "Expect one and only one output."
                    varnames = varnames[:len(spatial_shape)]

                    assert len(varnames) == len(spatial_shape)
                    keys = [str(i) for i in range(len(spatial_shape))]
                    spatial_vars = []
                    for k, s in zip(keys, varnames):
                        assert k in spatial_shape, "Can't find key=%s in shape %s." % (k, str(spatial_shape))
                        spatial_vars.append(tvm.tir.IterVar((0, spatial_shape[k]), s, 0))
                    
                    for_reduce, reduction_shape, reduce_op = for_spatial(*spatial_vars)
                    reduce_varnames = for_reduce.__code__.co_varnames[:len(reduction_shape)]
                    assert len(reduce_varnames) == len(reduction_shape)
                    assert reduce_op in self.supported_reduce_op, "Don't know reduce op: %s" % str(reduce_op)
                    reduction_vars = [tvm.te.reduce_axis((0, s), x) for x, s in zip(reduce_varnames, reduction_shape)]
                    body = for_reduce(*reduction_vars)
                    if len(reduce_varnames) > 0:
                        assert reduce_op != "none"
                        body = self.supported_reduce_op[reduce_op](body, axis=reduction_vars)

                    self.expression = ExpressionContext(inputs, body, spatial_vars, reduction_vars)
                except Exception as e:
                    print(e)
                    raise RuntimeError("Can't get the expression by calling fhint(%s)." % str(self.inputs))
            else:
                raise ValueError("Don't know how to get expression context from %s." % str(self.fhint))
        
    
    def __repr__(self):
        ret = "Concrete Op("
        ret += self.name
        ret += ", "
        ret += str(self.op_type)
        ret += ", "
        ret += "requires_grad=" + str(self.requires_grad)
        ret += ")"
        return ret
    
    def __str__(self):
        ret = "Concrete Op("
        ret += self.name
        ret += ",\n"
        ret += str(self.op_type)
        ret += ",\nfhint=\n"
        if isinstance(self.fhint, Callable):
            ret += inspect.getsource(self.fhint)
        ret += ",\nexpression=\n"
        ret += str(self.expression)
        ret += ",\nrequires_grad=" + str(self.requires_grad)
        ret += "\n)"
        return ret


class Tensor(_Tensor):
    """ Concrete Tensor
    
    Parameters:
        tensor_type: TensorType

        target: str

        device: int

        name: optional, str

        requires_grad: bool, optional
    """
    def __init__(self, tensor_type, target, device, name="", requires_grad=False):
        super(Tensor, self).__init__(tensor_type, name, requires_grad)
        
        self.placeholder = tvm.te.placeholder(tensor_type.get_shape(), tensor_type.dtype, name)

        self.target = target
        self.device = device

    @classmethod
    def from_te_tensor(cls, te_tensor, target="llvm", device=0):
        shape = to_tuple(te_tensor.shape)
        dtype = te_tensor.dtype
        tensor_type = TensorType(shape, dtype)
        ret = cls(
            tensor_type, target, device,
            name=te_tensor.name, requires_grad=te_tensor.requires_grad)
        ret.placeholder = te_tensor
        return ret 

    def get_target(self):
        return self.target

    def get_device(self):
        return self.device

    def to(self, target, device):
        self.target = target
        self.device = device

    def __getitem__(self, indices):
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        return self.placeholder(*indices)

    def __call__(self, *indices):
        return self.placeholder(*indices)

    def __repr__(self):
        ret = "Concrete Tensor("
        ret += self.name
        ret += ", "
        ret += str(self.tensor_type)
        ret += ", "
        ret += "requires_grad=" + str(self.requires_grad)
        ret += ", "
        ret += self.target + "(" + str(self.device) + ")"
        ret += ")"
        return ret
    
    def __str__(self):
        ret = "Concrete Tensor("
        ret += self.name
        ret += ", "
        ret += str(self.tensor_type)
        ret += ", "
        ret += "requires_grad=" + str(self.requires_grad)
        ret += ", "
        ret += self.target + "(" + str(self.device) + ")"
        ret += ")"
        return ret


class StateTensor(Tensor):
    def __init__(self, tensor_type, target, device, name="", requires_grad=False):
        super(StateTensor, self).__init__(tensor_type, target, device, name, requires_grad)

        self.update = None

    def set_update(self, tensor):
        assert isinstance(tensor, Tensor) and self.tensor_type.match(tensor.tensor_type)
        self.update = tensor

    @property
    def valid(self):
        return self.update is not None


def FloatTensor(shape, target="llvm", device=0, name="", requires_grad=False):
    """ Get Tensor of dtype float32
    Parameters:
        shape: list/tuple

        target: str

        device: int

        name: optional, str

        requires_grad: bool
    
    Returns:
        Tensor
    """
    tensor_type = TensorType(shape, dtype="float32")
    return Tensor(tensor_type, target, device, name=name, requires_grad=requires_grad)


def Compute(out_shape, out_dtype, *inputs, fhint=None, name="", requires_grad=False):
    """ Make an operator that computes one tensor
        this API links tensors and operators automatically
    Parameters:
        out_shape: list/tuple

        out_dtype: str

        inputs: Tensor

        fhint: Callable or ExpressionContext

        name: optional, str

        requires_grad: bool
    
    Returns:
        Tensor
    """
    target = ""
    device = -1
    for inp in inputs:
        assert isinstance(inp, Tensor), "Expect Tensor as inputs, but get %s." % (str(type(inp)))
        if target == "":
            target = inp.get_target()
        else:
            if inp.get_target() != target:
                raise ValueError(
                    "The input data come from different targets %s and %s" % (target, inp.get_target()))
        if device < 0:
            device = inp.get_device()
        else:
            if inp.get_device() != device:
                raise ValueError(
                    "The input data come from different device %d and %d" % (device, inp.get_device()))
    tensor_type = TensorType(out_shape, out_dtype)

    if isinstance(out_shape, dict):
        raise ValueError("Do not use dict shape for concrete tensor.")

    out_tensor = Tensor(tensor_type, target, device, name=name, requires_grad=requires_grad)

    in_tensor_types = [inp.tensor_type for inp in inputs]
    op_type = OperatorType(in_tensor_types, [out_tensor.tensor_type])
    op = Operator(op_type, fhint=fhint, name=name, requires_grad=requires_grad)
    op.set_inputs(inputs)
    op.set_outputs([out_tensor])
    op.make_expr()

    out_tensor.set_producer_op(op, 0)
    return out_tensor


class ComputeDAGMaker(object):
    def __init__(self, reserve_placeholder=False):
        self.visited = {}
        self.reserve_placeholder=reserve_placeholder

    def make_compute_from_tensor(self, concrete_tensor):
        shape = concrete_tensor.shape
        dtype = concrete_tensor.dtype
        name = concrete_tensor.name
        if self.reserve_placeholder:
            tensor = concrete_tensor.placeholder
        else:
            tensor = tvm.te.placeholder(shape, dtype, name=name, requires_grad=concrete_tensor.requires_grad)
        return tensor

    def make_compute_from_op(self, concrete_tensor):
        shape = concrete_tensor.shape
        op = concrete_tensor.producer_op
        context = op.expression
        assert context is not None
        inputs = []
        for inp in op.inputs:
            inputs.append(self.make_compute_dag(inp))

        reduce_axis = [tvm.te.reduce_axis((0, x.dom.extent), x.var.name) for x in context.reduce_axis]

        assert len(shape) == len(context.axis)

        def compute_func(*indices):
            return tvm.tg.substitute_expression(
                context.expression,
                context.inputs, inputs,
                [x.var for x in context.axis], indices,
                context.reduce_axis, reduce_axis)
        
        tensor = tvm.te.compute(shape, compute_func, name=op.name, requires_grad=op.requires_grad)
        return tensor

    def make_compute_dag(self, concrete_tensor):
        assert isinstance(concrete_tensor, Tensor), concrete_tensor
        if concrete_tensor in self.visited:
            return self.visited[concrete_tensor]
        if concrete_tensor.producer_op is None:
            ret = self.make_compute_from_tensor(concrete_tensor)
        else:
            ret = self.make_compute_from_op(concrete_tensor)
        self.visited[concrete_tensor] = ret
        return ret

    def __call__(self, concrete_tensors):
        ret = []
        for tensor in concrete_tensors:
            ret.append(self.make_compute_dag(tensor))
        return ret, self.visited


class ModelMaker(object):
    def __init__(self, target, device):
        self.visited = {}
        self.target = target
        self.device = device

    def make_tensor(self, te_tensor):
        shape = to_tuple(te_tensor.shape)
        dtype = te_tensor.dtype
        tensor_type = TensorType(shape, dtype)
        return Tensor(
            tensor_type, self.target, self.device,
            name=te_tensor.name, requires_grad=te_tensor.requires_grad)

    def make_op(self, te_tensor):
        inputs = []
        for inp in te_tensor.op.input_tensors:
            inputs.append(self.make_model(inp))
        out_shape = to_tuple(te_tensor.shape)
        out_dtype = te_tensor.dtype
        assert len(te_tensor.op.body) == 1, "Don't support multi-body compute."
        body = te_tensor.op.body[0]
        expression_context = ExpressionContext(
            list(te_tensor.op.input_tensors),
            body,
            list(te_tensor.op.axis),
            list(te_tensor.op.reduce_axis)
        )
        return Compute(out_shape, out_dtype, *inputs,
            fhint=expression_context, name=te_tensor.op.name, requires_grad=te_tensor.op.requires_grad)

    def make_model(self, te_tensor):
        assert isinstance(te_tensor, tvm.te.tensor.Tensor)
        if te_tensor in self.visited:
            return self.visited[te_tensor]
        if isinstance(te_tensor.op, tvm.te.tensor.PlaceholderOp):
            ret = self.make_tensor(te_tensor)
        else:
            ret = self.make_op(te_tensor)
        self.visited[te_tensor] = ret
        return ret

    def __call__(self, te_tensors):
        ret = []
        for tensor in te_tensors:
            ret.append(self.make_model(tensor))
        return ret, self.visited