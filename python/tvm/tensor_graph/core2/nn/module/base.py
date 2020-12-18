import tvm
from tensor_graph.core2.graph.concrete import Tensor, StateTensor


class Module(object):
    def __init__(self):
        self._train = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def requires_grad(self):
        ret = False
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                ret = ret or v.requires_grad
            if isinstance(v, Tensor):
                ret = ret or v.requires_grad
        return ret

    @property
    def weights(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                for w in v.weights:
                    yield w
            if isinstance(v, Tensor) and v.requires_grad:
                yield v

    @property
    def states(self):
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                for w in v.states:
                    yield w
            if isinstance(v, StateTensor):
                yield v

    def dumps(self, indent_bias=0, indent_factor=2):
        ret = indent_bias * " " + self.__class__.__name__ + "(\n"
        indent = indent_bias + indent_factor
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor):
                ret += " " * indent + str(v) + "\n"
            if isinstance(v, Module):
                ret += v.dumps(indent, indent_factor)
        ret += indent_bias * " " + ")\n"
        return ret

    def eval(self):
        self._train = False
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                v.eval()
            if isinstance(v, Tensor):
                v.requires_grad = False
    
    def train(self):
        self._train = True
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                v.train()
            if isinstance(v, Tensor):
                v.requires_grad = True

    def to(self, target, device):
        for k, v in self.__dict__.items():
            if isinstance(v, Module):
                v.to(target, device)
            if isinstance(v, Tensor):
                v.to(target, device)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)