from tvm.tensor_graph.core2.graph.concrete import Tensor, FloatTensor, Compute
from tvm.tensor_graph.core2.nn import functional as F

from .base import StateTensor, Module


class MSELoss(Module):
    def __init__(self):
        pass

    def forward(self, outputs, labels):
        if isinstance(labels, (list, tuple)):
            assert len(labels) == 1, "MSE should only take one label"
            labels = labels[0]
        if isinstance(outputs, (list, tuple)):
            assert len(outputs) == 1, "MSE should only take one output"
            outputs = outputs[0]
        return F.mse_loss(outputs, labels, output_dtype="float32", requires_grad=outputs.requires_grad)