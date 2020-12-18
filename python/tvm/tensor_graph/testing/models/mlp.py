from collections import OrderedDict

from tensor_graph.nn import Linear
from tensor_graph.nn.functional import flatten
from tensor_graph.nn.modules.model import Model


class MLP(Model):

    def __init__(self):
        super().__init__()
        self.fc = Linear('fc3', 28*28, 10, bias=False)

    def __call__(self, inputs, debug_mode=False):
        debug_tensors = OrderedDict()

        def D(tensor, key):
            if debug_mode:
                debug_tensors[key] = tensor
            return tensor

        outputs = D(flatten(inputs), 'flatten')
        outputs = self.fc(outputs)

        if debug_mode: return outputs, debug_tensors
        else: return outputs

    @property
    def weights(self):
        modules = [self.fc]
        return sum([m.weights for m in modules], [])
