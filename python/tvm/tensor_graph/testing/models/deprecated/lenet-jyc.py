from collections import OrderedDict

from tensor_graph.nn import Conv2d, Linear
from tensor_graph.nn.functional import relu, avg_pool2d, flatten
from tensor_graph.nn.modules.model import Model


class LeNet(Model):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d('conv1', 1, 6, 3, stride=1, padding=1, bias=False)
        self.conv2 = Conv2d('conv2', 6, 16, 5, stride=1, padding=0, bias=False)
        self.fc1 = Linear('fc1', 400, 120, bias=False)
        self.fc2 = Linear('fc2', 120, 84, bias=False)
        self.fc3 = Linear('fc3', 84, 10, bias=False)

    def __call__(self, inputs, debug_mode=False):
        debug_tensors = OrderedDict()

        def D(tensor, key):
            if debug_mode:
                debug_tensors[key] = tensor
            return tensor

        outputs = D(self.conv1(inputs), 'conv1')
        outputs = D(relu(outputs), 'relu1')
        outputs = D(avg_pool2d(outputs), 'pool1')
        outputs = D(self.conv2(outputs), 'conv2')
        outputs = D(relu(outputs), 'relu2')
        outputs = D(avg_pool2d(outputs), 'pool2')
        outputs = D(flatten(outputs), 'flatten')
        outputs = D(self.fc1(outputs), 'fc1')
        outputs = D(self.fc2(outputs), 'fc2')
        outputs = D(relu(outputs), 'relu3')
        outputs = D(self.fc3(outputs), 'fc3')
        outputs = relu(outputs)

        if debug_mode: return outputs, debug_tensors
        else: return outputs

    @property
    def weights(self):
        modules = [self.conv1, self.conv2, self.fc1, self.fc2, self.fc3]
        return sum([m.weights for m in modules], [])
