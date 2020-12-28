import tvm
import numpy as np
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, ForwardGraph, \
                              BackwardGraph, GraphMutator, PyTIRGraph
import tensor_graph.nn.functional as F
from tvm.tensor_graph.nn.modules.loss import CELoss
from tvm.tensor_graph.nn.modules.optimize import SGD

from tvm.tensor_graph.nn.layers import Layer, Conv2d, BatchNorm2d, ReLU, \
                                  AvgPool2d, GlobalAvgPool2d, Linear, Sequential


class YOLONetV1(Layer):
    def __init__(self, channel=3, height=448, width=448):
        super(YOLONetV1, self).__init__()

        self.conv1 = Conv2d(channel, 64, 7, stride=2, padding=3)
        self.pool1 = AvgPool2d(2, stride=2)

        self.conv2 = Conv2d(64, 192, 3, stride=1, padding=1)
        self.pool2 = AvgPool2d(2, stride=2)

        self.conv3 = Conv2d(192, 128, 1)
        self.conv4 = Conv2d(128, 256, 3, padding=1)
        self.conv5 = Conv2d(256, 256, 1)
        self.conv6 = Conv2d(256, 512, 3, padding=1)

        self.pool6 = AvgPool2d(2, stride=2)

        self.conv7 = Conv2d(512, 256, 1)
        self.conv8 = Conv2d(256, 512, 3, padding=1)
        self.conv9 = Conv2d(512, 256, 1)
        self.conv10 = Conv2d(256, 512, 3, padding=1)
        self.conv11 = Conv2d(512, 256, 1)
        self.conv12 = Conv2d(256, 512, 3, padding=1)
        self.conv13 = Conv2d(512, 256, 1)
        self.conv14 = Conv2d(256, 512, 3, padding=1)
        self.conv15 = Conv2d(512, 512, 1)
        self.conv16 = Conv2d(512, 1024, 3, padding=1)

        self.pool16 = AvgPool2d(2, stride=2)

        self.conv17 = Conv2d(1024, 512, 1)
        self.conv18 = Conv2d(512, 1024, 3, padding=1)
        self.conv19 = Conv2d(1024, 512, 1)
        self.conv20 = Conv2d(512, 1024, 3, padding=1)
        self.conv21 = Conv2d(1024, 1024, 3, padding=1)
        self.conv22 = Conv2d(1024, 1024, 3, stride=2, padding=1)
        self.conv23 = Conv2d(1024, 1024, 3, padding=1)
        self.conv24 = Conv2d(1024, 1024, 3, padding=1)

        self.fc25 = Linear(1024 * 7 * 7, 4096)

        self.fc26 = Linear(4096, 1470)

        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))

        x = self.pool1(x)

        x = self.relu(self.conv2(x))

        x = self.pool2(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))

        x = self.pool6(x)

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x))
        x = self.relu(self.conv16(x))

        x = self.pool16(x)

        x = self.relu(self.conv17(x))
        x = self.relu(self.conv18(x))
        x = self.relu(self.conv19(x))
        x = self.relu(self.conv20(x))
        x = self.relu(self.conv21(x))
        x = self.relu(self.conv22(x))
        x = self.relu(self.conv23(x))
        x = self.relu(self.conv24(x))

        x = F.batch_flatten(x)

        x = self.relu(self.fc25(x))

        x = self.fc26(x)

        return x


def yolo_v1():
  model = YOLONetV1()
  return model
