import numpy as np
import tvm
from tvm.tensor_graph.nn.layers import Layer, Conv2d
from tvm.tensor_graph.nn.functional import conv2d_nchw, reshape
from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph

def Sigmoid(X):
    def _inner_sigmoid(shape0, shape1, shape2, shape3, inputX, requires_grad=False):
        return compute([shape0, shape1, shape2, shape3],
                lambda i, j, k, l: tvm.te.sigmoid(inputX[i, j, k, l]),
                name="sigmoid",
                requires_grad=requires_grad)
    return GraphOp(X.shape, [], [X], _inner_sigmoid, name="sigmoid")

class WeightNet(Layer):
    # https://github.com/megvii-model/WeightNet/blob/669b5f4c0c46fd30cd0fedf5e5a63161e9e94bcc/weightnet.py

    def __init__(self, inp=448, oup=50176, ksize=1, stride=1, dtype="float32", out_dtype="float32"):
        super().__init__()

        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        inp_gap = inp // 16
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = Conv2d(inp_gap, self.M*oup, kernel_size=ksize, stride=stride, padding=0, groups=1, bias=True, dtype=dtype, out_dtype=out_dtype)
        self.sigmoid = Sigmoid
        self.wn_fc2 = Conv2d(self.M*oup, oup*inp*ksize*ksize, kernel_size=1, stride=1, padding=0, groups=self.G*oup, bias=False, dtype=dtype, out_dtype=out_dtype)


    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        # [1, 22478848, 1, 1] -> [50176, 448, 1, 1]
        x_w = reshape(x_w, [self.oup, self.inp, self.ksize, self.ksize])
        x = conv2d_nchw(x, weight=x_w, stride=self.stride, padding=self.pad)
        return x

if __name__ == "__main__":
    batch_size = 1 # Only target batch_size = 1

    dtype = "float16"
    model = WeightNet(dtype=dtype, out_dtype=dtype)
    model.eval()

    x = GraphTensor([batch_size, 448, 1, 1], dtype, name="x", requires_grad=False)
    x_gap = GraphTensor([batch_size, 28, 1, 1], dtype, name="x_gap", requires_grad=False)
    
    out = model(x, x_gap)
    print(out)
