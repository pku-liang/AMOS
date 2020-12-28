from tvm.tensor_graph.nn.layers import Layer, Conv2d, CapsuleConv2d
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, GraphNode
import tvm


def concat_eight_vector_lastdim(cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7):
    batch, channel, h, w = cap0.shape
    eight = 8
    def _inner_cat(batch, channel, h, w, eight, cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7, requires_grad=True):
        return compute([batch, channel, h, w, eight],
                lambda i, j, p, q, k:
                    tvm.te.if_then_else(k == 0, cap0[i, j, p, q],
                        tvm.te.if_then_else(k == 1, cap1[i, j, p, q],
                            tvm.te.if_then_else(k == 2, cap2[i, j, p, q],
                                tvm.te.if_then_else(k == 3, cap3[i, j, p, q],
                                    tvm.te.if_then_else(k == 4, cap4[i, j, p, q],
                                        tvm.te.if_then_else(k == 5, cap5[i, j, p, q], 
                                            tvm.te.if_then_else(k == 6, cap6[i, j, p, q],
                                                cap7[i, j, p, q]))))))),
                name="concat",
                tag="concat",
                requires_grad=requires_grad)
    return GraphOp([batch, channel, h, w, eight], [], 
                    [cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7],
                    _inner_cat, name="concat")

class SeparatedConv(Layer):
    def __init__(self, in_channels=256, out_channels=32):
        super(SeparatedConv, self).__init__()
        self.conv0 = Conv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0)
        self.conv1 = Conv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0)
        self.conv2 = Conv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0)
        self.conv3 = Conv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0)
        self.conv4 = Conv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0)
        self.conv5 = Conv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0)
        self.conv6 = Conv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0)
        self.conv7 = Conv2d(in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0)

    def forward(self, x):
        conved0 = self.conv0(x)
        conved1 = self.conv1(x)
        conved2 = self.conv2(x)
        conved3 = self.conv3(x)
        conved4 = self.conv4(x)
        conved5 = self.conv5(x)
        conved6 = self.conv6(x)
        conved7 = self.conv7(x)
        result = concat_eight_vector_lastdim(conved0, conved1, conved2, conved3, conved4, conved5, conved6, conved7)
        return result

class CapsConv(Layer):
    def __init__(self, in_channels=256, out_channels=32, num_capsules=8):
        super(CapsConv, self).__init__()
        self.capsules = CapsuleConv2d(
                          in_channel=in_channels, out_channel=out_channels, kernel_size=9, stride=2, padding=0, 
                          num_caps=num_capsules)
    
    def forward(self, x):
        result = self.capsules(x)
        return result

def get_model(separated):
    if separated == True:
        return SeparatedConv()
    else:
        return CapsConv()
