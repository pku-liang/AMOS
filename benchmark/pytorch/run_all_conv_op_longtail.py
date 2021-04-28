import torch
import numpy as np
import json
from collections import namedtuple
import conv2d


class ConvParams(object):
    def __init__(self, in_channels=None, out_channels=None, kernel_size=None, strides=(1, 1), padding=(0, 0), bias=True, groups=1, use_fp16=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.use_fp16 = use_fp16

    def valid(self):
        return (
            self.in_channels is not None and
            self.out_channels is not None and
            self.kernel_size is not None and
            self.strides is not None and
            self.padding is not None and
            self.bias is not None and
            self.groups is not None and
            self.use_fp16 is not None
        )

    def to_tuple(self):
        return (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.strides,
            self.padding,
            self.bias,
            self.groups,
            self.use_fp16
        )

    def to_tuple_key(self):
        tmp = (
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.strides[0],
            self.padding[0],
            self.groups
        )
        ret = ",".join([str(x) for x in tmp])
        return ret

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.to_tuple() == other.to_tuple()
        else:
            return self.to_tuple() == other

    def __repr__(self):
        return "ConvParams" + str(self.to_tuple())


ConvShape = namedtuple("ConvShape", "batch, channels, height, width")


ConvShapeItem = namedtuple("ConvShapeItem", "count, shapes")


ConvShapePerf = namedtuple("ConvShapeItem", "shape, perf")


def get_conv_shapes(filename="conv_op_config_longtail.txt"):
    ret = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            conv_param = ConvParams(*obj["param"])
            count = obj["count"]
            shapes = obj["shapes"]
            shapes = [ConvShape(*x) for x in shapes]
            item = ConvShapeItem(count, shapes)
            ret.append((conv_param, item))

    return ret


if __name__ == "__main__":
    assert torch.backends.cudnn.is_available()
    torch.backends.cudnn.enabled = True
    print("cudnn: %s" % ("enabled" if torch.backends.cudnn.enabled else "disabled"))
    print("N, C, H, W, K, R, S, stride, padding, dilation, type, cost")
    shapes = get_conv_shapes()
    ret = []
    for shape in shapes:
        dilation = 1
        # bias is not used
        C, K, kernel_size, stride, padding, bias, groups, use_fp16 = shape[0].to_tuple()
        dtype = "FP32" if not use_fp16 else "FP16"
        shape_perf = []
        for ss in shape[1].shapes:
            N, _, H, W = tuple(ss)
            mean_cost = conv2d.conv2d(N, C, H, W, K, *kernel_size, stride, padding, dilation, groups, dtype, allow_tf32=True)
            print(",".join(map(str, [N, C, H, W, K, kernel_size, stride, padding, dilation, groups, dtype, mean_cost])))
            shape_perf.append(ConvShapePerf(ss, mean_cost))
        item = ConvShapeItem(shape[1].count, shape_perf)
        ret.append({"param": shape[0].to_tuple(), "count": item.count, "shape_perf": [{"shape": s, "perf": p} for (s, p) in item.shapes]})
    
    with open("conv_op_perf_longtail.txt", "w") as fout:
        for v in ret:
            string = json.dumps(v)
            fout.write(string + "\n")
