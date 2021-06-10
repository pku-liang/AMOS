import numpy as np
import json
from collections import namedtuple
from functools import reduce


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

    def to_flatten_tuple(self):
        return (
            self.in_channels,
            self.out_channels,
            *self.kernel_size,
            *self.strides,
            *self.padding,
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


class ConvFullParams(object):
    def __init__(self, batch=None, H=None, W=None, in_channels=None, out_channels=None, kernel_size=None, strides=(1, 1), padding=(0, 0), bias=True, groups=1, use_fp16=False, dilation=1):
        self.batch = batch
        self.H = H
        self.W = W
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.use_fp16 = use_fp16
        self.dilation = dilation

    def gflop(self):
        return (
            self.batch * self.H * self.W
            * self.in_channels * self.out_channels
            * self.kernel_size[0] * self.kernel_size[1]
            / self.strides[0] / self.strides[1]
            / self.groups * 2 / 1e9
        )

    def valid(self):
        return (
            self.batch is not None and
            self.H is not None and
            self.W is not None and
            self.in_channels is not None and
            self.out_channels is not None and
            self.kernel_size is not None and
            self.strides is not None and
            self.padding is not None and
            self.bias is not None and
            self.groups is not None and
            self.use_fp16 is not None and
            self.dilation is not None
        )

    def to_tuple(self):
        return (
            self.batch,
            self.H,
            self.W,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.strides,
            self.padding,
            self.bias,
            self.groups,
            self.use_fp16,
            self.dilation
        )

    def to_flatten_tuple(self):
        return (
            self.batch,
            self.H,
            self.W,
            self.in_channels,
            self.out_channels,
            *self.kernel_size,
            *self.strides,
            *self.padding,
            self.bias,
            self.groups,
            self.use_fp16,
            self.dilation
        )

    @staticmethod
    def from_flatten_tuple(tup):
        return ConvFullParams(
            tup[0],
            tup[1],
            tup[2],
            tup[3],
            tup[4],
            (tup[5], tup[6]),
            (tup[7], tup[8]),
            (tup[9], tup[10]),
            tup[11],
            tup[12],
            tup[13],
            tup[14]
        )

    def to_tuple_key(self):
        tmp = (
            self.batch,
            self.H,
            self.W,
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            self.strides[0],
            self.padding[0],
            self.groups,
            self.dilation
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
        return "ConvFullParams" + str(self.to_tuple())


ConvShape = namedtuple("ConvShape", "batch, channels, height, width")


ConvShapeItem = namedtuple("ConvShapeItem", "count, shapes")


ConvShapePerf = namedtuple("ConvShapeItem", "shape, perf")


def make_conv_full_param(param, input):
    return ConvFullParams(
        input.batch, input.height, input.width,
        param.in_channels, param.out_channels, param.kernel_size,
        param.strides, param.padding, param.bias, param.groups,
        param.use_fp16
    )


def make_conv_full_param_lst(param, item):
    ret = [make_conv_full_param(param, x) for x in item.shapes]
    counts = [item.count for x in item.shapes]
    return ret, counts


def make_conv_full_param_lst_from_param_groups(param_groups):
    full_param_input_lst = []
    count_lst = []
    for (param, shape_item) in param_groups:
        part_param_input_lst, part_count_lst = make_conv_full_param_lst(
            param, shape_item)
        full_param_input_lst.extend(part_param_input_lst)
        count_lst.extend(part_count_lst)
    return full_param_input_lst, count_lst


def get_conv_shapes(filename):
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


def cluster_kernel_params(cluster, info, total_kernels):
    X = np.array([list(map(float, x.to_flatten_tuple())) for x, _ in info])
    X = normalize_params(X)
    cluster.fit(X)
    predicts = cluster.predict(X).squeeze()
    n_clusters = cluster.n_clusters
    param_clusters = [[] for _ in range(n_clusters)]
    cluster_weight = [0 for _ in range(n_clusters)]
    total_weight = 0

    for x, y in  zip(info, predicts):
        y = int(y)
        param_clusters[y].append((x[0], x[1]))
        cluster_weight[y] += len(x[1].shapes) # x[1].count
        total_weight += len(x[1].shapes) # x[1].count
    
    num_kernels = [np.ceil(float(x) / total_weight * total_kernels) for x in cluster_weight]
    excess_kernels = reduce(lambda i, j: i + j, num_kernels, 0) - total_kernels

    sorted_num = np.argsort(num_kernels)
    p = len(sorted_num) - 1
    while p >= 0 and excess_kernels > 0:
        if num_kernels[p] > 1:
            num_kernels[p] -= 1
            excess_kernels -= 1
        p -= 1
        if p < 0:
            p = len(sorted_num) - 1
    if excess_kernels > 0:
        print("Warning: use additional", excess_kernels, "kernels", flush=True)
    
    return cluster, predicts, param_clusters, list(map(int, num_kernels))


def cluster_param_inputs(cluster, full_param_input_lst, count_lst, assigned_kernels):
    """
    cluster: tool such as KMeans
    param_group: [(param, shape_item)]
    assigned_kernels: number of kernels for this group
    """
    X = np.array(
        [list(map(float, x.to_flatten_tuple())) for x in full_param_input_lst])
    X = normalize_params(X)
    cluster.fit(X)
    predicts = cluster.predict(X).squeeze()
    n_clusters = cluster.n_clusters
    param_input_clusters = [[] for _ in range(n_clusters)]
    cluster_weight = [0 for _ in range(n_clusters)]
    total_weight = 0

    for x, y, z in  zip(full_param_input_lst, count_lst, predicts):
        z = int(z)
        param_input_clusters[z].append(x)
        cluster_weight[z] += y
        total_weight += y

    num_kernels = [np.ceil(
        float(x) / total_weight * assigned_kernels) for x in cluster_weight]
    excess_kernels = reduce(lambda i, j: i + j, num_kernels, 0) - assigned_kernels

    sorted_num = np.argsort(num_kernels)
    p = len(sorted_num) - 1
    while p >= 0 and excess_kernels > 0:
        if num_kernels[p] > 1:
            num_kernels[p] -= 1
            excess_kernels -= 1
        p -= 1
        if p < 0:
            p = len(sorted_num) - 1
    if excess_kernels > 0:
        print("Warning: use additional", int(excess_kernels), "kernels", flush=True)
    
    return cluster, predicts, param_input_clusters, list(map(int, num_kernels))



