from . import _ffi_api
import tvm
from functools import reduce


def index(name="tmp"):
    return tvm.tir.Var(name, "int32")


def multi_index(num, name="tmp"):
    return [tvm.tir.Var(name + str(i)) for i in range(num)]


def multi_reduce_axis(extents, name="tmp"):
    return [tvm.te.reduce_axis(
        [0, extents[i]], name + str(i)) for i in range(len(extents))]


def return_conv2d_vars(N, K, H, W, C, R, S):
    return [N, K, H, W, C, R, S]


def ceil(a, b):
    return (a + b - 1) // b


def reduce_mul(lst):
    return reduce(lambda i, j: i * j, lst, 1)


def reduce_add(lst):
    return reduce(lambda i, j: i + j, lst, 0)


def tile_axes_outer_to_inner(sch, op, axis, factors):
    ret = []
    for f in factors:
        outer, axis = sch[op].split(axis, factor=f)
        ret.append(outer)
    ret.append(axis)
    return ret


def array_view(array, shape):
    return _ffi_api.NDArrayView(
        array, [int(x) for x in shape], tvm.runtime.String(array.dtype))