import tvm
import numpy as np
from tvm import saber
from collections import OrderedDict


TEST_CASES = OrderedDict()


def register_test(func):
    name = func.__name__
    prefix = "test"
    assert name[:len(prefix)] == prefix
    try:
        number = int(name[len(prefix):])

        def _inner(*args, **kwargs):
            print(func.__doc__)
            func(*args, **kwargs)
        assert number not in TEST_CASES, "Repeated test case number %d" % number
        TEST_CASES[number] = _inner
    except ValueError as e:
        print(e)
        print("Can't convert to number", name[len(prefix):])


@register_test
def test1():
    shapes = saber.distribution.conv_op_dist.get_conv_shapes()
    cluster = saber.analysis.kmeans.FaissKMeans(n_clusters=4)
    cluster, predicts, param_clusters, num_kernels = saber.distribution.conv_op_dist.cluster_kernel_params(cluster, shapes, 40)
    print(num_kernels)
    with open("conv_op_cluster_predicts.txt", "w") as fout:
        for p in predicts:
            fout.write(str(p) + "," + str(num_kernels[p]))
            fout.write("\n")

    for param_cluster, assigned_kernels in zip(param_clusters, num_kernels):
        sub_cluster = saber.analysis.kmeans.FaissKMeans(n_clusters=assigned_kernels)
        sub_cluster, predicts, param_input_cluster, num_kernels_in_group = \
            saber.distribution.conv_op_dist.cluster_param_inputs(
                sub_cluster,
                param_cluster,
                assigned_kernels
            )
        print(num_kernels_in_group)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", help="test case", type=int, default=1)
    parser.add_argument("--all", help="test all", action="store_true")

    args = parser.parse_args()
    if args.all:
        for k, v in TEST_CASES.items():
            print("############################################")
            print("test", k)
            v()
            print("Pass!")
    else:
        assert args.case in TEST_CASES, "Can't find case %s." % (
            str(args.case))
        case = TEST_CASES[args.case]
        case()
