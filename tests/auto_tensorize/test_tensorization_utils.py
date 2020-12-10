from tvm import auto_tensorize as at
from tvm.auto_tensorize.tensorization_phases.utils import *


def test_split():
    lst = any_factor_split(1024, 4)
    for v in lst:
        print(v)


def test_remap():
    lst = any_factor_split(112, 2)
    lst, fmap, dim, sum_val = remap_factors(lst)
    for factors in lst:
        print(factors, [fmap[x] for x in factors])
    print(dim)
    print(sum_val)


def test_directions():
    ret = get_directions(3)
    for v in ret:
        print(v)


def test_bi_product():
    ret = bi_product(7)
    for v in ret:
        print(v)
    print(len(ret))


def test_partial_directions():
    ret = get_partial_directions(3)
    for v in ret:
        print(v)
    print(len(ret))


def test_softmax():
    ret = softmax([1, 2, 3])
    print(ret)
    ret = softmax([0])
    print(ret)


if __name__ == "__main__":
    test_split()
    test_remap()
    test_directions()
    test_bi_product()
    test_partial_directions()
    test_softmax()
