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
    gemm = saber.GemmCUDATensorCore()
    func = gemm.compile()
    cost = gemm.evaluate(func, 512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")
    cost = gemm.evaluate(func, 512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")


@register_test
def test2():
    gemm = saber.GemmCUDATensorCore()
    cost = gemm.try_with(512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")
    cost = gemm.try_with(512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")


@register_test
def test3():
    gemm = saber.GemmCUDATensorCore(
        threadblock_problem_size=[16, 16, 128],
        warp_problem_size=[16, 16, 32],
        tensorcore_problem_size=[16, 16, 16],
        split_K=2)
    cost = gemm.try_with(512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")
    cost = gemm.try_with(512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")


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
