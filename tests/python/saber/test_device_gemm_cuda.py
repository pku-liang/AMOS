import tvm
import math
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
    cost = gemm.evaluate(func, 16, 1024, 32, new_process=False)
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
        split_K=4)
    cost = gemm.try_with(512, 64, 1024, new_process=True)
    print("Cost is", cost, "ms")
    cost = gemm.try_with(512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")
    cost = gemm.try_with(2, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")
    cost = gemm.try_with(16, 512, 128, new_process=False)
    print("Cost is", cost, "ms")


def geomean(lst):
    assert len(lst) > 0
    val = 1
    for v in lst:
        val *= v
    return math.pow(val, 1/(len(lst)))


@register_test
def test4():
    def device_impl(params):
        assert isinstance(params, saber.CUDAParams)
        gemm = saber.GemmCUDATensorCore(
            in_dtype="int8",
            out_dtype="int32",
            threadblock_problem_size=params.threadblock_problem_size[0],
            warp_problem_size=params.warp_problem_size[0],
            tensorcore_problem_size=params.instruction_problem_size[0],
            split_K=params.split_K[0][0])
        shapes = [
            [16, 512, 128],
            [1024, 16, 256],
            [256, 1024, 256],
            [512, 256, 16],
            [1024, 1024, 1024]
        ]
        targets = [
            0.025,
            0.013,
            0.029,
            0.018,
            0.103
        ]
        relative_lst = []
        for shape, target in zip(shapes, targets):
            cost = gemm.try_with(*shape, new_process=True)
            relative = target / cost
            relative_lst.append(relative)
        return 1 / geomean(relative_lst)

    generator = saber.CUDADeviceTensorCoreGenerator([16, 16, 16], arch=80)
    saber.serial_minimize(
        device_impl,
        generator,
        saber.MeasureOptions(target="cuda", number=10, min_repeat_ms=600),
        trials=200
    )


@register_test
def test5():
    gemm = saber.GemmCUDAGeneral(
        threadblock_problem_size=[128, 128, 8],
        warp_problem_size=[128, 128, 8],
        instruction_problem_size=[8, 8, 8])
    first = True
    for m in (128, 256, 512):
        for n in (128, 256, 512):
            for k in (128, 256):
                cost = gemm.try_with(m, n, k, new_process=first)
                first = False
                print(f"Cost of ({m}, {n}, {k}) is {cost} ms")


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
