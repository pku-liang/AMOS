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
    gemm = saber.GemmMaliGeneral()
    func = gemm.compile(dump=True)
    cost = gemm.evaluate(func, 512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")
    cost = gemm.evaluate(func, 16, 1024, 32, new_process=False)
    print("Cost is", cost, "ms")


@register_test
def test2():
    gemm = saber.GemmMaliGeneral()
    cost = gemm.try_with(512, 64, 1024, new_process=True)
    print("Cost is", cost, "ms")
    cost = gemm.try_with(512, 64, 1024, new_process=False)
    print("Cost is", cost, "ms")


@register_test
def test3():
    gemm = saber.GemmMaliGeneral(
        threadblock_problem_size=[32, 32, 32],
        warp_problem_size=[8, 8, 16],
        instruction_problem_size=[4, 4, 8])
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
        assert isinstance(params, saber.MaliParams)
        gemm = saber.GemmMaliGeneral(
            in_dtype="float32",
            out_dtype="float32",
            threadblock_problem_size=params.threadblock_problem_size[0],
            warp_problem_size=params.warp_problem_size[0],
            instruction_problem_size=params.instruction_problem_size[0],
            split_K=params.split_K[0][0])
        shapes = [
            [128, 128, 128],
        ]
        targets = [
            0.13752,
        ]
        relative_lst = []
        for shape, target in zip(shapes, targets):
            cost = gemm.try_with(*shape, new_process=True)
            relative = target / cost
            relative_lst.append(relative)
        return 1 / geomean(relative_lst)

    generator = saber.MaliDeviceGeneralGenerator(arch="g76")
    saber.serial_minimize(
        device_impl,
        generator,
        saber.MeasureOptions(
            target="opencl",
            target_host="llvm -mtriple=aarch64-linux-android",
            timeout=10, number=10,
            min_repeat_ms=80,
            build_func="ndk",
            key="android",
            host="0.0.0.0",
            port=9190,
            cooldown_interval=5),
        trials=200
        )
    


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
