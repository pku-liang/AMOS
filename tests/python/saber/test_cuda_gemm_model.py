from os import confstr
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
    perf_model = saber.model.SimpleAnalyticCUDAGemmGeneralPerfModel("just-for-test")
    tb = [128, 128, 8]
    wp = [32, 64, 8]
    it = [4, 8, 8]
    configs = [
        [[128, 128, 8], [32, 64, 8], [4, 8, 8]],
        [[64, 256, 8], [32, 64, 8], [4, 8, 8]],
        [[64, 128, 8], [32, 32, 8], [4, 8, 8]],
        [[64, 128, 8], [32, 64, 8], [4, 8, 8]],
        [[64, 64, 8], [32, 64, 8], [4, 8, 8]],
    ]
    first = True
    def run(tb, wp, it):
        nonlocal first
        config = dict(
            threadblock_problem_size=tb,
            warp_problem_size=wp,
            instruction_problem_size=it,
        )
        gemm = saber.GemmCUDAGeneral(**config)
        DIMS = (256, 512, 1024)
        SHAPES = [(M, N, K) for M in DIMS for N in DIMS for K in DIMS]
        with open(f"./saber-cuda-general-col-row-gemm-{tb}-{wp}-{it}-model-hongv0.csv", "w") as fp:
            print("M,N,K,pred(gflops),cost(gflops)", file=fp, flush=True)
            for M, N, K in SHAPES:
                shape = saber.distribution.gemm.GEMMParams(M, N, K)
                pred = perf_model.predict([(config, shape)])[0]
                cost = shape.gflop() / (gemm.try_with(M, N, K, new_process=first) * 1e-3 + 1e-10) 
                first = False
                print(f"{M},{N},{K},{pred},{cost}", file=fp, flush=True)
    
    for cfg in configs:
        run(*cfg)


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
