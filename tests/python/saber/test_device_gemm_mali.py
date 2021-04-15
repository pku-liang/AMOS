import tvm
import os
import math
import time
import tempfile
from tvm.contrib import tar, ndk
import numpy as np
from tvm import saber
from collections import OrderedDict
from tvm import auto_tensorize as at


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


gemm_shapes = [
    # M, N, K
    (128, 128, 128),
    # (128, 256, 128),
    # (128, 128, 256),
    # (128, 256, 256),
    # (256, 128, 128),
    # (256, 256, 128),
    # (256, 128, 256),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    # (2048, 2048, 2048),
    # (4096, 4096, 4096)
]


@register_test
def test1():
    gemm = saber.GemmMaliGeneral(
        # threadblock_problem_size=[64, 16, 8],
        # warp_problem_size=[32, 8, 8],
        # instruction_problem_size=[4, 1, 4],

        # threadblock_problem_size=[32, 128, 4],
        # warp_problem_size=[32, 128, 4],
        # instruction_problem_size=[4, 4, 4],

        threadblock_problem_size=[32, 32, 32],
        warp_problem_size=[32, 32, 32],
        instruction_problem_size=[4, 4, 4],

        # threadblock_problem_size=[128, 32, 16],
        # warp_problem_size=[64, 16, 16],
        # instruction_problem_size=[4, 1, 4],
        code="g71",
        tag="single_buffer"
    )
    func = gemm.compile(dump=False)
    print(func.imported_modules[0].get_source())
    measure_opt_hikey = saber.MeasureOptions(
                target="opencl",
                target_host="llvm -mtriple=aarch64-linux-gnu",
                timeout=10, number=10,
                min_repeat_ms=80,
                build_func="default",
                key="hikey960",
                host="0.0.0.0",
                port=9190,
                cooldown_interval=5)
    measure_opt_android = saber.MeasureOptions(
                target="opencl",
                target_host="llvm -mtriple=aarch64-linux-android",
                timeout=10, number=10,
                min_repeat_ms=80,
                build_func="ndk",
                key="android",
                host="0.0.0.0",
                port=9190,
                cooldown_interval=5)
    measure_opt = measure_opt_hikey
    cost = gemm.evaluate(func, 256, 256, 256, new_process=False,
                            measure_opt=measure_opt
                        )
    print("Cost is", cost, "ms")
    beg = 0
    end = 4
    print("M,N,K,cost(ms)")
    for i, shape in enumerate(gemm_shapes):
        M, N, K = shape
        if i >= beg and i < end:
            cost = gemm.evaluate(func, M, N, K, new_process=False, measure_opt=measure_opt)
            time.sleep(3)
        else:
            cost = -1
        # print("%d,%d,%d,%f" % (M, N, K, cost))
        print(cost)

    M, N, K = 128, 128, 128

    A = np.random.uniform(-1, 1, [M, K]).astype("float32")
    B = np.random.uniform(-1, 1, [N, K]).astype("float32")
    C = np.zeros([M, N]).astype("float32")

    key = measure_opt.key
    host = measure_opt.host
    port = measure_opt.port
    priority = measure_opt.priority
    timeout = measure_opt.timeout
    from tvm import auto_scheduler
    remote = auto_scheduler.utils.request_remote(
        key, host, port, priority, timeout)
    ctx = remote.context(measure_opt.target)
    A_tvm = tvm.nd.array(A, ctx)
    B_tvm = tvm.nd.array(B, ctx)
    C_tvm = tvm.nd.array(C, ctx)

    build_func = measure_opt.build_func
    if build_func == "default":
        fd, lib = tempfile.mkstemp(prefix="tmp_func", suffix=".tar")
        os.close(fd)
        func.export_library(lib, tar.tar)
    elif build_func == "ndk":
        fd, lib = tempfile.mkstemp(prefix="tmp_func", suffix=".so")
        os.close(fd)
        func.export_library(lib, ndk.create_shared)
    else:
        raise ValueError()
    remote.upload(lib)
    func = remote.load_module(os.path.split(lib)[-1])
    os.unlink(lib)
    gemm.calculate(func, A_tvm, B_tvm, C_tvm)

    import torch
    A_torch = torch.tensor(A).cuda()
    B_torch = torch.tensor(B).cuda()
    C_torch = torch.mm(A_torch, B_torch.permute(1, 0))
    from tvm import testing
    testing.assert_allclose(C_torch.cpu().numpy(), C_tvm.asnumpy(), atol=1e-1, rtol=1e-1)
    print("Compute result correct!")

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
        threadblock_problem_size=[16, 16, 64],
        warp_problem_size=[16, 16, 64],
        instruction_problem_size=[4, 4, 2])
    cost = gemm.try_with(1024, 16, 128, new_process=True, dump=True)
    print("Cost is", cost, "ms")
    # cost = gemm.try_with(512, 64, 1024, new_process=False)
    # print("Cost is", cost, "ms")
    # cost = gemm.try_with(2, 64, 1024, new_process=False)
    # print("Cost is", cost, "ms")
    # cost = gemm.try_with(16, 512, 128, new_process=False)
    # print("Cost is", cost, "ms")


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
            [256, 256, 256],
        ]
        # targets = [
        #     0.2521,
        # ]
        # relative_lst = []
        # for shape, target in zip(shapes, targets):
        #     cost = gemm.try_with(*shape, new_process=True)
        #     relative = target / cost
        #     relative_lst.append(relative)
        # return 1 / geomean(relative_lst)
        cost = gemm.try_with(*shapes[0], new_process=True)
        return cost

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
        trials=1000
        )


@register_test
def test5():
    def compile_impl(params):
        assert isinstance(params, saber.MaliParams)
        gemm = saber.GemmMaliGeneral(
            in_dtype="float32",
            out_dtype="float32",
            threadblock_problem_size=params.threadblock_problem_size[0],
            warp_problem_size=params.warp_problem_size[0],
            instruction_problem_size=params.instruction_problem_size[0],
            split_K=params.split_K[0][0])
        return gemm.expose_compile_context()

    def evaluate_impl(params):
        assert isinstance(params, saber.MaliParams)
        gemm = saber.GemmMaliGeneral(
            in_dtype="float32",
            out_dtype="float32",
            threadblock_problem_size=params.threadblock_problem_size[0],
            warp_problem_size=params.warp_problem_size[0],
            instruction_problem_size=params.instruction_problem_size[0],
            split_K=params.split_K[0][0])
        shapes = [
            [256, 256, 256],
        ]
        
        tensor_lst = []
        var_value_lst = []
        for shape in shapes:
            args, vars = gemm.expose_evaluate_context(*shape)
            tensor_lst.append(args)
            var_value_lst.append(vars)
        return tensor_lst, var_value_lst

    targets = [
        1
    ]

    def relative_perf_geo(lst):
        rel = []
        for cost, target in zip(lst, targets):
            rel.append(target / cost)
        return geomean(rel)

    class Checker(object):
        def check(self, *args, **kwargs):
            return True

    generator = saber.MaliDeviceGeneralGenerator(arch="g76")
    saber.parallel_maximize(
        compile_impl,
        evaluate_impl,
        relative_perf_geo,
        [generator],
        saber.MeasureOptions(
            use_rpc=True,
            target="opencl",
            target_host="llvm -mtriple=aarch64-linux-gnu",
            timeout=10, number=10,
            min_repeat_ms=80,
            build_func="default",
            key="hikey960",
            host="0.0.0.0",
            port=9190,
            cooldown_interval=5),
        # at.search.MaliProgramChecker(arch="g76"),
        Checker(),
        iterations=1000,
        verbose=True,
        build_parallel=4,
        report_period=1
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
