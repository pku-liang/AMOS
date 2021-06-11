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
    shapes = saber.distribution.conv2d.get_conv_shapes("conv_op_config_longtail.txt")
    cluster = saber.analysis.kmeans.FaissKMeans(n_clusters=10)
    shape_class = saber.distribution.conv2d.ConvFullParams
    shapes, counts = saber.distribution.conv2d.make_conv_full_param_lst_from_param_groups(shapes)
    groups = saber.distribution.general.group_shapes(
        cluster, shapes, counts)
    print(groups[0].shapes)


@register_test
def test2():
    shape1 = saber.distribution.gemm.GEMMParams(256, 256, 256)
    shape2 = saber.distribution.gemm.GEMMParams(512, 256, 256)
    shape3 = saber.distribution.gemm.GEMMParams(256, 512, 256)
    group = saber.distribution.general.ShapeGroup(0, [
        shape1, shape2, shape3
    ])
    # tb_space = saber.space.SubSpace([
    #     [32, 32, 32]
    # ])
    # wp_space = saber.space.SubSpace([
    #     [32, 32, 8]
    # ])
    # it_space = saber.space.SubSpace([
    #     [4, 8, 8],
    #     [8, 4, 8]
    # ])
    tb_space = saber.space.ThreadblockProblemSizeSpace()
    wp_space = saber.space.WarpProblemSizeSpace()
    it_space = saber.space.InstructionProblemSizeSpace()
    
    def valid_func(x):
        tb = x["threadblock_problem_size"]
        wp = x["warp_problem_size"]
        it = x["instruction_problem_size"]
        return (
            tb[0] >= wp[0] >= it[0] and
            tb[1] >= wp[1] >= it[1] and
            tb[2] >= wp[2] >= it[2]
        )
    
    space = saber.space.JoinedSpace(valid_func)
    space.add_subspace("threadblock_problem_size", tb_space)
    space.add_subspace("warp_problem_size",wp_space)
    space.add_subspace("instruction_problem_size", it_space)

    kernel_ctx = saber.offline.KernelContext(
        "gemm",
        "cuda",
        "general",
        "llvm",
        "default",
        {
            "in_dtype": "float32",
            "out_dtype": "float32",
            "arch": "ampere",
            "code": "sm80"
        },
        space,
        build_timeout = 10,
        build_parallel = 1,
        verbose = False
    )
    evaluate_ctx = saber.offline.EvaluationContext(
        timeout=10,
        verbose=False,
        number=100,
        repeat=1,
        min_repeat_ms=500,
        cooldown_interval=1,
        enable_cpu_cache_flush=0,
        dev_id=0
    )
    num_rounds = 1000
    perf_model = saber.model.PerfModel(saber.model.SimpleMLPCUDAGemmGeneralPerfModel)
    # perf_model = saber.model.PerfModel(saber.model.RandomOpPerfModel)
    result = saber.offline.train_for_one_group(
        group,
        kernel_ctx,
        evaluate_ctx,
        num_rounds,
        perf_model,
        verbose=True
    )

    kernel = saber.offline.CompiledKernel.make_compiled_kernel_from_result_kernel_context(result, "test")
    print(kernel)
    cost = kernel.evaluate(
        shape1,
        saber.MeasureOptions(
            target="cuda", number=100,
            min_repeat_ms=500),
        False
    )
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
