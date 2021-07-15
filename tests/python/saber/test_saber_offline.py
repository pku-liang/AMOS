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
    num_rounds = 10
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


yolo_shapes_b1 = [
    # batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
    # yolo
    (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    (1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    (1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1),  # conv5   4
    (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv7   6
    (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv9
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv10
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv11
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv12
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv13
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv14
    (1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1),  # conv15      8
    (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv17    10
    (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    # # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv19
    # # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv20
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
    # (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv24
]


@register_test
def test3():
    shapes = []
    for ss in yolo_shapes_b1:
        N, C, H, W, K, _, R, S, _, stride, padding, dilation, groups = ss
        shape = saber.distribution.conv2d.Conv2dParams(
            N, C, H, W, K, R, S,
            stride, stride,
            padding, padding,
            dilation, dilation,
            groups
        )
        shapes.append(shape)
    cluster = saber.analysis.kmeans.FaissKMeans(n_clusters=4)
    shape_groups = saber.distribution.general.group_shapes(
        cluster, shapes, [])
    for group in shape_groups:
        print("Group", group.group_id)
        for ss in group.shapes:
            print("\t", ss)
    


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
