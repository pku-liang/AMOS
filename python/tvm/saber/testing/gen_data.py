import tvm
import json
from tvm import saber
import numpy as np
from tvm.saber import space
from tvm.saber import offline
from tvm.saber import model
from tvm.saber.device import measure
from tvm.saber.distribution import conv2d, gemm, general



def generate_random_profiled_data(
    group, kernel_ctx, evaluate_ctx, num_rounds,
    num_kernels_explore=10, verbose=False,
    filename="data.log"):
    """
    group: ShapeGroup
    kernel_ctx: KernelContext
    num_rounds: int
    ------
    Returns:
    """
    fout = open(filename, "a")
    print(f"Find kernel (type={kernel_ctx.kernel_type}) for group id={group.group_id} (totally {len(group.shapes)} shapes)", flush=True)
    print(f"The space size is {kernel_ctx.space.size()}", flush=True)
    # some constants
    num_iterations = (num_rounds + num_kernels_explore - 1) // num_kernels_explore
    print(f"Searching for {num_iterations} iterations, {num_kernels_explore} items per iteration.", flush=True)
    # prepare temp space
    unexplored_space = kernel_ctx.space

    for it in range(num_iterations):
        print(f"Iteration {it + 1}: ", flush=True, end="")
        # steps:
        # random sample num_kernels_explore points from space
        # measure them using the model
        # send top num_kernels_explore to evaluate
        # use the results to update model
        kernel_configs = unexplored_space.random(batch=num_kernels_explore)
        print(f"#explore: {len(kernel_configs)} ", flush=True, end="")

        print(flush=True)
        full_kernel_configs = [
                (kernel_ctx.kernel_type,
                dict(list(x.items()) + list(kernel_ctx.static_params.items())))
                for x in kernel_configs
        ]

        build_results = measure.local_builder_build_shape_oblivious(
            full_kernel_configs,
            kernel_ctx.build_timeout,
            kernel_ctx.target,
            kernel_ctx.target_host,
            kernel_ctx.build_parallel,
            kernel_ctx.build_func,
            kernel_ctx.verbose
        )
        print(f"#build: {np.sum([1 if x.error_no == 0 else 0 for x in build_results])} ", flush=True, end="")
        if verbose:
            print(build_results, flush=True)

        full_evaluate_configs = [
            (x[0], x[1], group.shapes) for x in full_kernel_configs
        ]
        evaluate_results = measure.local_run(
            full_evaluate_configs,
            build_results,
            kernel_ctx.target,
            evaluate_ctx.dev_id,
            evaluate_ctx.timeout,
            evaluate_ctx.number,
            evaluate_ctx.repeat,
            evaluate_ctx.min_repeat_ms,
            evaluate_ctx.cooldown_interval,
            evaluate_ctx.enable_cpu_cache_flush,
            evaluate_ctx.verbose
        )

        results_lst = [
                    [float(y) if isinstance(y, float) else float(y.value) for y in x.costs] for x in evaluate_results]
        print(f"#run: {np.sum([1 if x.error_no == 0 else 0 for x in evaluate_results])} ", flush=True, end="")
        if verbose:
            print(evaluate_results, flush=True)
        else:
            print(flush=True)

        if results_lst:
            results_matrix = results_lst
            results_matrix = np.array(results_matrix)
            print(results_matrix)
            for i in range(results_matrix.shape[0]):
                for j in range(results_matrix.shape[1]):
                    print(
                        dict(list(kernel_configs[i].items()) + list(kernel_ctx.static_params.items())),
                        "-",
                        group.shapes[j],
                        "-",
                        results_matrix[i][j],
                        file=fout,
                        flush=True)

    fout.close()


yolo_shapes_b1 = [
    # batch, in_channel, height, width, out_channel, _, k_h, k_w, _, stride, padding, dilation, groups = shape
    # yolo
    # (1, 3, 448, 448, 64, 3, 7, 7, 1, 2, 3, 1, 1),  # conv1  0
    (1, 64, 112, 112, 192, 64, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
    # (1, 192, 56, 56, 128, 192, 1, 1, 1, 1, 0, 1, 1),  # conv3   2
    (1, 128, 56, 56, 256, 128, 3, 3, 1, 1, 1, 1, 1),  # conv4   3
    # (1, 256, 56, 56, 256, 256, 1, 1, 1, 1, 0, 1, 1),  # conv5   4
    (1, 256, 56, 56, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv6   5
    # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv7   6
    (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv8   7
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv9
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv10
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv11
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv12
    # # (1, 512, 28, 28, 256, 512, 1, 1, 1, 1, 0, 1, 1),  # conv13
    # # (1, 256, 28, 28, 512, 256, 3, 3, 1, 1, 1, 1, 1),  # conv14
    # (1, 512, 28, 28, 512, 512, 1, 1, 1, 1, 0, 1, 1),  # conv15      8
    (1, 512, 28, 28, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv16     9
    # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv17    10
    (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv18     11
    # # (1, 1024, 14, 14, 512, 1024, 1, 1, 1, 1, 0, 1, 1),  # conv19
    # # (1, 512, 14, 14, 1024, 512, 3, 3, 1, 1, 1, 1, 1),  # conv20
    (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv21   12
    # (1, 1024, 14, 14, 1024, 1024, 3, 3, 1, 2, 1, 1, 1),  # conv22   13
    (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv23     14
    # (1, 1024, 7, 7, 1024, 1024, 3, 3, 1, 1, 1, 1, 1),  # conv24
]


test_shapes = [
    (1, 512, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1, 1),  # conv2   1
]


def generate_data(group, filename="data.log"):
    tb_space = space.ThreadblockProblemSizeSpace()
    wp_space = space.WarpProblemSizeSpace()
    # it_space = space.TensorCoreSizeSpace()
    it_space = space.InstructionProblemSizeSpace()
    
    def valid_func(x):
        tb = x["threadblock_problem_size"]
        wp = x["warp_problem_size"]
        it = x["instruction_problem_size"]
        return (
            tb[0] >= wp[0] >= it[0] and
            tb[1] >= wp[1] >= it[1] and
            tb[2] >= wp[2] >= it[2] and
            wp[0] * wp[1] % 128 == 0 and
            it[0] * it[1] % 32 == 0
        )
    
    custom_space = space.JoinedSpace(valid_func)
    custom_space.add_subspace("threadblock_problem_size", tb_space)
    custom_space.add_subspace("warp_problem_size",wp_space)
    custom_space.add_subspace("instruction_problem_size", it_space)

    kernel_ctx = offline.KernelContext(
        "conv2d",
        "cuda",
        "general",
        "llvm",
        "default",
        {
            "in_dtype": "float32",
            "out_dtype": "float32",
            "arch": "ampere",
            "code": "sm80",
            "stride": 1,
            "padding": 1
        },
        custom_space,
        build_timeout = 10,
        build_parallel = 20,
        verbose = False
    )
    evaluate_ctx = offline.EvaluationContext(
        timeout=10,
        verbose=False,
        number=100,
        repeat=1,
        min_repeat_ms=500,
        cooldown_interval=1,
        enable_cpu_cache_flush=0,
        dev_id=0
    )
    num_rounds = 100

    generate_random_profiled_data(
        group,
        kernel_ctx,
        evaluate_ctx,
        num_rounds,
        verbose=True,
        filename=filename,
        num_kernels_explore=kernel_ctx.build_parallel
    )



if __name__ == "__main__":
    filename = "conv_op_config_longtail.txt"
    conv_shapes = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            in_channels = obj["param"][0]
            out_channels = obj["param"][1]
            kernel_size = obj["param"][2]
            strides = obj["param"][3]
            padding = obj["param"][4]
            bias = obj["param"][5]
            groups = obj["param"][6]
            use_fp16 = obj["param"][7]

            if padding[0] != 1 or groups != 1:
                continue

            count = obj["count"]
            shapes = obj["shapes"]
            for shape in shapes:
                N, C, H, W = shape
                conv_shape = conv2d.Conv2dParams(
                    N, C, H, W, out_channels,
                    *kernel_size,
                    *strides,
                    *padding,
                    1, 1,
                    groups
                )
                conv_shapes.append(conv_shape)
    group = general.ShapeGroup(0, conv_shapes)
    generate_data(group, "conv_mm_profile_general_fp32fp32.log")

    # shapes = []
    # for ss in test_shapes:
    #     N, C, H, W, K, _, R, S, _, stride, padding, dilation, groups = ss
    #     shape = conv2d.Conv2dParams(
    #         N, C, H, W, K, R, S,
    #         stride, stride,
    #         padding, padding,
    #         dilation, dilation,
    #         groups
    #     )
    #     shapes.append(shape)
    # group = general.ShapeGroup(0, shapes)