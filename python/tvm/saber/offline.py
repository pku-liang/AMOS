import os
import json
import tvm
import numpy as np
from tvm.contrib import ndk, tar
from .distribution import general
from .analysis import kmeans
from .device import registry
from .device import measure
from . import space
from . import model


class KernelContext(object):
    def __init__(
        self,
        op_name,
        target,
        algorithm,
        target_host,
        build_func,
        static_params,
        space,
        build_timeout = 10,
        build_parallel = 4,
        verbose = False,
    ):
        """
        kernel_type: str
        space: JoinedSpace
        """
        self.kernel_type = ":".join([op_name, target, algorithm])
        self.op_name = op_name
        self.target = target
        self.algorithm = algorithm
        self.target_host = target_host
        self.build_func = build_func
        self.static_params = static_params
        self.space = space
        self.build_timeout = build_timeout
        self.build_parallel = build_parallel
        self.verbose = verbose


def average_gflops(times, gflop):
    gflop = np.array(gflop).astype("float32")
    times = np.array(times).astype("float32")
    return gflop.sum(axis=1) / (times.sum(axis=1) + 1e-10)


class EvaluationContext(object):
    def __init__(
        self,
        timeout=10,
        verbose=False,
        number=100,
        repeat=1,
        min_repeat_ms=150,
        cooldown_interval=1,
        enable_cpu_cache_flush=0,
        dev_id=0,
        use_rpc=False,
        key=None,
        host=None,
        port=None,
        priority=1,
        aggregate_func=average_gflops
    ):
        """
        EvaluationContext
        """
        self.timeout = timeout
        self.verbose = verbose
        self.number = number
        self.repeat = repeat
        self.min_repeat_ms = min_repeat_ms
        self.cooldown_interval = cooldown_interval
        self.enable_cpu_cache_flush = enable_cpu_cache_flush
        self.dev_id = dev_id
        self.use_rpc = use_rpc
        self.key = key
        self.host = host
        self.port = port
        self.priority = priority
        self.aggregate_func = aggregate_func


class ResultKernelContext(object):
    def __init__(
        self,
        op_name,
        target,
        algorithm,
        target_host,
        build_func,
        static_params,
        configs,
        timeout
    ):
        """
        kernel_type: str
        space: JoinedSpace
        """
        self.kernel_type = ":".join([op_name, target, algorithm])
        self.target = target
        self.target_host = target_host
        self.build_func = build_func
        self.static_params = static_params
        self.configs = configs
        self.timeout = timeout


class ResultMultiKernelContext(object):
    def __init__(self):
        self.result_kernel_contexts = {}

    def add_kernel_context(self, group_id, ctx):
        self.result_kernel_contexts[group_id] = ctx


class CompiledKernel(object):
    def __init__(self, name, func, kernel_config, kernel_type, build_func="default"):
        """
        name: str
        func: tvm.runtime.Module
        kernel_config: dict {str: list}
        kernel_type: str
        build_func: str
        """
        self.name = name
        self.func = func
        self.kernel_config = kernel_config
        self.kernel_type = kernel_type
        self.build_func = build_func

    @staticmethod
    def make_compiled_kernel_from_result_kernel_context(kernel_ctx, kernel_name):
        full_kernel_configs = {}
        full_kernel_configs.update(kernel_ctx.static_params)
        full_kernel_configs.update(kernel_ctx.configs)
        build_results = measure.local_builder_build_shape_oblivious(
            [(kernel_ctx.kernel_type, full_kernel_configs)],
            kernel_ctx.timeout,
            kernel_ctx.target,
            kernel_ctx.target_host,
            1, # build parallel
            kernel_ctx.build_func,
            True # verbose
        )
        res = build_results[0]
        if res.error_no:
            raise RuntimeError(f"Can't build kernel: {res.error_msg}")
        func = tvm.runtime.load_module(res.filename)
        print("\nGet func!", flush=True)
        return CompiledKernel(
            kernel_name,
            func,
            full_kernel_configs,
            kernel_ctx.kernel_type,
            kernel_ctx.build_func
        )

    def run(self, inputs, run_shape):
        """
        inputs: dict {str: tvm.nd.array}
        run_shape:
        """
        tensors, runtime_params = registry.DEVICE_GET_RUNTIME_CTX(
            self.kernel_type,
            self.kernel_config,
            run_shape)
        self.func(*inputs, *run_shape, *runtime_params)

    def evaluate(self, run_shape, measure_opt, new_process):
        return registry.DEVICE_GET_RUNTIME_EVALUATE(
            self, run_shape, measure_opt, new_process
        )

    def save(self, libpath):
        if self.build_func == "default":
            build_func = tar.tar
        elif self.build_func == "ndk":
            build_func = ndk.create_shared
        else:
            raise ValueError(f"Unsupported build_func: {self.build_func}")
        self.func.export_library(
            os.path.join(libpath, self.name + build_func.output_format), build_func)
        meta = {
            "name": self.name,
            "kernel_config": self.kernel_config,
            "kernel_type": self.kernel_type,
            "build_func": self.build_func
        }
        with open(os.path.join(libpath, "meta.json"), "w") as fout:
            fout.write(json.dumps(meta) + "\n")

    @staticmethod
    def load(libpath):
        with open(os.path.join(libpath, "meta.json"), "r") as fin:
            meta = json.loads(fin.readline())
        if meta["build_func"] == "default":
            build_func = tar.tar
        elif meta["build_func"] == "ndk":
            build_func = ndk.create_shared
        else:
            raise ValueError(f"Unsupported build_func: {meta['build_func']}")
        func = tvm.runtime.load_module(
            os.path.join(libpath, meta["name"] + build_func.output_format)
        )
        ret = CompiledKernel(
            meta["name"], func,
            meta["kernel_config"],
            meta["kernel_type"],
            meta["build_func"])
        return ret


class CompiledMultiKernel(object):
    def __init__(self):
        self.kernels = {}

    def add_kernel(self, group_id, compiled_kernel):
        """
        group_id: int
        compiled_kernel: CompiledKernel
        """
        self.kernels[group_id] = compiled_kernel

    def run(self):
        raise NotImplementedError()

    def save(self, path):
        meta = {}
        os.makedirs(path, exist_ok=True)
        for gid, kern in self.kernels.items():
            libpath = os.path.join(path, "kernel" + str(gid))
            os.makedirs(libpath, exist_ok=True)
            kern.save(libpath)
            meta[gid] = libpath
        with open(os.path.join(path, "meta.json"), "w") as fout:
            fout.write(json.dumps(meta) + "\n")

    @staticmethod
    def load(path):
        ret = CompiledMultiKernel()
        with open(os.path.join(path, "meta.json"), "r") as fin:
            meta = json.loads(fin.readline())
        for gid, libpath in meta.items():
            kern = CompiledKernel.load(libpath)
            ret.add_kernel(int(gid), kern)
        return ret


def train_for_one_group(
    group, kernel_ctx, evaluate_ctx, num_rounds, perf_model,
    num_kernels_explore=40, num_kernels_evaluate=20, verbose=False):
    """
    group: ShapeGroup
    kernel_ctx: KernelContext
    num_rounds: int
    ------
    Returns:
    ResultKernelContext
    """
    print(f"Find kernel (type={kernel_ctx.kernel_type}) for group id={group.group_id} (totally {len(group.shapes)} shapes)", flush=True)
    print(f"The space size is {kernel_ctx.space.size()}", flush=True)
    # some constants
    num_iterations = (num_rounds + num_kernels_evaluate - 1) // num_kernels_evaluate
    print(f"Searching for {num_iterations} iterations, {num_kernels_evaluate} items per iteration.", flush=True)
    # prepare temp space
    evaluated_space = space.HeapSpace() # heap
    explored_space = space.HeapSpace() # heap
    unexplored_space = kernel_ctx.space

    GFLOP = [[shape.gflop() for shape in group.shapes]]
    print("GFLOP of each shape:", GFLOP, flush=True)
    for it in range(num_iterations):
        print(f"Iteration {it + 1}: ", flush=True, end="")
        # steps:
        # random sample num_kernels_explore points from space
        # measure them using the model
        # send top num_kernels_evaluate to evaluate
        # use the results to update model
        kernel_configs = unexplored_space.random(batch=num_kernels_explore)
        print(f"#explore: {len(kernel_configs)} ", flush=True, end="")

        predict_matrix = []
        for shape in group.shapes:
            evaluate_configs = [
                (x, shape) for x in kernel_configs
            ]
            evaluate_results = perf_model.predict(kernel_ctx, evaluate_configs)
            predict_matrix.append(evaluate_results)
        predict_matrix = np.array(predict_matrix).transpose()
        
        agg_results = evaluate_ctx.aggregate_func(predict_matrix, GFLOP)
        explored_space.update(kernel_configs, agg_results, predict_matrix)

        selected_configs = explored_space.topk(k=num_kernels_evaluate)
        print(f"#selected: {len(selected_configs)} ", flush=True)
        print(flush=True)
        full_kernel_configs = [
                (kernel_ctx.kernel_type,
                dict(list(x.items()) + list(kernel_ctx.static_params.items())))
                for x in selected_configs
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

            agg_results = evaluate_ctx.aggregate_func(results_matrix, GFLOP)
            evaluated_space.update(selected_configs, agg_results, results_matrix)

            perf_model.update(kernel_ctx, evaluated_space.read_records(), group.shapes)
            res_str = '\n\t\t'.join(map(str, results_lst))
            agg_str = '\n\t\t'.join(map(str, agg_results))
            print(f"Current:\n\tresults=\n\t\t{res_str}\n\tscore=\n\t\t{agg_str}", flush=True)

        all_explored_configs = explored_space.all()
        predict_matrix = []
        for shape in group.shapes:
            evaluate_configs = [
                (x, shape) for x in all_explored_configs
            ]
            evaluate_results = perf_model.predict(kernel_ctx, evaluate_configs)
            predict_matrix.append(evaluate_results)
        predict_matrix = np.array(predict_matrix).transpose()

        agg_results = evaluate_ctx.aggregate_func(predict_matrix, GFLOP)
        explored_space.global_update(all_explored_configs, agg_results, predict_matrix)
        print(f"Best: {evaluated_space.top_value()}", flush=True)

    best_configs = evaluated_space.top_value()[0]
    
    return ResultKernelContext(
        kernel_ctx.op_name,
        kernel_ctx.target,
        kernel_ctx.algorithm,
        kernel_ctx.target_host,
        kernel_ctx.build_func,
        kernel_ctx.static_params,
        best_configs,
        kernel_ctx.build_timeout
    )


def train(
    shapes,
    preprocess_func,
    kernel_ctx,
    evaluate_ctx,
    num_groups=10,
    representative_num=20,
    num_rounds=100,
    model_type="random",
    num_kernels_explore=40,
    num_kernels_evaluate=20,
    verbose=False
):
    """
    num_rounds: one round is to measure one kernel for all the shapes in one group
    ------
    Returns:
    ResultMultiKernelContext
    """
    cluster = kmeans.FaissKMeans(n_clusters=num_groups)
    shapes, counts = preprocess_func(shapes)
    shape_groups = general.group_shapes(
        cluster, shapes, counts)
    
    result_multi_kernel_ctx = ResultMultiKernelContext()

    if model_type == "random":
        perf_model = model.PerfModel(model.RandomOpPerfModel)
    else:
        raise ValueError(f"No support for model type: {model_type}")

    for i, group in enumerate(shape_groups):
        repre_shapes = np.random.choice(group.shapes, representative_num)
        repre_group = general.ShapeGroup(group.group_id, repre_shapes)
        result_kernel_ctx = train_for_one_group(
            repre_group, kernel_ctx, evaluate_ctx, num_rounds, perf_model,
            num_kernels_explore=num_kernels_explore,
            num_kernels_evaluate=num_kernels_evaluate,
            verbose=verbose)
        result_multi_kernel_ctx.add_kernel_context(repre_group.group_id, result_kernel_ctx)

    return result_multi_kernel_ctx