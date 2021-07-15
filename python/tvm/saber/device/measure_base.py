import tvm
import os
import tempfile
import numpy as np
from tvm.contrib import tar, ndk
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from tvm.contrib import ndk


class MeasureOptions(object):
    def __init__(
        self, target="llvm", build_func="default", target_host="llvm", timeout=10,
            verbose=1, number=100, repeat=1, min_repeat_ms=150,
            cooldown_interval=1, enable_cpu_cache_flush=1,
            dev_id=0, use_rpc=False, key=None, host=None, port=None, priority=1):
        self.target = target
        self.build_func = build_func
        self.target_host = target_host
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


EVALUTE_SCHEDULE_INPUTS = None
EVALUTE_FUNCTION_INPUTS = None
MAX_FLOAT = 1e10


def get_np_arrays(tensors):
    ret = []
    for t in tensors:
        dtype = t.dtype
        if str(dtype) == "bfloat16":
            # For now, just simply use float16
            dtype = "float16"
        np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(dtype)
        ret.append(np_ary)
    return ret


def get_tvm_arrays_from_np_arrays(arys, ctx):
    ret = []
    for ary in arys:
        tvm_ary = tvm.nd.array(ary, ctx)
        ret.append(tvm_ary)
    return ret


def get_tvm_arrays(tensors, ctx):
    ret = []
    for t in tensors:
        dtype = t.dtype
        if str(dtype) == "bfloat16":
            # For now, just simply use float16
            dtype = "float16"
        if str(dtype) in ["int4", "int1"]:
            tvm_ary = tvm.nd.empty([int(x) for x in t.shape], dtype, ctx)
        else:
            np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(dtype)
            tvm_ary = tvm.nd.array(np_ary, ctx)
        ret.append(tvm_ary)
    return ret


def evaluate_schedule_worker(dummy):
    global EVALUTE_SCHEDULE_INPUTS
    sch, args, vars, arg_values, var_values, measure_opt = EVALUTE_SCHEDULE_INPUTS
    target = measure_opt.target
    dev_id = measure_opt.dev_id
    number = measure_opt.number
    min_repeat_ms = measure_opt.min_repeat_ms
    use_rpc = measure_opt.key is not None
    if use_rpc:
        key = measure_opt.key
        host = measure_opt.host
        port = measure_opt.port
        priority = measure_opt.priority
        timeout = measure_opt.timeout
        from tvm import auto_scheduler
        remote = auto_scheduler.utils.request_remote(
            key, host, port, priority, timeout)
    ctx = (remote if use_rpc else tvm).context(target, dev_id)
    arrays = get_tvm_arrays(arg_values, ctx)
    func = tvm.build(sch, args + vars, target=target,
                    target_host=measure_opt.target_host if use_rpc else None)
    if use_rpc:
        fd, lib = tempfile.mkstemp(prefix="tmp_func", suffix=".so")
        os.close(fd)
        func.export_library(lib, ndk.create_shared)
        remote.upload(lib)
        func = remote.load_module(os.path.split(lib)[-1])
        os.unlink(lib)
    evaluator = func.time_evaluator(
        func.entry_name, ctx, number=number, min_repeat_ms=min_repeat_ms)
    ctx.sync()
    cost = evaluator(*arrays, *var_values).mean * 1e3
    return cost


def evaluate_schedule(sch, args, vars,
        arg_values, var_values, measure_opt, new_process=True):
    if not new_process:
        target = measure_opt.target
        dev_id = measure_opt.dev_id
        number = measure_opt.number
        min_repeat_ms = measure_opt.min_repeat_ms
        remote = None
        use_rpc = measure_opt.key is not None
        if use_rpc:
            key = measure_opt.key
            host = measure_opt.host
            port = measure_opt.port
            priority = measure_opt.priority
            timeout = measure_opt.timeout
            from tvm import auto_scheduler
            remote = auto_scheduler.utils.request_remote(
                key, host, port, priority, timeout)
        ctx = (remote if use_rpc else tvm).context(target, dev_id)
        arrays = get_tvm_arrays(arg_values, ctx)
        func = tvm.build(sch, args + vars, target=target,
                        target_host=measure_opt.target_host if use_rpc else None)
        if use_rpc:
            fd, lib = tempfile.mkstemp(prefix="tmp_func", suffix=".so")
            os.close(fd)
            func.export_library(lib, ndk.create_shared)
            remote.upload(lib)
            func = remote.load_module(os.path.split(lib)[-1])
            os.unlink(lib)
        evaluator = func.time_evaluator(
            func.entry_name, ctx, number=number, min_repeat_ms=min_repeat_ms)
        ctx.sync()
        cost = evaluator(*arrays, *var_values).mean * 1e3
        return cost
    else:
        global EVALUTE_SCHEDULE_INPUTS
        EVALUTE_SCHEDULE_INPUTS = (sch, args, vars, arg_values, var_values, measure_opt)
        with ProcessPool(1) as pool:
            future = pool.map(evaluate_schedule_worker, [0], timeout=100)
            iterator = future.result()

            while True:
                try:
                    results = next(iterator)
                    print(".Y", end="", flush=True)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(".T", end="", flush=True)
                    results = MAX_FLOAT
                except Exception as error:
                    print(".E", end="", flush=True)
                    results = MAX_FLOAT
                    from tvm import auto_scheduler
                    print(auto_scheduler.utils.make_traceback_info())

        return results


def evaluate_function_worker(dummy):
    global EVALUTE_FUNCTION_INPUTS
    func, args, var_values, measure_opt = EVALUTE_FUNCTION_INPUTS
    # print(args, var_values)
    target = measure_opt.target
    dev_id = measure_opt.dev_id
    number = measure_opt.number
    min_repeat_ms = measure_opt.min_repeat_ms
    use_rpc = measure_opt.key is not None
    if use_rpc:
        key = measure_opt.key
        host = measure_opt.host
        port = measure_opt.port
        priority = measure_opt.priority
        timeout = measure_opt.timeout
        from tvm import auto_scheduler
        remote = auto_scheduler.utils.request_remote(
            key, host, port, priority, timeout)
    ctx = (remote if use_rpc else tvm).context(target, dev_id)
    arrays = get_tvm_arrays(arg_values, ctx)
    func = tvm.build(sch, args + vars, target=target,
                    target_host=measure_opt.target_host if use_rpc else None)
    if use_rpc:
        fd, lib = tempfile.mkstemp(prefix="tmp_func", suffix=".so")
        os.close(fd)
        func.export_library(lib, ndk.create_shared)
        remote.upload(lib)
        func = remote.load_module(os.path.split(lib)[-1])
        os.unlink(lib)
    evaluator = func.time_evaluator(
        func.entry_name, ctx, number=number, min_repeat_ms=min_repeat_ms)
    ctx.sync()
    cost = evaluator(*arrays, *var_values).mean * 1e3
    return cost


def evaluate_function(func, args, var_values, measure_opt, new_process=True):
    if not new_process:
        target = measure_opt.target
        dev_id = measure_opt.dev_id
        number = measure_opt.number
        min_repeat_ms = measure_opt.min_repeat_ms
        build_func = measure_opt.build_func
        remote = None
        use_rpc = measure_opt.key is not None
        if use_rpc:
            key = measure_opt.key
            host = measure_opt.host
            port = measure_opt.port
            priority = measure_opt.priority
            timeout = measure_opt.timeout
            from tvm import auto_scheduler
            remote = auto_scheduler.utils.request_remote(
                key, host, port, priority, timeout)
        ctx = (remote if use_rpc else tvm).context(target, dev_id)
        arrays = get_tvm_arrays(args, ctx)
        if use_rpc:
            if build_func == "default":
                build_func = tar.tar
            elif build_func == "ndk":
                build_func = ndk.create_shared
            else:
                raise ValueError("Invalid build_func" + build_func)
            fd, lib = tempfile.mkstemp(prefix="tmp_func", suffix="." + build_func.output_format)
            os.close(fd)
            func.export_library(lib, build_func)
            remote.upload(lib)
            func = remote.load_module(os.path.split(lib)[-1])
            os.unlink(lib)
        evaluator = func.time_evaluator(
            func.entry_name, ctx, number=number, min_repeat_ms=min_repeat_ms)
        ctx.sync()
        cost = evaluator(*arrays, *var_values).mean * 1e3
        return cost
    else:
        global EVALUTE_FUNCTION_INPUTS
        EVALUTE_FUNCTION_INPUTS = (func, args, var_values, measure_opt)
        with ProcessPool(1) as pool:
            future = pool.map(evaluate_function_worker, [0], timeout=100)
            iterator = future.result()

            while True:
                try:
                    results = next(iterator)
                    print(".Y", end="", flush=True)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(".T", end="", flush=True)
                    results = MAX_FLOAT
                except Exception as error:
                    print(".E", end="", flush=True)
                    results = MAX_FLOAT
                    from tvm import auto_scheduler
                    # print(auto_scheduler.utils.make_traceback_info())

        return results