import tvm
import os
import time
import tempfile
import shutil
import traceback
import numpy as np
from tvm.contrib import tar, ndk
from tvm import auto_scheduler
from tvm.driver import build_module
from tvm.ir import transform
from tvm.runtime import Object, module, ndarray
import multiprocessing as multi
from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from tvm import tg
from collections import OrderedDict
from tempfile import mkstemp
from tvm import rpc
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
GLOBAL_BUILD_INPUTS = None
GLOBAL_RUN_INPUTS = None
GLOBAL_RPC_RUN_INPUTS = None
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
                    # print(auto_scheduler.utils.make_traceback_info())

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


def pebble_local_build_worker_shape_oblivious(index):
    """
    Build function of LocalBuilder to be ran in the Builder thread pool.

    Parameters
    ----------
    index : int
        The MeasureInput index to be processed by the current Builder thread.

    Returns
    -------
    res : BuildResult
        The build result of this Builder thread.
    """
    global GLOBAL_BUILD_INPUTS

    # We use fork and a global variable to copy arguments between processes.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    if not GLOBAL_BUILD_INPUTS:
        raise ValueError("GLOBAL_BUILD_INPUTS not found")
    (
        compile_impl,
        params_lst,
        build_func,
        name,
        target,
        target_host,
        timeout,
        verbose,
        checker
    ) = GLOBAL_BUILD_INPUTS
    assert isinstance(build_func, str)

    if build_func == "default":
        build_func = tar.tar
    elif build_func == "ndk":
        build_func = ndk.create_shared
    else:
        raise ValueError("Invalid build_func" + build_func)

    def timed_func():
        tic = time.time()
        params = params_lst[index]
        sch_impl, args, vars = compile_impl(params)

        error_no = auto_scheduler.measure.MeasureErrorNo.NO_ERROR
        error_msg = None

        try:
            sch = sch_impl()
            ir_module = tvm.lower(sch, [*args, *vars], simple_mode=True)
            checker.check(ir_module)
        # pylint: disable=broad-except
        except Exception:
            error_no = auto_scheduler.measure.MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = auto_scheduler.utils.make_traceback_info()
            # print(error_msg)

        if error_no == 0:
            dirname = tempfile.mkdtemp()
            filename = os.path.join(dirname, "tmp_func." + build_func.output_format)

            try:
                # TODO(merrymercy): Port the unroll pass.
                with transform.PassContext():
                    func = build_module.build(
                        sch, [*args, *vars], target=target, target_host=target_host,
                        name=name
                    )
                func.export_library(filename, build_func)
            # pylint: disable=broad-except
            except Exception:
                error_no = auto_scheduler.measure.MeasureErrorNo.COMPILE_HOST
                error_msg = auto_scheduler.utils.make_traceback_info()
                # print(error_msg)
        else:
            filename = ""

        if verbose >= 1:
            if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                print(".Y", end="", flush=True)
            else:
                print(".E", end="", flush=True)  # Build error

        return (filename, args, error_no, error_msg, time.time() - tic)

    return timed_func()


def pebble_local_builder_build_shape_oblivious(
    compile_impl, params_lst, measure_opt, checker, n_parallel=1, name="main"):
    target = measure_opt.target
    target_host = measure_opt.target_host
    build_func = measure_opt.build_func
    timeout = measure_opt.timeout
    verbose = measure_opt.verbose
    # We use fork and a global variable to copy arguments between processes.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    global GLOBAL_BUILD_INPUTS

    GLOBAL_BUILD_INPUTS = (
        compile_impl, params_lst, build_func, name,
        target, target_host, timeout, verbose, checker)

    with ProcessPool(n_parallel) as pool:
        future = pool.map(pebble_local_build_worker_shape_oblivious, range(len(params_lst)), timeout=timeout)
        iterator = future.result()

        results = []
        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError as error:
                if verbose >= 1:
                    print(".T", end="", flush=True)
                result = None, [], auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT, None, timeout
            except Exception as error:
                if verbose >= 1:
                    print(".F", end="", flush=True)
                result = None, [], auto_scheduler.measure.MeasureErrorNo.COMPILE_HOST, None, timeout
            results.append(auto_scheduler.measure.BuildResult(*result))
    
    if verbose >= 1:
        print("", flush=True)

    return results


def pebble_local_run_worker_shape_oblivious(index):
    global GLOBAL_RUN_INPUTS
    (
        target,
        dev_id,
        build_results,
        param_lst,
        evaluate_impl,
        name,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose,
    ) = GLOBAL_RUN_INPUTS

    def timed_func(build_res, params):
        if build_res.error_no != 0:
            res = (
                (MAX_FLOAT,),
                build_res.error_no,
                build_res.error_msg,
                build_res.time_cost,
                time.time(),
            )
            return res
        tic = time.time()
        error_no = 0
        error_msg = None
        if build_res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
            return (
                (MAX_FLOAT,),
                build_res.error_no,
                build_res.error_msg,
                build_res.time_cost,
                time.time(),
            )

        try:
            func = module.load_module(build_res.filename)
            ctx = ndarray.context(str(target), dev_id)
            # Limitation:
            # We can not get PackFunction directly in the remote mode as it is wrapped
            # under the std::function. We could lift the restriction later once we fold
            # the PackedFunc as an object. Currently, we pass function name to work
            # around it.
            f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
            time_f = func.time_evaluator(
                func.entry_name if name is None else name,
                ctx,
                number=number,
                repeat=repeat,
                min_repeat_ms=min_repeat_ms,
                # f_preproc=f_prepare,
            )
        # pylint: disable=broad-except
        except Exception:
            costs = (MAX_FLOAT,)
            error_no = auto_scheduler.measure.MeasureErrorNo.COMPILE_DEVICE
            error_msg = auto_scheduler.utils.make_traceback_info()

        if error_no == 0:
            costs = []
            tensor_lst, var_value_lst = evaluate_impl(params)
            for tensors, var_values in zip(tensor_lst, var_value_lst):
                try:
                    args = [
                        ndarray.empty(
                            auto_scheduler.utils.get_const_tuple(x.shape),
                            x.dtype, ctx) for x in tensors
                    ]
                    random_fill = tvm.get_global_func("tvm.contrib.random.random_fill", True)
                    assert random_fill, "Please make sure USE_RANDOM is ON in the config.cmake"
                    for arg in args:
                        if str(arg.dtype) in ["int4"]:
                            continue
                        random_fill(arg)
                    ctx.sync()
                    cost = time_f(*args, *var_values).mean
                    costs.append(cost)
                    # print("peek costs:", costs, flush=True)
                # pylint: disable=broad-except
                except Exception:
                    costs.append(MAX_FLOAT)
                    error_no = auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE
                    error_msg = auto_scheduler.utils.make_traceback_info()

        shutil.rmtree(os.path.dirname(build_res.filename))
        toc = time.time()
        time.sleep(cooldown_interval)

        if verbose >= 1:
            if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                print("*Y", end="", flush=True)
            else:
                print("*E", end="", flush=True)  # Run error
        return (costs, error_no, error_msg, toc - tic + build_res.time_cost, toc)

    return timed_func(build_results[index], param_lst[index])


def pebble_local_runner_run_shape_oblivious(build_results, param_lst, evaluate_impl, measure_opt, name="main"):
    target = measure_opt.target
    dev_id = measure_opt.dev_id
    timeout = measure_opt.timeout
    number = measure_opt.number
    repeat = measure_opt.repeat
    min_repeat_ms = measure_opt.min_repeat_ms
    cooldown_interval = measure_opt.cooldown_interval
    enable_cpu_cache_flush = measure_opt.enable_cpu_cache_flush
    verbose = measure_opt.verbose
    global GLOBAL_RUN_INPUTS
    GLOBAL_RUN_INPUTS = (
        target,
        dev_id,
        build_results,
        param_lst,
        evaluate_impl,
        name,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose
    )
    measure_results = []
    with ProcessPool(1) as pool:
        future = pool.map(pebble_local_run_worker_shape_oblivious, range(len(build_results)), timeout=timeout)
        iterator = future.result()

        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError:
                if verbose >= 1:
                    print("*T", end="", flush=True)  # Run timeout
                result = (
                    (MAX_FLOAT,),
                    auto_scheduler.measure.MeasureErrorNo.RUN_TIMEOUT,
                    None,
                    timeout + timeout,
                    time.time(),
                )
            except Exception as error:
                if verbose >= 1:
                    print("*F", end="", flush=True)  # Run fatal error
                result = (
                    (MAX_FLOAT,),
                    auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE,
                    None,
                    timeout + timeout,
                    time.time(),
                )
            measure_results.append(
                auto_scheduler.measure.MeasureResult(*result))

    if verbose >= 1:
        print("", flush=True)

    return measure_results


def pebble_rpc_run_worker_shape_oblivious(index):
    """Function to be ran in the RPCRunner thread pool.

    Parameters
    ----------
    index : int
        The MeasureInput and BuildResult index to be processed by the current Runner thread.

    Returns
    -------
    res : MeasureResult
        The measure result of this Runner thread.
    """
    global GLOBAL_RPC_RUN_INPUTS
    (
        target,
        dev_id,
        build_results,
        param_lst,
        evaluate_impl,
        name,
        key,
        host,
        port,
        priority,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose,
    ) = GLOBAL_RPC_RUN_INPUTS

    max_float = MAX_FLOAT
    build_res = build_results[index]
    params = param_lst[index]

    if build_res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
        return (
            (max_float,),
            build_res.error_no,
            build_res.error_msg,
            build_res.time_cost,
            time.time(),
        )

    def timed_func():
        tic = time.time()
        error_no = 0
        error_msg = None
        try:
            # upload built module
            remote = auto_scheduler.utils.request_remote(key, host, port, priority, timeout)
            remote.upload(build_res.filename)
            func = remote.load_module(os.path.split(build_res.filename)[1])
            ctx = remote.context(str(target), dev_id)
            # Limitation:
            # We can not get PackFunction directly in the remote mode as it is wrapped
            # under the std::function. We could lift the restriction later once we fold
            # the PackedFunc as an object. Currently, we pass function name to work
            # around it.
            f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
            time_f = func.time_evaluator(
                func.entry_name if name is None else name,
                ctx,
                number=number,
                repeat=repeat,
                min_repeat_ms=min_repeat_ms,
                # f_preproc=f_prepare,
            )
        # pylint: disable=broad-except
        except Exception:
            costs = (max_float,)
            error_no = auto_scheduler.measure.MeasureErrorNo.COMPILE_DEVICE
            error_msg = auto_scheduler.utils.make_traceback_info()
            # print(error_msg)

        if error_no == 0:
            costs = []
            tensor_lst, var_value_lst = evaluate_impl(params)
            for tensors, var_values in zip(tensor_lst, var_value_lst):
                try:
                    args = [
                        ndarray.empty(auto_scheduler.utils.get_const_tuple(x.shape), x.dtype, ctx) for x in tensors
                    ]
                    # try:
                    #     random_fill = remote.get_function("tvm.contrib.random.random_fill")
                    # except AttributeError:
                    #     raise AttributeError(
                    #         "Please make sure USE_RANDOM is ON in the config.cmake "
                    #         "on the remote devices"
                    #     )
                    # for arg in args:
                    #     if str(arg.dtype) in ["int4", "int8"]:
                    #         continue
                    #     random_fill(arg)
                    ctx.sync()

                    cost = time_f(*args, *var_values).mean
                    costs.append(cost)
                    # clean up remote files
                    remote.remove(build_res.filename)
                    remote.remove(os.path.splitext(build_res.filename)[0] + ".so")
                    remote.remove("")
                # pylint: disable=broad-except
                except Exception:
                    costs.append(max_float)
                    error_no = auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE
                    error_msg = auto_scheduler.utils.make_traceback_info()
                    # print(error_msg)

        shutil.rmtree(os.path.dirname(build_res.filename))
        toc = time.time()

        time.sleep(cooldown_interval)
        if verbose >= 1:
            if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                print("*Y", end="", flush=True)
            else:
                print("*E", end="", flush=True)  # Run error

        return costs, error_no, error_msg, toc - tic + build_res.time_cost, toc

    return timed_func()


def pebble_rpc_runner_run_shape_oblivious(build_results, param_lst, evaluate_impl, measure_opt, name="main"):
    target = measure_opt.target
    dev_id = measure_opt.dev_id
    timeout = measure_opt.timeout
    number = measure_opt.number
    repeat = measure_opt.repeat
    min_repeat_ms = measure_opt.min_repeat_ms
    cooldown_interval = measure_opt.cooldown_interval
    enable_cpu_cache_flush = measure_opt.enable_cpu_cache_flush
    verbose = measure_opt.verbose
    key = measure_opt.key
    host = measure_opt.host
    port = measure_opt.port
    priority = measure_opt.priority

    global GLOBAL_RPC_RUN_INPUTS
    GLOBAL_RPC_RUN_INPUTS = (
        target,
        dev_id,
        build_results,
        param_lst,
        evaluate_impl,
        name,
        key,
        host,
        port,
        priority,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose,
    )

    measure_results = []
    with ProcessPool(1) as pool:
        future = pool.map(pebble_rpc_run_worker_shape_oblivious, range(len(build_results)), timeout=timeout)
        iterator = future.result()

        while True:
            try:
                result = next(iterator)
            except StopIteration:
                break
            except TimeoutError:
                if verbose >= 1:
                    print("*T", end="", flush=True)  # Run timeout
                result = (
                    (MAX_FLOAT,),
                    auto_scheduler.measure.MeasureErrorNo.RUN_TIMEOUT,
                    None,
                    timeout + timeout,
                    time.time(),
                )
            except Exception as error:
                if verbose >= 1:
                    print("*F", end="", flush=True)  # Run fatal error
                result = (
                    (MAX_FLOAT,),
                    auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE,
                    None,
                    timeout + timeout,
                    time.time(),
                )
            measure_results.append(
                auto_scheduler.measure.MeasureResult(*result))

    if verbose >= 1:
        print("", flush=True)

    return measure_results