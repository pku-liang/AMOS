import tvm
import os
import time
import tempfile
import shutil
import traceback
import numpy as np
import multiprocessing
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
from . import registry
from .measure_base import *


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


GLOBAL_BUILD_INPUTS = None
GLOBAL_RUN_INPUTS = None
GLOBAL_RPC_RUN_INPUTS = None


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


def _timed_func(kernel_type, kernel_config, build_func, target, target_host, verbose):
    tic = time.time()
    error_no = auto_scheduler.measure.MeasureErrorNo.NO_ERROR
    error_msg = None
    args = []

    try:
        sch, args, vars = registry.DEVICE_GET_COMPILE_CTX(kernel_type, kernel_config)
    # pylint: disable=broad-except
    except Exception:
        error_no = auto_scheduler.measure.MeasureErrorNo.INSTANTIATION_ERROR
        error_msg = auto_scheduler.utils.make_traceback_info()

    if error_no == 0:
        dirname = tempfile.mkdtemp()
        filename = os.path.join(dirname, "tmp_func." + build_func.output_format)
        try:
            with transform.PassContext():
                func = build_module.build(
                    sch, [*args, *vars], target=target, target_host=target_host
                )
            func.export_library(filename, build_func)
        # pylint: disable=broad-except
        except Exception:
            error_no = auto_scheduler.measure.MeasureErrorNo.COMPILE_HOST
            error_msg = auto_scheduler.utils.make_traceback_info()
    else:
        filename = ""

    if verbose:
        if error_no ==  auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
            print(".", end="", flush=True)
        else:
            print(".E", end="", flush=True)  # Build error

    return filename, args, error_no, error_msg, time.time() - tic


def local_build_worker_shape_oblivious(args):
    """
    Build function of LocalBuilder to be ran in the Builder thread pool.

    Parameters
    ----------
    args: Tuple[MeasureInput, str, int, int]
        inputs, build-func, time, verbose args passed to local_builder_build

    Returns
    -------
    res : BuildResult
        The build result of this Builder thread.
    """
    kernel_type, kernel_config, build_func, target, target_host, timeout, verbose = args
    if build_func == "ndk":
        build_func = ndk.create_shared
    elif build_func == "default":
        build_func = tar.tar
    else:
        raise ValueError(f"Unsupported build_func: {build_func}")

    res = auto_scheduler.utils.call_func_with_timeout(
        timeout, _timed_func, args=(
            kernel_type, kernel_config, build_func, target, target_host, verbose))
    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print(".T", end="", flush=True)  # Build timeout
        res = None, [], auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT, None, timeout
    elif isinstance(res, Exception):
        if verbose >= 1:
            print(".E", end="", flush=True)  # Build error
        res = None, [], auto_scheduler.measure.MeasureErrorNo.COMPILE_HOST, str(res), timeout

    return res


def local_builder_build_shape_oblivious(
    inputs, timeout, target, target_host, n_parallel, build_func="default", verbose=False):
    """
    Build function of LocalBuilder to build the MeasureInputs to runnable modules.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be built.
    timeout : int
        The timeout limit (in second) for each build thread.
        This is used in a wrapper of the multiprocessing.Process.join().
    n_parallel : int
        Number of threads used to build in parallel.
    build_func : str = 'default'
        The name of build function to process the built module.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program building.

    Returns
    -------
    res : List[BuildResult]
        The build results of these MeasureInputs.
    """
    # This pool is not doing computationally intensive work, so we can use threads
    pool = multiprocessing.pool.ThreadPool(n_parallel)
    tuple_res = pool.map(
        local_build_worker_shape_oblivious,
        [
            (
                kernel_type,
                kernel_config,
                build_func,
                target,
                target_host,
                timeout,
                verbose,
            )
            for (kernel_type, kernel_config) in inputs
        ],
    )
    pool.terminate()
    pool.join()
    del pool

    results = []
    for res in tuple_res:
        results.append(auto_scheduler.measure.BuildResult(*res))

    return results


def _timed_eval_func(
    kernel_type,
    kernel_config,
    run_shapes,
    build_res,
    target,
    dev_id,
    number,
    repeat,
    min_repeat_ms,
    cooldown_interval,
    enable_cpu_cache_flush,
    verbose,
):
    tic = time.time()
    error_no = 0
    error_msg = None
    try:
        func = module.load_module(build_res.filename)
        ctx = ndarray.context(target, dev_id)
        # Limitation:
        # We can not get PackFunction directly in the remote mode as it is wrapped
        # under the std::function. We could lift the restriction later once we fold
        # the PackedFunc as an object. Currently, we pass function name to work
        # around it.
        f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
        time_f = func.time_evaluator(
            func.entry_name,
            ctx,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            f_preproc=f_prepare,
        )
    # pylint: disable=broad-except
    except Exception:
        costs = (MAX_FLOAT,)
        error_no = auto_scheduler.measure.MeasureErrorNo.COMPILE_DEVICE
        error_msg = auto_scheduler.utils.make_traceback_info()

    if error_no == 0:
        try:
            costs = []
            for run_shape in run_shapes:
                tensors, var_values = registry.DEVICE_GET_RUNTIME_CTX(kernel_type, kernel_config, run_shape)
                args = [ndarray.empty(
                    auto_scheduler.utils.get_const_tuple(x.shape), x.dtype, ctx) for x in tensors]
                random_fill = tvm.get_global_func("tvm.contrib.random.random_fill", True)
                assert random_fill, "Please make sure USE_RANDOM is ON in the config.cmake"
                for arg in args:
                    random_fill(arg)
                ctx.sync()
                tmp_costs = time_f(*args, *var_values).results
                tmp_costs = np.mean(
                    np.array(
                        [float(x) if isinstance(x, float) else float(x.value) for x in tmp_costs]
                    )
                )
                costs.append(float(tmp_costs))
        # pylint: disable=broad-except
        except Exception:
            costs = (MAX_FLOAT,)
            error_no = auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE
            error_msg = auto_scheduler.utils.make_traceback_info()

    shutil.rmtree(os.path.dirname(build_res.filename))
    toc = time.time()
    time.sleep(cooldown_interval)

    if verbose >= 1:
        if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
            print("*", end="", flush=True)
        else:
            print("*E", end="", flush=True)  # Run error
    return costs, error_no, error_msg, toc - tic + build_res.time_cost, toc


def local_run(
    inputs,
    build_results,
    target,
    dev_id,
    timeout=10,
    number=3,
    repeat=1,
    min_repeat_ms=0,
    cooldown_interval=0,
    enable_cpu_cache_flush=False,
    verbose=1,
):
    """
    Run function of LocalRunner to test the performance of the input BuildResults.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be measured.
    build_results : List[BuildResult]
        The BuildResults to be measured.
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program measuring.

    Returns
    -------
    res : List[MeasureResult]
        The measure results of these MeasureInputs.
    """

    measure_results = []
    assert len(inputs) == len(build_results), "Measure input size should be equal to build results"
    for (kernel_type, kernel_config, run_shapes), build_res in zip(inputs, build_results):
        if build_res.error_no != 0:
            res = (
                (MAX_FLOAT,),
                build_res.error_no,
                build_res.error_msg,
                build_res.time_cost,
                time.time(),
            )
        else:
            res = auto_scheduler.utils.call_func_with_timeout(
                timeout,
                _timed_eval_func,
                args=(
                    kernel_type,
                    kernel_config,
                    run_shapes,
                    build_res,
                    target,
                    dev_id,
                    number,
                    repeat,
                    min_repeat_ms,
                    cooldown_interval,
                    enable_cpu_cache_flush,
                    verbose,
                ),
                add_thread_wrapper=True,
            )
            if isinstance(res, TimeoutError):
                if verbose >= 1:
                    print("*T", end="", flush=True)  # Run timeout
                res = (
                    (MAX_FLOAT,),
                    auto_scheduler.measure.MeasureErrorNo.RUN_TIMEOUT,
                    None,
                    build_res.time_cost + timeout,
                    time.time(),
                )
            elif isinstance(res, Exception):
                if verbose >= 1:
                    print("*E", end="", flush=True)  # Run error
                res = (
                    (MAX_FLOAT,),
                    auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE,
                    str(res),
                    build_res.time_cost + timeout,
                    time.time(),
                )

        measure_results.append(auto_scheduler.measure.MeasureResult(*res))

    if verbose >= 1:
        print("", flush=True)

    return measure_results


def _timed_rpc_run(
    kernel_type,
    kernel_config,
    run_shape,
    build_res,
    target,
    dev_id,
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
):
    tic = time.time()
    error_no = 0
    error_msg = None
    try:
        # upload built module
        remote = auto_scheduler.utils.request_remote(key, host, port, priority, timeout)
        remote.upload(build_res.filename)
        func = remote.load_module(os.path.split(build_res.filename)[1])
        ctx = remote.context(target, dev_id)
        # Limitation:
        # We can not get PackFunction directly in the remote mode as it is wrapped
        # under the std::function. We could lift the restriction later once we fold
        # the PackedFunc as an object. Currently, we pass function name to work
        # around it.
        f_prepare = "cache_flush_cpu_non_first_arg" if enable_cpu_cache_flush else ""
        time_f = func.time_evaluator(
            func.entry_name,
            ctx,
            number=number,
            repeat=repeat,
            min_repeat_ms=min_repeat_ms,
            f_preproc=f_prepare,
        )
    # pylint: disable=broad-except
    except Exception:
        costs = (MAX_FLOAT,)
        error_no = auto_scheduler.measure.MeasureErrorNo.COMPILE_DEVICE
        error_msg = auto_scheduler.utils.make_traceback_info()

    if error_no == 0:
        try:
            tensors, var_values = registry.DEVICE_GET_RUNTIME_CTX(kernel_type, kernel_config, run_shape)
            args = [ndarray.empty(
                auto_scheduler.utils.get_const_tuple(x.shape), x.dtype, ctx) for x in tensors]
            try:
                random_fill = remote.get_function("tvm.contrib.random.random_fill")
            except AttributeError:
                raise AttributeError(
                    "Please make sure USE_RANDOM is ON in the config.cmake " "on the remote devices"
                )
            for arg in args:
                random_fill(arg)
            ctx.sync()

            costs = time_f(*args, *var_values).results
            # clean up remote files
            remote.remove(build_res.filename)
            remote.remove(os.path.splitext(build_res.filename)[0] + ".so")
            remote.remove("")
        # pylint: disable=broad-except
        except Exception:
            costs = (MAX_FLOAT,)
            error_no = auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE
            error_msg = auto_scheduler.utils.make_traceback_info()

    shutil.rmtree(os.path.dirname(build_res.filename))
    toc = time.time()

    time.sleep(cooldown_interval)
    if verbose >= 1:
        if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
            print("*", end="")
        else:
            print("*E", end="")  # Run error

    return costs, error_no, error_msg, toc - tic + build_res.time_cost, toc


def _rpc_run_worker(args):
    """Function to be ran in the RPCRunner thread pool.

    Parameters
    ----------
    args : Tuple[MeasureInput, BuildResult, ...]
        Single input and build result plus the rest of the arguments to `rpc_runner_run`.

    Returns
    -------
    res : MeasureResult
        The measure result of this Runner thread.
    """
    _, _, _, build_res, _, _, _, _, _, _, timeout, _, _, _, _, _, verbose = args
    if build_res.error_no != auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
        return (
            (MAX_FLOAT,),
            build_res.error_no,
            build_res.error_msg,
            build_res.time_cost,
            time.time(),
        )

    res = auto_scheduler.utils.call_func_with_timeout(timeout, _timed_rpc_run, args=args)
    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print("*T", end="")  # Run timeout
        res = (
            (MAX_FLOAT,),
            auto_scheduler.measure.MeasureErrorNo.RUN_TIMEOUT,
            None,
            build_res.time_cost + timeout,
            time.time(),
        )
    elif isinstance(res, Exception):
        if verbose >= 1:
            print("*E", end="")  # Run error
        res = (
            (MAX_FLOAT,),
            auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE,
            str(res),
            build_res.time_cost + timeout,
            time.time(),
        )

    return res


def rpc_runner_run(
    inputs,
    build_results,
    target,
    dev_id,
    key,
    host,
    port,
    priority=1,
    n_parallel=1,
    timeout=10,
    number=3,
    repeat=1,
    min_repeat_ms=0,
    cooldown_interval=0.0,
    enable_cpu_cache_flush=False,
    verbose=1,
):
    """Run function of RPCRunner to test the performance of the input BuildResults.

    Parameters
    ----------
    inputs : List[MeasureInput]
        The MeasureInputs to be measured.
    build_results : List[BuildResult]
        The BuildResults to be measured.
    key : str
        The key of the device registered in the RPC tracker.
    host : str
        The host address of the RPC Tracker.
    port : int
        The port of RPC Tracker.
    priority : int = 1
        The priority of this run request, larger is more prior.
    n_parallel : int = 1
        The number of tasks run in parallel.
    timeout : int = 10
        The timeout limit (in second) for each run.
        This is used in a wrapper of the multiprocessing.Process.join().
    number : int = 3
        The number of times to run the generated code for taking average.
        We call these runs as one `repeat` of measurement.
    repeat : int = 1
        The number of times to repeat the measurement.
        In total, the generated code will be run (1 + number x repeat) times,
        where the first "1" is warm up and will be discarded.
        The returned result contains `repeat` costs,
        each of which is an average of `number` costs.
    min_repeat_ms : int = 0
        The minimum duration of one `repeat` in milliseconds.
        By default, one `repeat` contains `number` runs. If this parameter is set,
        the parameters `number` will be dynamically adjusted to meet the
        minimum duration requirement of one `repeat`.
        i.e., When the run time of one `repeat` falls below this time, the `number` parameter
        will be automatically increased.
    cooldown_interval : float = 0.0
        The cool down interval between two measurements.
    enable_cpu_cache_flush: bool = False
        Whether to flush cache on CPU between repeated measurements.
        Flushing cache can make the measured latency of one operator closer to
        its actual latency during end-to-end inference.
        To make this option effective, the argument `number` should also be set to 1.
        This is only has effect on CPU task.
    verbose: int = 1
        Verbosity level. 0 for silent, 1 to output information during program measuring.

    Returns
    -------
    res : List[MeasureResult]
        The measure results of these MeasureInputs.
    """
    assert len(inputs) == len(build_results), "Measure input size should be equal to build results"
    # This pool is not doing computationally intensive work, so we can use threads
    pool = multiprocessing.pool.ThreadPool(n_parallel)
    tuple_res = pool.map(
        _rpc_run_worker,
        [
            (
                kernel_type,
                kernel_config,
                run_shape,
                build_res,
                target,
                dev_id,
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
            for (kernel_type, kernel_config, run_shape), build_res in zip(inputs, build_results)
        ],
    )
    pool.terminate()
    pool.join()
    del pool

    results = []
    for res in tuple_res:
        results.append(auto_scheduler.measure.MeasureResult(*res))

    if verbose >= 1:
        print("")

    return results
