import tvm
import os
import time
import tempfile
import shutil
import numpy as np
from tvm import testing
from tvm import auto_tensorize as at
from tvm.contrib import tar, ndk
from tvm import auto_scheduler
from tvm.ir import transform
from tvm.driver import build_module
from tvm.runtime import Object, module, ndarray
import multiprocessing as multi
from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from tvm import tg
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


def conv2d(N, C, H, W, K, R, S, stride, padding, with_bias=True, in_dtype=["float16", "float16"], out_dtype="float32"):
    H = H + 2 * padding
    W = W + 2 * padding
    A = tvm.te.placeholder([N, C, H, W], dtype=in_dtype[0], name="A")
    B = tvm.te.placeholder([K, C, R, S], dtype=in_dtype[1], name="B")
    rc = tvm.te.reduce_axis([0, C], name="rc")
    rr = tvm.te.reduce_axis([0, R], name="rr")
    rs = tvm.te.reduce_axis([0, S], name="rs")

    P = (H - R) // stride + 1
    Q = (W - S) // stride + 1
    if in_dtype[0] == "uint8":
        Conv = tvm.te.compute(
            [N, K, P, Q],
            lambda n, k, p, q:
                tvm.te.sum((A[n, rc, p+rr, q+rs].astype(out_dtype) * B[k, rc, rr, rs].astype(out_dtype)
                            ), axis=[rc, rr, rs]),
            name="Conv"
        )
    else:
        Conv = tvm.te.compute(
            [N, K, P, Q],
            lambda n, k, p, q:
                tvm.te.sum((A[n, rc, p+rr, q+rs] * B[k, rc, rr, rs]
                            ).astype(out_dtype), axis=[rc, rr, rs]),
            name="Conv"
        )
    if not with_bias:
        return [A, B, Conv]
    bias = tvm.te.placeholder([N, K, P, Q], dtype=out_dtype, name="bias")
    E = tvm.te.compute(
        [N, K, P, Q],
        lambda bn, bk, bp, bq: Conv[bn, bk, bp, bq] + bias[bn, bk, bp, bq],
        name="E"
    )
    return [A, B, bias, E]


def get_np_arrays(tensors):
    ret = []
    for t in tensors:
        np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(t.dtype)
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
        np_ary = np.random.uniform(-1, 1, [int(x)
                                           for x in t.shape]).astype(t.dtype)
        tvm_ary = tvm.nd.array(np_ary, ctx)
        ret.append(tvm_ary)
    return ret


@register_test
def test1():
    print("##########################")
    print("Test 1")
    recipe = at.WMMAFp16Fp32()
    compute_key = "nnn"
    shape_key = "8x32x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_np_arrays = get_np_arrays(inputs_ref)
    inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[0].axis
    rc, rr, rs = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [n, n, n, p, p, q, q],
        jj: [k, k, k, k, k, k, k],
        kk: [rc, rr, rs, rc, rs, rc, rr]
    }
    match_result = at.IntrinMatchResult(
        recipe, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    gen = at.TransformGenerator(match_result)
    for i in range(1):
        record = gen.get()
        record.unfold_choice = ([1, 0, 0, 0, 0, 0, 1], record.unfold_choice[1])
        print(record.to_json())
        app = at.TransformApplier(match_result)
        new_state = app.apply(record)

        # print("Compare new state and old state:")
        # print("new axis map:", new_state.axis_map)
        # tmp = []
        # for k, v in new_state.axis_map.items():
        #     tmp.append(v)
        # for tri in zip(*tmp):
        #     print(tri)
        # print("new main op map:", new_state.main_op_map)

        # new_target_dag = new_state.target_dag
        # print("org dag len:", len(new_target_dag.op_lst))
        # new_target_main_op = None
        # for k, v in new_state.main_op_map.items():
        #     new_target_main_op = v
        # assert new_target_main_op is not None

        # new_target_dag, _ = at.reconstruct_dag_as_intrin(
        #     new_target_dag, new_target_main_op, recipe, compute_key, shape_key)
        # print("new dag len:", len(new_target_dag.op_lst))

        # print("new dag load A op:",
        #       new_target_dag.op_lst[2].axis, new_target_dag.op_lst[2].body)
        # print("new dag load B op:",
        #       new_target_dag.op_lst[5].axis, new_target_dag.op_lst[5].body)
        # print("new dag main op:",
        #       new_target_dag.op_lst[6].axis, new_target_dag.op_lst[6].body)
        # print("new dag store op:",
        #       new_target_dag.op_lst[7].axis, new_target_dag.op_lst[7].body)

        schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
        params = schedule_gen.get()
        my_params = {
            'vectorize': (1, 1),
            'spatial_factors': [([1, 1, 1], (0, 0)), ([4, 1, 1], (-1, 1)), ([14, 1, 1], (-1, -1))],
            'reduce_factors': [([3, 3, 4], (1, 1)), ([1, 1, 3], (0, -1))],
            'last_factors': [([-1, 32], (-1,))],
            'output_unroll_step': (64, -1),
            'last_unroll_step': (512, 1)}
        params.from_json(my_params)
        print(params.to_json())
        
        new_target_dag = sc_info.target_dag
        new_inputs = new_target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
        # print(tvm.lower(
        #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True), flush=True)
        # print("new dag len:", len(new_target_dag.op_lst))

        # print("new dag load A op:",
        #       new_target_dag.op_lst[2].axis, new_target_dag.op_lst[2].body)
        # print("new dag load B op:",
        #       new_target_dag.op_lst[5].axis, new_target_dag.op_lst[5].body)
        # print("new dag main op:",
        #       new_target_dag.op_lst[6].axis, new_target_dag.op_lst[6].body)
        # print("new dag store op:",
        #       new_target_dag.op_lst[7].axis, new_target_dag.op_lst[7].body)

        schedule_app.apply(sch, params)

        # print(tvm.lower(
        #     sch, new_inputs + list(new_target_dag.tensors), simple_mode=True), flush=True)
        func = tvm.build(sch, new_inputs +
                         list(new_target_dag.tensors), "cuda")
        # print(func.imported_modules[0].get_source())
        # print(new_target_dag.tensors)
        ctx = tvm.gpu()
        inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
        outputs_arrays = get_tvm_arrays(list(new_target_dag.tensors), ctx)
        func(*inputs_arrays, *outputs_arrays)
        for a, b in zip(outputs_arrays_ref, outputs_arrays):
            testing.assert_allclose(
                a.asnumpy(), b.asnumpy(), atol=1e-3, rtol=1e-2)
        
        # get performance
        evaluator = func.time_evaluator(func.entry_name, ctx, number=10)
        costs = evaluator(*inputs_arrays, *outputs_arrays)
        print("Time cost: %f ms." % (costs.mean * 1e3))

        gen.feedback(record, np.random.random())
    print("Pass!\n")


GLOBAL_BUILD_INPUTS = None
GLOBAL_RUN_INPUTS = None
GLOBAL_RPC_RUN_INPUTS = None
MAX_FLOAT = 1e10  # We use 1e10 instead of sys.float_info.max for better readability in log


# this is similar to auto_scheduler
def native_loacl_build_worker(index):
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
        sch_app,
        params_lst,
        build_func,
        target,
        target_host,
        timeout,
        verbose
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
        target_dag = sch_app.target_dag
        inputs = target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])

        error_no = auto_scheduler.measure.MeasureErrorNo.NO_ERROR
        error_msg = None
        args = inputs + list(target_dag.tensors)

        try:
            sch = sch_app.apply(sch, params)
        # pylint: disable=broad-except
        except Exception:
            error_no = auto_scheduler.measure.MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = auto_scheduler.measure.make_error_msg()

        if error_no == 0:
            dirname = tempfile.mkdtemp()
            filename = os.path.join(dirname, "tmp_func." + build_func.output_format)

            try:
                # TODO(merrymercy): Port the unroll pass.
                with transform.PassContext():
                    func = build_module.build(
                        sch, args, target=target, target_host=target_host
                    )
                func.export_library(filename, build_func)
            # pylint: disable=broad-except
            except Exception:
                error_no = auto_scheduler.measure.MeasureErrorNo.COMPILE_HOST
                error_msg = auto_scheduler.measure.make_error_msg()
        else:
            filename = ""

        if verbose >= 1:
            if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                print(".Y", end="", flush=True)
            else:
                print(".E", end="", flush=True)  # Build error

        return (filename, args, error_no, error_msg, time.time() - tic)

    res = auto_scheduler.utils.call_func_with_timeout(timeout, timed_func)
    if isinstance(res, TimeoutError):
        if verbose >= 1:
            print(".T", end="")  # Build timeout
        res = None, [], auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT, None, timeout

    return res


def native_local_builder_build(sch_app, params_lst, target, target_host, timeout, n_parallel, build_func="default", verbose=1):
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
    # We use fork and a global variable to copy arguments between processes.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    global GLOBAL_BUILD_INPUTS

    GLOBAL_BUILD_INPUTS = (sch_app, params_lst, build_func, target, target_host, timeout, verbose)

    pool = auto_scheduler.measure.NoDaemonPool(n_parallel)
    tuple_res = pool.map(native_loacl_build_worker, range(len(params_lst)))
    pool.terminate()
    pool.join()
    del pool

    results = []
    for res in tuple_res:
        results.append(auto_scheduler.measure.BuildResult(*res))
    
    if verbose >= 1:
        print("", flush=True)

    return results


# this is similar to auto_scheduler
def native_local_run_worker():
    global GLOBAL_RUN_INPUTS
    (
        target,
        dev_id,
        build_results,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose,
    ) = GLOBAL_RUN_INPUTS

    def timed_func(build_res):
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
                func.entry_name,
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
            error_msg = auto_scheduler.measure.make_error_msg()

        if error_no == 0:
            try:
                args = [
                    ndarray.empty(auto_scheduler.utils.get_const_tuple(x.shape), x.dtype, ctx) for x in build_res.args
                ]
                random_fill = tvm.get_global_func("tvm.contrib.random.random_fill", True)
                assert random_fill, "Please make sure USE_RANDOM is ON in the config.cmake"
                for arg in args:
                    random_fill(arg)
                ctx.sync()
                costs = time_f(*args).results
            # pylint: disable=broad-except
            except Exception:
                costs = (MAX_FLOAT,)
                error_no = auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE
                error_msg = auto_scheduler.measure.make_error_msg()

        shutil.rmtree(os.path.dirname(build_res.filename))
        toc = time.time()
        time.sleep(cooldown_interval)

        if verbose >= 1:
            if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                print("*Y", end="", flush=True)
            else:
                print("*E", end="", flush=True)  # Run error
        return (costs, error_no, error_msg, toc - tic + build_res.time_cost, toc)

    measure_results = []
    for build_res in build_results:
        if build_res.error_no != 0:
            res = (
                (MAX_FLOAT,),
                build_res.error_no,
                build_res.error_msg,
                build_res.time_cost,
                time.time(),
            )
        else:
            res = auto_scheduler.utils.call_func_with_timeout(timeout, timed_func, args=(build_res,))
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
        measure_results.append(auto_scheduler.measure.MeasureResult(*res))

    if verbose >= 1:
        print("", flush=True)

    return measure_results


@register_test
def test2():
    print("##########################")
    print("Test 2")
    recipe = at.WMMAFp16Fp32()
    compute_key = "nnn"
    shape_key = "8x32x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_np_arrays = get_np_arrays(inputs_ref)
    inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[0].axis
    rc, rr, rs = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [n, n, n, p, p, q, q],
        jj: [k, k, k, k, k, k, k],
        kk: [rc, rr, rs, rc, rs, rc, rr]
    }
    match_result = at.IntrinMatchResult(
        recipe, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    gen = at.TransformGenerator(match_result)
    beg = time.time()
    for i in range(1):
        record = gen.get(policy="random")
        record.unfold_choice = ([1, 0, 0, 0, 0, 0, 1], record.unfold_choice[1])
 
        print("transform decision:")
        for k, v in record.to_json().items():
            print(k, "=", v)

        app = at.TransformApplier(match_result)
        new_state = app.apply(record)

        schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
        params_lst = []
        trials = 10
        print("trials=", trials)
        for j in range(trials):
            params = schedule_gen.get(policy="random")
            my_params = {
                'vectorize': (1, 1),
                'spatial_factors': [([1, 1, 1], (0, 0)), ([4, 1, 1], (-1, 1)), ([14, 1, 1], (-1, -1))],
                'reduce_factors': [([3, 3, 4], (1, 1)), ([1, 1, 3], (0, -1))],
                'last_factors': [([-1, 32], (-1,))],
                'output_unroll_step': (64, -1),
                'last_unroll_step': (512, 1)}
            params.from_json(my_params)
            params_lst.append(params)

        global GLOBAL_BUILD_INPUTS
        global GLOBAL_RUN_INPUTS
        build_func = "default"
        target_host = "llvm"
        timeout = 15
        verbose = 1

        GLOBAL_BUILD_INPUTS = (
            schedule_app,
            params_lst,
            build_func,
            recipe.target,
            target_host,
            verbose
        )

        build_results = native_local_builder_build(
            schedule_app,
            params_lst,
            recipe.target,
            target_host,
            timeout,
            1,
            build_func=build_func,
            verbose=verbose)

        for r in build_results:
            print(r)
        
        number = 1
        repeat = 1
        min_repeat_ms = 150
        cooldown_interval = 1
        enable_cpu_cache_flush = 1

        GLOBAL_RUN_INPUTS = (
            recipe.target,
            0,
            build_results,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
            verbose
        )

        run_results = native_local_run_worker()

        for r in run_results:
            print(r)

        gen.feedback(record, np.random.random())
    end = time.time()
    print("Pass %f seconds." % (end - beg))
    print("Pass!\n")


def pebble_local_build_worker(index):
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
        sch_app,
        params_lst,
        build_func,
        name,
        target,
        target_host,
        timeout,
        verbose
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
        target_dag = sch_app.target_dag
        inputs = target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])

        error_no = auto_scheduler.measure.MeasureErrorNo.NO_ERROR
        error_msg = None
        args = inputs + list(target_dag.tensors)

        try:
            sch = sch_app.apply(sch, params)
            print(tvm.lower(sch, args, simple_mode=True))
        # pylint: disable=broad-except
        except Exception:
            error_no = auto_scheduler.measure.MeasureErrorNo.INSTANTIATION_ERROR
            error_msg = auto_scheduler.measure.make_error_msg()
            print(error_msg)

        if error_no == 0:
            dirname = tempfile.mkdtemp()
            filename = os.path.join(dirname, "tmp_func." + build_func.output_format)

            try:
                # TODO(merrymercy): Port the unroll pass.
                with transform.PassContext():
                    func = build_module.build(
                        sch, args, target=target, target_host=target_host,
                        name=name
                    )
                func.export_library(filename, build_func)
            # pylint: disable=broad-except
            except Exception:
                error_no = auto_scheduler.measure.MeasureErrorNo.COMPILE_HOST
                error_msg = auto_scheduler.measure.make_error_msg()
        else:
            filename = ""

        if verbose >= 1:
            if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                print(".Y", end="", flush=True)
            else:
                print(".E", end="", flush=True)  # Build error

        return (filename, args, error_no, error_msg, time.time() - tic)

    return timed_func()


def pebble_local_builder_build(
    sch_app, params_lst, target, target_host, timeout, n_parallel,
    build_func="default", verbose=1, name="main"):
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
    # We use fork and a global variable to copy arguments between processes.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    global GLOBAL_BUILD_INPUTS

    GLOBAL_BUILD_INPUTS = (
        sch_app, params_lst, build_func, name,
        target, target_host, timeout, verbose)

    with ProcessPool(n_parallel) as pool:
        future = pool.map(pebble_local_build_worker, range(len(params_lst)), timeout=timeout)
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


# this is similar to auto_scheduler
def pebble_local_run_worker(index):
    global GLOBAL_RUN_INPUTS
    (
        target,
        dev_id,
        build_results,
        name,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose,
    ) = GLOBAL_RUN_INPUTS

    def timed_func(build_res):
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
            error_msg = auto_scheduler.measure.make_error_msg()

        if error_no == 0:
            try:
                args = [
                    ndarray.empty(auto_scheduler.utils.get_const_tuple(x.shape), x.dtype, ctx) for x in build_res.args
                ]
                random_fill = tvm.get_global_func("tvm.contrib.random.random_fill", True)
                assert random_fill, "Please make sure USE_RANDOM is ON in the config.cmake"
                for arg in args:
                    random_fill(arg)
                ctx.sync()
                costs = time_f(*args).results
            # pylint: disable=broad-except
            except Exception:
                costs = (MAX_FLOAT,)
                error_no = auto_scheduler.measure.MeasureErrorNo.RUNTIME_DEVICE
                error_msg = auto_scheduler.measure.make_error_msg()

        shutil.rmtree(os.path.dirname(build_res.filename))
        toc = time.time()
        time.sleep(cooldown_interval)

        if verbose >= 1:
            if error_no == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                print("*Y", end="", flush=True)
            else:
                print("*E", end="", flush=True)  # Run error
        return (costs, error_no, error_msg, toc - tic + build_res.time_cost, toc)

    return timed_func(build_results[index])


def pebble_local_runner_run(
        target,
        dev_id,
        build_results,
        timeout,
        number,
        repeat,
        min_repeat_ms,
        cooldown_interval,
        enable_cpu_cache_flush,
        verbose=1,
        name="main"):
    global GLOBAL_RUN_INPUTS
    GLOBAL_RUN_INPUTS = (
        target,
        dev_id,
        build_results,
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
        future = pool.map(pebble_local_run_worker, range(len(build_results)), timeout=timeout)
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


@register_test
def test3():
    print("##########################")
    print("Test 3")
    recipe = at.WMMAFp16Fp32()
    compute_key = "nnn"
    shape_key = "8x32x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_np_arrays = get_np_arrays(inputs_ref)
    inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[0].axis
    rc, rr, rs = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [n, n, n, p, p, q, q],
        jj: [k, k, k, k, k, k, k],
        kk: [rc, rr, rs, rc, rs, rc, rr]
    }
    match_result = at.IntrinMatchResult(
        recipe, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    gen = at.TransformGenerator(match_result)
    beg = time.time()
    for i in range(1):
        record = gen.get(policy="random")
        record.unfold_choice = ([1, 1, 1, 1, 1, 1, 1], record.unfold_choice[1])
 
        print("transform decision:")
        for k, v in record.to_json().items():
            print(k, "=", v)

        app = at.TransformApplier(match_result)
        new_state = app.apply(record)

        schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
        params_lst = []
        trials = 32
        print("trials=", trials)
        for j in range(trials):
            params = schedule_gen.get(policy="random")
            # my_params = {
            #     'vectorize': (1, 1),
            #     'spatial_factors': [([1, 1, 1], (0, 0)), ([4, 1, 1], (-1, 1)), ([14, 1, 1], (-1, -1))],
            #     'reduce_factors': [([3, 3, 4], (1, 1)), ([1, 1, 3], (0, -1))],
            #     'last_factors': [([-1, 32], (-1,))],
            #     'output_unroll_step': (64, -1),
            #     'last_unroll_step': (512, 1)}
            # params.from_json(my_params)
            params_lst.append(params)

        build_func = "default"
        target_host = "llvm"
        timeout = 15
        verbose = 1

        build_results = pebble_local_builder_build(
            schedule_app,
            params_lst,
            recipe.target,
            target_host,
            timeout,
            1,
            build_func=build_func,
            verbose=verbose)

        for r in build_results:
            print(r)
        
        number = 1
        repeat = 1
        min_repeat_ms = 150
        cooldown_interval = 1
        enable_cpu_cache_flush = 1

        run_results = pebble_local_runner_run(
            recipe.target,
            0,
            build_results,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
            verbose
        )

        for r in run_results:
            print(r)

        gen.feedback(record, np.random.random())
    end = time.time()
    print("Pass %f seconds." % (end - beg))
    print("Pass!\n")


def tg_parallel_build_worker(name):
    global GLOBAL_BUILD_INPUTS

    # We use fork and a global variable to copy arguments between processes.
    # This can avoid expensive serialization of TVM IR when using multiprocessing.Pool
    if not GLOBAL_BUILD_INPUTS:
        raise ValueError("GLOBAL_BUILD_INPUTS not found")
    (
        sch_app,
        params_lst,
        build_func,
        target,
        target_host,
        verbose
    ) = GLOBAL_BUILD_INPUTS
    assert isinstance(build_func, str)

    if build_func == "default":
        build_func = tar.tar
    elif build_func == "ndk":
        build_func = ndk.create_shared
    else:
        raise ValueError("Invalid build_func" + build_func)

    target_dag = sch_app.target_dag
    inputs = target_dag.get_inputs()
    args = inputs + list(target_dag.tensors)
    schs = []
    err_nos = []
    err_msgs = []
    filenames = []
    rets = []
    tic = time.time()
    for i, params in enumerate(params_lst):
        sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])
        try:
            sch = sch_app.apply(sch, params)
            schs.append(sch)
            err_nos.append(auto_scheduler.measure.MeasureErrorNo.NO_ERROR)
            err_msgs.append(None)
        except Exception:
            err_nos.append(auto_scheduler.measure.MeasureErrorNo.INSTANTIATION_ERROR)
            err_msgs.append(auto_scheduler.measure.make_error_msg())
        filenames.append("")
    
    if schs:
        mod_err_nos = []
        mod_err_msgs = []
        mods = tg.parallel_build(
            schs, args, target=target, target_host=target_host, name=name)
        p_mod = 0
        for i, err in enumerate(err_nos):
            if err == auto_scheduler.measure.MeasureErrorNo.NO_ERROR:
                mod = mods[p_mod]
                p_mod += 1
                if mod is None:
                    if verbose >= 1:
                        print(".E", end="", flush=True)
                    mod_err_nos.append(
                        auto_scheduler.measure.MeasureErrorNo.COMPILE_HOST
                    )
                    mod_err_msgs.append(
                        "Build error in tg.parallel_build"
                    )
                else:
                    if verbose >= 1:
                        print(".Y", end="", flush=True)
                    mod_err_nos.append(err)
                    mod_err_msgs.append(err_msgs[i])
                    dirname = tempfile.mkdtemp()
                    filename = os.path.join(
                        dirname, "tmp_func." + build_func.output_format)
                    filenames[i] = filename
                    mod.export_library(filename, build_func)
            else:
                if verbose >= 1:
                    print(".I", end="", flush=True)
                mod_err_nos.append(err)
                mod_err_msgs.append(err_msgs[i])
        err_nos = mod_err_nos
        err_msgs = mod_err_msgs
    
    toc = time.time()
    for no, msg, filename in zip(err_nos, err_msgs, filenames):
        rets.append(
            (
                filename, args, no, msg, toc - tic
            )
        )

    if verbose >= 1:
        print("", flush=True)
    return rets


def tg_parallel_builder_build(
    sch_app, params_lst, target, target_host,
    build_func="default", timeout=150, verbose=1, name="main"):
    global GLOBAL_BUILD_INPUTS

    GLOBAL_BUILD_INPUTS = (
        sch_app,
        params_lst,
        build_func,
        target,
        target_host,
        verbose
    )

    with ProcessPool(1) as pool:
        future = pool.map(tg_parallel_build_worker, [name], timeout=timeout)
        iterator = future.result()

        while True:
            try:
                results = next(iterator)
            except StopIteration:
                break
            except TimeoutError as error:
                if verbose >= 1:
                    print("Build Timeout.", flush=True)
                results = [
                    (None, [],
                     auto_scheduler.measure.MeasureErrorNo.BUILD_TIMEOUT,
                     None, timeout) for i in range(len(params_lst))]
            except Exception as error:
                if verbose >= 1:
                    print("Build Fatal Error\n",
                        auto_scheduler.measure.make_error_msg(), flush=True)
                results = [
                    (None, [], 
                     auto_scheduler.measure.MeasureErrorNo.COMPILE_HOST,
                     None, timeout) for i in range(len(params_lst))]
            
        results = [auto_scheduler.measure.BuildResult(*x) for x in results]

    return results


@register_test
def test4():
    print("##########################")
    print("Test 4")
    recipe = at.WMMAFp16Fp32()
    compute_key = "nnn"
    shape_key = "8x32x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, bias, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1)
    target_dag = at.compute_dag_from_tensors([E])

    inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    func_ref = tvm.build(sch_ref, inputs_ref +
                         list(target_dag.tensors), "llvm")
    ctx = tvm.cpu()
    inputs_np_arrays = get_np_arrays(inputs_ref)
    inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
    outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[0].axis
    rc, rr, rs = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [n, n, n, p, p, q, q],
        jj: [k, k, k, k, k, k, k],
        kk: [rc, rr, rs, rc, rs, rc, rr]
    }
    match_result = at.IntrinMatchResult(
        recipe, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    gen = at.TransformGenerator(match_result)
    beg = time.time()
    for i in range(1):
        record = gen.get(policy="random")
        record.unfold_choice = ([1, 1, 1, 1, 1, 1, 1], record.unfold_choice[1])
 
        print("transform decision:")
        for k, v in record.to_json().items():
            print(k, "=", v)

        app = at.TransformApplier(match_result)
        new_state = app.apply(record)

        schedule_gen = at.CUDAScheduleGenerator(match_result, new_state)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.CUDAScheduleApplier(match_result, sc_info)
        params_lst = []
        trials = 32
        print("trials=", trials)
        for j in range(trials):
            params = schedule_gen.get(policy="random")
            # my_params = {
            #     'vectorize': (1, 1),
            #     'spatial_factors': [([1, 1, 1], (0, 0)), ([4, 1, 1], (-1, 1)), ([14, 1, 1], (-1, -1))],
            #     'reduce_factors': [([3, 3, 4], (1, 1)), ([1, 1, 3], (0, -1))],
            #     'last_factors': [([-1, 32], (-1,))],
            #     'output_unroll_step': (64, -1),
            #     'last_unroll_step': (512, 1)}
            # params.from_json(my_params)
            params_lst.append(params)

        build_func = "default"
        target_host = "llvm"
        timeout = 150
        verbose = 1

        build_results = tg_parallel_builder_build(
            schedule_app,
            params_lst,
            recipe.target,
            target_host,
            timeout=timeout,
            build_func=build_func,
            verbose=verbose)

        for r in build_results:
            print(r)
        
        number = 1
        repeat = 1
        min_repeat_ms = 150
        cooldown_interval = 1
        enable_cpu_cache_flush = 1

        run_results = pebble_local_runner_run(
            recipe.target,
            0,
            build_results,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
            verbose
        )

        for r in run_results:
            print(r)

        gen.feedback(record, np.random.random())
    end = time.time()
    print("Pass %f seconds." % (end - beg))
    print("Pass!\n")


@register_test
def test5():
    print("##########################")
    print("Test 5")
    recipe = at.WMMAFp16Fp32()
    compute_key = "nnn"
    shape_key = "16x16x16"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1, with_bias=False)
    target_dag = at.compute_dag_from_tensors([E])

    # inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    print(tvm.lower(sch_ref, [A, B, E], simple_mode=True))
    # func_ref = tvm.build(sch_ref, inputs_ref +
    #                      list(target_dag.tensors), "llvm")
    # ctx = tvm.cpu()
    # inputs_np_arrays = get_np_arrays(inputs_ref)
    # inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
    # outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    # func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
    }
    ii, jj = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[0].axis
    rc, rr, rs = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [n, n, n, p, p, q, q],
        jj: [k, k, k, k, k, k, k],
        kk: [rc, rr, rs, rc, rs, rc, rr]
    }
    match_result = at.IntrinMatchResult(
        recipe, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    gen = at.TransformGenerator(match_result)
    beg = time.time()
    for i in range(1):
        record = gen.get(policy="random")
        record.unfold_choice = ([1, 1, 1, 1, 1, 1, 1], record.unfold_choice[1])
 
        print("transform decision:")
        for k, v in record.to_json().items():
            print(k, "=", v)

        app = at.TransformApplier(match_result)
        new_state = app.apply(record)

        schedule_gen = at.CUDAScheduleGeneratorSplitK(match_result, new_state)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.CUDAScheduleApplierSplitK(match_result, sc_info)
        params_lst = []
        trials = 1
        print("trials=", trials)
        for j in range(trials):
            params = schedule_gen.get(policy="random")
            my_params = {
                'split_K': (4, 0),
                'inline': (0, 1),
                'vectorize': (1, 1),
                'spatial_factors': [([2, 1, 1, 2], (0, 0)), ([4, 1, 1, 2], (-1, 1)), ([14, 1, 1, 1], (-1, -1))],
                'reduce_factors': [([3, 2, 2], (1, 1)), ([1, 1, 3], (0, -1))],
                'last_factors': [([-1, 32], (-1,))],
                'output_unroll_step': (64, -1),
                'last_unroll_step': (512, 1)}
            params.from_json(my_params)
            params_lst.append(params)

        build_func = "default"
        target_host = "llvm"
        timeout = 150
        verbose = 1

        # build_results = tg_parallel_builder_build(
        #     schedule_app,
        #     params_lst,
        #     recipe.target,
        #     target_host,
        #     timeout=timeout,
        #     build_func=build_func,
        #     verbose=verbose)

        build_results = pebble_local_builder_build(
            schedule_app,
            params_lst,
            recipe.target,
            target_host,
            timeout,
            1,
            build_func=build_func)

        for r in build_results:
            print(r)
        
        number = 1
        repeat = 1
        min_repeat_ms = 150
        cooldown_interval = 1
        enable_cpu_cache_flush = 1

        run_results = pebble_local_runner_run(
            recipe.target,
            0,
            build_results,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
            verbose
        )

        for r in run_results:
            print(r)

        gen.feedback(record, np.random.random())
    end = time.time()
    print("Pass %f seconds." % (end - beg))
    print("Pass!\n")


@register_test
def test6():
    print("##########################")
    print("Test 6")
    recipe = at.AVX512SkylakeGemvRecipe()
    compute_key = "dummy"
    shape_key = "16x4"
    intrin_dag, _ = recipe.get_effective_compute_dag(compute_key, shape_key)
    A, B, E = conv2d(1, 128, 14, 14, 64, 3, 3, 1, 1, with_bias=False, in_dtype=["uint8", "int8"], out_dtype="int32")
    target_dag = at.compute_dag_from_tensors([E])

    # inputs_ref = target_dag.get_inputs()
    sch_ref = tvm.te.create_schedule([x.op for x in target_dag.tensors])
    print(tvm.lower(sch_ref, [A, B, E], simple_mode=True))
    # func_ref = tvm.build(sch_ref, inputs_ref +
    #                      list(target_dag.tensors), "llvm")
    # ctx = tvm.cpu()
    # inputs_np_arrays = get_np_arrays(inputs_ref)
    # inputs_arrays = get_tvm_arrays_from_np_arrays(inputs_np_arrays, ctx)
    # outputs_arrays_ref = get_tvm_arrays(list(target_dag.tensors), ctx)
    # func_ref(*inputs_arrays, *outputs_arrays_ref)

    main_op_map = {
        intrin_dag.op_lst[0]: target_dag.op_lst[0]
    }
    elem_op_map = {
    }
    ii, = intrin_dag.op_lst[0].axis
    kk, = intrin_dag.op_lst[0].reduce_axis
    n, k, p, q = target_dag.op_lst[0].axis
    rc, rr, rs = target_dag.op_lst[0].reduce_axis
    axis_map = {
        ii: [k, k, k],
        kk: [rc, rr, rs]
    }
    match_result = at.IntrinMatchResult(
        recipe, compute_key, shape_key,
        main_op_map, elem_op_map,
        axis_map, target_dag, intrin_dag
    )

    gen = at.TransformGenerator(match_result)
    beg = time.time()
    for i in range(1):
        record = gen.get(policy="random")
        record.unfold_choice = ([1, 1, 1], record.unfold_choice[1])
 
        print("transform decision:")
        for k, v in record.to_json().items():
            print(k, "=", v)

        app = at.TransformApplier(match_result)
        new_state = app.apply(record)

        schedule_gen = at.LLVMScheduleGenerator(match_result, new_state)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = at.LLVMScheduleApplier(match_result, sc_info)
        params_lst = []
        trials = 1
        print("trials=", trials)
        for j in range(trials):
            params = schedule_gen.get(policy="random")
            my_params = {
                'inline': (0, 1),
                'vectorize': (1, 1),
                'spatial_factors': [([2, 1], (0, 0)), ([4, 1], (-1, 1)), ([14, 1], (-1, -1)), ([14, 1], (-1, -1))],
                'reduce_factors': [([3, 2], (1, 1)), ([1, 1, 3], (0, -1))],
                'last_factors': [([-1, 32], (-1,))],
                }
            params.from_json(my_params)
            params_lst.append(params)

        build_func = "default"
        target_host = "llvm -mcpu=skylake-avx512"
        timeout = 150
        verbose = 1

        # build_results = tg_parallel_builder_build(
        #     schedule_app,
        #     params_lst,
        #     recipe.target,
        #     target_host,
        #     timeout=timeout,
        #     build_func=build_func,
        #     verbose=verbose)

        build_results = pebble_local_builder_build(
            schedule_app,
            params_lst,
            recipe.target,
            target_host,
            timeout,
            1,
            build_func=build_func)

        for r in build_results:
            print(r)
        
        number = 1
        repeat = 1
        min_repeat_ms = 150
        cooldown_interval = 1
        enable_cpu_cache_flush = 1

        run_results = pebble_local_runner_run(
            recipe.target,
            0,
            build_results,
            timeout,
            number,
            repeat,
            min_repeat_ms,
            cooldown_interval,
            enable_cpu_cache_flush,
            verbose
        )

        for r in run_results:
            print(r)

        gen.feedback(record, np.random.random())
    end = time.time()
    print("Pass %f seconds." % (end - beg))
    print("Pass!\n")


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
