import os
import time
import math
from .search.measure import MAX_FLOAT
import tvm
import tvm._ffi
from .search import pebble_local_builder_build, pebble_local_runner_run
from .tensorization_phases import get_match_results, MappingGenerator, MappingApplier
from .tensorization_phases import (
    CUDAScheduleGenerator,
    CUDAScheduleApplier,
    CUDAScheduleGeneratorV2,
    CUDAScheduleApplierV2,
    CUDAScheduleGeneratorV3,
    CUDAScheduleApplierV3,
    CUDAScheduleGeneratorSplitK,
    CUDAScheduleApplierSplitK,
    CUDAScheduleGeneratorTenet,
    CUDAScheduleApplierTenet,
)
from .tensorization_phases import CUDAScheduleGeneratorMultiReduce, CUDAScheduleApplierMultiReduce
from .tensorization_phases import MaliScheduleGenerator, MaliScheduleApplier
from .tensorization_phases import LLVMScheduleGenerator, LLVMScheduleApplier
from .tensorization_phases import TenetScheduleGenerator, TenetScheduleApplier
from .search import (
    EmptyChecker,
    CUDAProgramChecker,
    MaliProgramChecker,
    find_optimized_parameters,
    find_optimized_parameters_v2,
    find_optimized_parameters_v3,
)
from .target import get_cuda_compute_version
from .policy import first_fit, best_fit, all_fit, choose_one


class AutoTensorizeResult(object):
    def __init__(self, sch_gen=None, sch_app=None, params=None, perf=None, mapping=None):
        self.sch_gen = sch_gen
        self.sch_app = sch_app
        self.params = params
        self.perf = perf
        self.mapping = mapping

    def defined(self):
        return (
            (self.sch_gen is not None)
            and (self.sch_app is not None)
            and (self.params is not None)
            and (self.perf is not None)
        )


def auto_tensorize_compute(
    target_dag,
    target,
    log_file,
    measure_opt,
    verbose=False,
    transform_dump=False,
    transform_strict=True,
    drop_output=False,
    transform_policy="all_fit",
):
    # refactor target
    measure_opt.target = target
    if str(target).startswith("tenet"):
        parts = str(target).split(" ")
        if parts[1] == "cuda":
            match_results = get_match_results(target_dag, "cuda")
        else:
            match_results = get_match_results(target_dag, target)
    else:
        match_results = get_match_results(target_dag, target)

    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target: %s" % target, flush=True)
        return None, None
    print("Possible matchings:", flush=True)
    for i, m in enumerate(match_results):
        print(i, ":", str(m), flush=True)

    if transform_policy == "all_fit":
        match_result, record = all_fit(match_results)
    elif transform_policy == "first_fit":
        match_result, record = first_fit(match_results)
    elif transform_policy == "best_fit":
        match_result, record = best_fit(match_results)
    elif transform_policy[:7] == "choose:":
        suffix = transform_policy[7:]
        mat_id, map_id = suffix.split(",")
        mat_id = int(mat_id)
        map_id = int(map_id)
        match_result, record = choose_one(match_results, mat_id, map_id)
    else:
        raise RuntimeError("Unknown transform policy: %s" % transform_policy)
    print("Selected matching:", str(match_result), flush=True)
    print("Axis mapping:", flush=True)
    for i, v in match_result.axis_map.items():
        print(i.var, ":", [x.var for x in v], flush=True)
    print("Selected mapping:", str(record), flush=True)
    app = MappingApplier(match_result, verbose=transform_dump, strict=transform_strict)
    new_state = app.apply(record, drop_output=drop_output)

    if transform_dump:
        print("Dump IR after transform:", flush=True)
        new_target_dag = new_state.target_dag
        new_inputs = new_target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
        print(
            tvm.lower(sch, new_inputs + list(new_target_dag.tensors), simple_mode=True), flush=True
        )

    return match_result, new_state


def auto_tensorize_schedule(
    target_dag,
    target,
    log_file,
    measure_opt,
    match_result,
    new_state,
    trials=200,
    builder=pebble_local_builder_build,
    runner=pebble_local_runner_run,
    verbose=False,
    search_group_size=16,
    enable_split_K=False,
    use_lagacy=False,
    build_parallel=1,
    run_parallel=1,
):
    if match_result is None or new_state is None:
        return AutoTensorizeResult(None, None, None, None)
    if str(target) == "cuda":
        if enable_split_K:
            schedule_gen = CUDAScheduleGeneratorSplitK(
                match_result,
                new_state,
                log_file=log_file,
                arch=get_cuda_compute_version(measure_opt.dev_id),
            )
            if os.path.exists(log_file) and os.path.isfile(log_file):
                schedule_gen.load_from_file(log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = CUDAScheduleApplierSplitK(match_result, sc_info)
            # relaxed checker for split K
            checker = EmptyChecker()
        elif use_lagacy:
            schedule_gen = CUDAScheduleGenerator(
                match_result,
                new_state,
                log_file=log_file,
                arch=get_cuda_compute_version(measure_opt.dev_id),
            )
            if os.path.exists(log_file) and os.path.isfile(log_file):
                schedule_gen.load_from_file(log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = CUDAScheduleApplier(match_result, sc_info)
            checker = CUDAProgramChecker(arch=get_cuda_compute_version(measure_opt.dev_id))
        else:
            schedule_gen = CUDAScheduleGeneratorV2(
                match_result,
                new_state,
                log_file=log_file,
                arch=get_cuda_compute_version(measure_opt.dev_id),
            )
            if os.path.exists(log_file) and os.path.isfile(log_file):
                schedule_gen.load_from_file(log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = CUDAScheduleApplierV2(match_result, sc_info)
            checker = CUDAProgramChecker(arch=get_cuda_compute_version(measure_opt.dev_id))
    elif str(target) == "opencl":
        schedule_gen = MaliScheduleGenerator(match_result, new_state, log_file=log_file)
        if os.path.exists(log_file) and os.path.isfile(log_file):
            schedule_gen.load_from_file(log_file)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = MaliScheduleApplier(match_result, sc_info)
        # TODO: write a checker for MALI GPU
        checker = MaliProgramChecker(arch="g76")
    elif str(target) == "llvm -mcpu=skylake-avx512":
        schedule_gen = LLVMScheduleGenerator(match_result, new_state, log_file=log_file)
        if os.path.exists(log_file) and os.path.isfile(log_file):
            schedule_gen.load_from_file(log_file)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = LLVMScheduleApplier(match_result, sc_info)
        # TODO: write a checker for CPU
        checker = EmptyChecker()
    elif str(target).startswith("tenet"):
        target = str(target)
        parts = target.split(" ")
        assert len(parts) > 1
        if parts[1] == "cuda":
            schedule_gen = CUDAScheduleGeneratorTenet(
                match_result,
                new_state,
                log_file=log_file,
                arch=get_cuda_compute_version(measure_opt.dev_id),
            )
            if os.path.exists(log_file) and os.path.isfile(log_file):
                schedule_gen.load_from_file(log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = CUDAScheduleApplierTenet(match_result, sc_info)
            checker = CUDAProgramChecker(arch=get_cuda_compute_version(measure_opt.dev_id))
        else:
            schedule_gen = TenetScheduleGenerator(match_result, new_state, log_file=log_file)
            if os.path.exists(log_file) and os.path.isfile(log_file):
                schedule_gen.load_from_file(log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = TenetScheduleApplier(match_result, sc_info)
            # TODO: write a checker for TENET
            checker = EmptyChecker()
    else:
        raise RuntimeError("Do not support target: %s" % target)

    # use tuning to find params
    if trials:
        value, params = find_optimized_parameters(
            match_result,
            schedule_gen,
            schedule_app,
            measure_opt,
            checker,
            trials,  # policy="random",
            builder=builder,
            runner=runner,
            verbose=verbose,
            search_group_size=search_group_size,
            build_parallel=build_parallel,
            run_parallel=run_parallel,
        )

    entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in file
    params, value = entry.record, 1 / entry.value
    # print("Evaluation only:", params, value, flush=True)

    return AutoTensorizeResult(schedule_gen, schedule_app, params, value)


def auto_tensorize(
    target_dag,
    target,
    log_file,
    measure_opt,
    trials=200,
    builder=pebble_local_builder_build,
    runner=pebble_local_runner_run,
    verbose=False,
    transform_dump=False,
    transform_strict=True,
    drop_output=False,
    transform_policy="all_fit",
    search_group_size=16,
    enable_split_K=False,
    build_parallel=1,
    run_parallel=1,
):
    print(
        "[AMOS] Mapping starts...\nUsing deterministic mapping logic with dynamic schedule tuning",
        flush=True,
    )
    match_result, new_state = auto_tensorize_compute(
        target_dag,
        target,
        log_file,
        measure_opt,
        verbose,
        transform_dump,
        transform_strict,
        drop_output,
        transform_policy,
    )

    return auto_tensorize_schedule(
        target_dag,
        target,
        log_file,
        measure_opt,
        match_result,
        new_state,
        trials,
        builder,
        runner,
        verbose,
        search_group_size,
        enable_split_K,
        build_parallel=build_parallel,
        run_parallel=run_parallel,
    )


def get_schedule(sch_app, params):
    target_dag = sch_app.target_dag
    inputs = target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])

    args = inputs + list(target_dag.tensors)
    sch = sch_app.apply(sch, params)
    return sch, args


def auto_tensorize_v2(
    target_dag,
    target,
    log_file,
    measure_opt,
    trials=200,
    builder=pebble_local_builder_build,
    runner=pebble_local_runner_run,
    verbose=False,
    transform_dump=False,
):
    """
    Warning: this function is depricated
    """
    # refactor target
    measure_opt.target = target
    match_results = get_match_results(target_dag, target)

    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target: %s" % target, flush=True)
        return AutoTensorizeResult(None, None, None, None)
    elif verbose:
        print("Matched results:", flush=True)
        for m in match_results:
            print(str(m), flush=True)
    # here is match intrin policy
    match_result = match_results[0]
    if verbose:
        print("Selected:", str(match_result), flush=True)
        print("Axis map:", flush=True)
        for k, v in match_result.axis_map.items():
            print(k, ":", v, flush=True)

    gen = MappingGenerator(match_result)
    record = gen.get(policy="random")
    # here is transform policy
    record.vmap_choice = ([1 for _ in record.vmap_choice[0]], record.vmap_choice[1])
    app = MappingApplier(match_result, verbose=transform_dump)
    new_state = app.apply(record)

    if transform_dump:
        print("Dump IR after transform:", flush=True)
        new_target_dag = new_state.target_dag
        new_inputs = new_target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
        print(
            tvm.lower(sch, new_inputs + list(new_target_dag.tensors), simple_mode=True), flush=True
        )

    if str(target) == "cuda":
        schedule_gen = CUDAScheduleGeneratorMultiReduce(match_result, new_state, log_file=log_file)
        if os.path.exists(log_file) and os.path.isfile(log_file):
            schedule_gen.load_from_file(log_file)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = CUDAScheduleApplierMultiReduce(match_result, sc_info)
        checker = CUDAProgramChecker(arch=get_cuda_compute_version(measure_opt.dev_id))
    else:
        raise RuntimeError("Do not support target: %s" % target)

    # use tuning to find params
    if trials:
        value, params = find_optimized_parameters(
            match_result,
            schedule_gen,
            schedule_app,
            measure_opt,
            checker,
            trials,  # policy="random",
            builder=builder,
            runner=runner,
            verbose=verbose,
        )
    else:
        entry = schedule_gen.get_best_entry()
        # we store 1/time_cost in file
        params, value = entry.record, 1 / entry.value

    return AutoTensorizeResult(schedule_gen, schedule_app, params, value)


def auto_tensorize_v3(
    target_dag,
    target,
    transform_log_file,
    schedule_log_file,
    measure_opt,
    trials=200,
    schedule_trials=40,
    builder=pebble_local_builder_build,
    runner=pebble_local_runner_run,
    verbose=False,
    verbose_schedule=False,
    transform_dump=False,
    transform_strict=True,
    search_group_size=5,
    desired_compute_key=None,
    desired_shape_key=None,
    enable_split_K=False,
    use_shared_store=False,
    drop_output=False,
    build_parallel=1,
    run_parallel=1,
):

    measure_opt.target = target
    match_results = get_match_results(target_dag, target)

    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target: %s" % target, flush=True)
        return AutoTensorizeResult()
    elif verbose:
        print("Matched results:", flush=True)
        for m in match_results:
            print(str(m), flush=True)

    compute_key_match_results = []
    if desired_compute_key is not None:
        for m in match_results:
            if m.compute_key == desired_compute_key:
                compute_key_match_results.append(m)
    else:
        compute_key_match_results = match_results
    if len(compute_key_match_results) == 0:
        print("No match result matches desired compute key:", desired_compute_key, flush=True)
        return AutoTensorizeResult()
    shape_key_match_results = []
    if desired_shape_key is not None:
        for m in compute_key_match_results:
            if m.shape_key == desired_shape_key:
                shape_key_match_results.append(m)
    else:
        shape_key_match_results = compute_key_match_results
    if len(shape_key_match_results) == 0:
        print("No match result matches desired shape key:", desired_shape_key, flush=True)
        return AutoTensorizeResult()

    match_result = shape_key_match_results[0]
    if verbose:
        print("Selected:", str(match_result), flush=True)
        print("Axis map:", flush=True)
        for k, v in match_result.axis_map.items():
            print(k, ":", v, flush=True)

    gen = MappingGenerator(match_result, log_file=transform_log_file, allow_repeat=True)
    if os.path.exists(transform_log_file) and os.path.isfile(transform_log_file):
        gen.load_from_file(transform_log_file)
    app = MappingApplier(match_result, verbose=transform_dump, strict=transform_strict)

    class ScheduleContext:
        def __init__(self, schedule_gen, schedule_app, sc_info, checker, generate_schedule):
            self.schedule_gen = schedule_gen
            self.schedule_app = schedule_app
            self.sc_info = sc_info
            self.checker = checker
            self.generate_schedule = generate_schedule

    schedule_context_cache = {}
    best_value = 1 / MAX_FLOAT
    best_ctx = None
    best_params = None
    pure_test = False

    iterations = trials // schedule_trials
    print("Total iterations:", iterations, flush=True)
    print("Trials per iteration:", schedule_trials, flush=True)
    if iterations == 0:
        iterations = 1
        schedule_trials = 0
        pure_test = True
        print("Pure testing mode...", flush=True)
    beg = time.time()
    for it in range(iterations):
        if not pure_test:
            feasible = False
            while not feasible:
                record = gen.get_next(policy="random")
                try:
                    tmp_app = MappingApplier(match_result, strict=transform_strict)
                    tmp_app.apply(record, drop_output=drop_output)
                    feasible = True
                except RuntimeError as e:
                    print("Catch an infeasible mapping:", flush=True)
                    print(record, flush=True)

        else:
            try:
                entry = gen.get_best_entry()
                record = entry.record
                print("Best Mapping:", flush=True)
                print(record, flush=True)
            except Exception as e:
                raise RuntimeError("Can't get previous results for test mode.")
        print(f"Choose transform: {record}", flush=True)
        new_state = app.apply(record, drop_output=drop_output)

        if transform_dump:
            print("Dump IR after transform:", flush=True)
            new_target_dag = new_state.target_dag
            new_inputs = new_target_dag.get_inputs()
            sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
            print(
                tvm.lower(sch, new_inputs + list(new_target_dag.tensors), simple_mode=True),
                flush=True,
            )

        record_key = record.as_key()
        if record_key in schedule_context_cache:
            sch_ctx = schedule_context_cache[record_key]
        else:
            current_log_file = str(record_key) + "_" + schedule_log_file
            if str(target) == "cuda":
                if not enable_split_K:
                    if use_shared_store:
                        raise NotImplementedError()
                        # schedule_gen = CUDAScheduleGeneratorV3(
                        #     match_result, new_state, log_file=current_log_file,
                        #     arch=get_cuda_compute_version(measure_opt.dev_id))
                        # if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                        #     schedule_gen.load_from_file(current_log_file)
                        # sc_info = schedule_gen.get_schedule_compute_info()
                        # schedule_app = CUDAScheduleApplierV3(
                        #     match_result, sc_info)
                    else:
                        schedule_gen = CUDAScheduleGeneratorV2(
                            match_result,
                            new_state,
                            log_file=current_log_file,
                            arch=get_cuda_compute_version(measure_opt.dev_id),
                        )
                        if verbose:
                            print(f"All mappings: {schedule_gen.size()}", flush=True)
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = CUDAScheduleApplierV2(match_result, sc_info)
                else:
                    schedule_gen = CUDAScheduleGeneratorSplitK(
                        match_result,
                        new_state,
                        log_file=current_log_file,
                        arch=get_cuda_compute_version(measure_opt.dev_id),
                    )
                    if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                        schedule_gen.load_from_file(current_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = CUDAScheduleApplierSplitK(match_result, sc_info)
                checker = CUDAProgramChecker(arch=get_cuda_compute_version(measure_opt.dev_id))
            elif str(target) == "opencl":
                schedule_gen = MaliScheduleGenerator(
                    match_result, new_state, log_file=current_log_file
                )
                if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                    schedule_gen.load_from_file(current_log_file)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = MaliScheduleApplier(match_result, sc_info)
                # TODO: write a checker for MALI GPU
                checker = MaliProgramChecker(arch="g76")
            elif str(target) == "llvm -mcpu=skylake-avx512":
                schedule_gen = LLVMScheduleGenerator(
                    match_result, new_state, log_file=current_log_file
                )
                if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                    schedule_gen.load_from_file(current_log_file)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = LLVMScheduleApplier(match_result, sc_info)
                # TODO: write a checker for CPU
                checker = EmptyChecker()
            elif str(target).startswith("tenet"):
                target = str(target)
                parts = target.split(" ")
                assert len(parts) > 1
                if parts[1] == "cuda":
                    schedule_gen = CUDAScheduleGeneratorTenet(
                        match_result,
                        new_state,
                        log_file=current_log_file,
                        arch=get_cuda_compute_version(measure_opt.dev_id),
                    )
                    if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                        schedule_gen.load_from_file(current_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = CUDAScheduleApplierTenet(match_result, sc_info)
                    checker = CUDAProgramChecker(arch=get_cuda_compute_version(measure_opt.dev_id))
                else:
                    schedule_gen = TenetScheduleGenerator(
                        match_result, new_state, log_file=current_log_file
                    )
                    if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                        schedule_gen.load_from_file(current_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = TenetScheduleApplier(match_result, sc_info)
                    # TODO: write a checker for TENET
                    checker = EmptyChecker()
            else:
                raise RuntimeError("Do not support target: %s" % target)

            # use tuning to find params
            if schedule_trials:
                generate_schedule = find_optimized_parameters_v2(
                    match_result,
                    schedule_gen,
                    schedule_app,
                    measure_opt,
                    checker,
                    schedule_trials,  # policy="random",
                    builder=builder,
                    runner=runner,
                    verbose=verbose_schedule,
                    search_group_size=search_group_size,
                    build_parallel=build_parallel,
                    run_parallel=run_parallel,
                )
            else:
                generate_schedule = None

            sch_ctx = ScheduleContext(
                schedule_gen, schedule_app, sc_info, checker, generate_schedule
            )
            schedule_context_cache[record_key] = sch_ctx

        if sch_ctx.generate_schedule is not None:
            value, params = next(sch_ctx.generate_schedule)
        try:
            entry = sch_ctx.schedule_gen.get_best_entry()
            # we store 1/time_cost in file
            params, value = entry.record, entry.value
            # print("Evaluation only:", params, value, flush=True)
            if not pure_test:
                gen.feedback(record, value)
        except Exception as e:
            params = None
            value = 1 / MAX_FLOAT

        # record the best
        if value > best_value:
            best_value = value
            best_ctx = sch_ctx
            best_params = params

        print(f"Record:{1/best_value} G cycle", flush=True)
        print(
            f"Iteration: {it+1}: {value}/{best_value}({1/best_value*1e3} ms), {str(record)}, {str(params)}",
            flush=True,
        )

        if (it + 1) % 10 == 0:
            print("Show transformation explore summary:", flush=True)
            for k, v in schedule_context_cache.items():
                print(f"{str(k)}: {v.schedule_gen.num_entries()}", flush=True)

    end = time.time()
    print(f"Tensorize use time {(end - beg)} s.", flush=True)
    return AutoTensorizeResult(
        best_ctx.schedule_gen, best_ctx.schedule_app, best_params, 1 / best_value
    )


def auto_tensorize_v4(
    target_dag,
    target,
    schedule_log_file,
    measure_opt,
    schedule_log_dir="schedules",
    trials=200,
    repeat_rounds=10,
    builder=pebble_local_builder_build,
    runner=pebble_local_runner_run,
    verbose_schedule=False,
    transform_dump=False,
    transform_strict=True,
    search_group_size=5,
    desired_compute_key=None,
    desired_shape_key=None,
    enable_split_K=False,
    use_shared_store=False,
    drop_output=False,
    build_parallel=1,
    run_parallel=1,
    explore_full_match=False,
    enable_perf_model=False,
    perf_percentage=0.5,
):

    measure_opt.target = target
    match_results = get_match_results(target_dag, target)

    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target: %s" % target, flush=True)
        return AutoTensorizeResult()
    else:
        print("Possible matchings:", flush=True)
        for i, m in enumerate(match_results):
            print(i, ":", str(m), flush=True)

    compute_key_match_results = []
    if desired_compute_key is not None:
        for m in match_results:
            if m.compute_key == desired_compute_key:
                compute_key_match_results.append(m)
    else:
        compute_key_match_results = match_results
    if len(compute_key_match_results) == 0:
        print("No match result matches desired compute key:", desired_compute_key, flush=True)
        return AutoTensorizeResult()
    shape_key_match_results = []
    if desired_shape_key is not None:
        for m in compute_key_match_results:
            if m.shape_key == desired_shape_key:
                shape_key_match_results.append(m)
    else:
        shape_key_match_results = compute_key_match_results
    if len(shape_key_match_results) == 0:
        print("No match result matches desired shape key:", desired_shape_key, flush=True)
        return AutoTensorizeResult()

    # Here we are supposed to use MappingGenerator to
    # search for a good transformation.
    # However, we expose the searching logic directly here.
    # TODO: merge the following logics into MappingGenerator
    all_matches = []
    all_mappings = []
    appliers = []
    mapping_weights = []
    weights_updates = []
    momentum = 0.8
    if not explore_full_match:
        # use all_fit logic to choose the one with minimum padding
        match_result, _ = all_fit(shape_key_match_results)
        shape_key_match_results = [match_result]
    total_matchings = 0
    total_mappings = 0
    for match_result in shape_key_match_results:
        all_matches.append(match_result)
        gen = MappingGenerator(match_result)
        mappings = gen.get_all()
        # filter out infeasible mappings
        feasible_mappings = []
        tmp_app = MappingApplier(match_result, strict=transform_strict)
        for mapping in mappings:
            try:
                tmp_app.apply(mapping, drop_output=drop_output)
                feasible_mappings.append(mapping)
            except RuntimeError as e:
                print("Catch an infeasible mapping:", flush=True)
                print(mapping, flush=True)
        mappings = feasible_mappings
        # record the feasible mappings
        all_mappings.append(mappings)
        total_matchings += 1
        assert len(mappings) > 0
        total_mappings += len(mappings)
        mapping_weights.append([1.0 / len(mappings) for m in mappings])
        weights_updates.append([0.0 for m in mappings])
        app = MappingApplier(match_result, verbose=transform_dump, strict=transform_strict)
        appliers.append(app)
    if total_mappings == 0:
        print("Can't find any mappings!", flush=True)
        return AutoTensorizeResult()

    class ScheduleContext:
        def __init__(self, schedule_gen, schedule_app, sc_info, checker, generate_schedule):
            self.schedule_gen = schedule_gen
            self.schedule_app = schedule_app
            self.sc_info = sc_info
            self.checker = checker
            self.generate_schedule = generate_schedule

    # global context for overall exploration
    schedule_context_cache = {}
    best_value = 1 / MAX_FLOAT
    best_ctx = None
    best_params = None
    best_mapping = None
    pure_test = False

    if trials == 0:
        # pure test mode, no tuning
        pure_test = True
        repeat_rounds = 1
        print("Pure testing mode...", flush=True)
    elif trials < total_mappings * repeat_rounds * search_group_size:
        print(
            f"[Warning] Too few trials, expect at least {total_mappings * repeat_rounds * search_group_size} trials.",
            flush=True,
        )
        trials = total_mappings * repeat_rounds * search_group_size
        print(
            f"Increase trials to {total_mappings * repeat_rounds * search_group_size}.", flush=True
        )
    else:
        print("Total trials:", trials, flush=True)
    trials_per_matching = trials // repeat_rounds // total_matchings

    print("Num rounds:", repeat_rounds, flush=True)
    print("Num matching:", total_matchings, flush=True)
    print("Num mapping:", total_mappings, flush=True)
    print("Initial trials per matching:", trials_per_matching, flush=True)

    if not (os.path.exists(schedule_log_dir) and os.path.isdir(schedule_log_dir)):
        os.mkdir(schedule_log_dir)
    beg = time.time()
    for round in range(repeat_rounds):
        for match_id in range(total_matchings):
            match_result = all_matches[match_id]
            app = appliers[match_id]
            weights = mapping_weights[match_id]
            updates = weights_updates[match_id]
            tune_trials = [math.ceil(trials_per_matching * x) for x in weights]
            best_values_of_mappings = []
            print("Original weights", weights, flush=True)
            print("Original trials for each mapping", tune_trials, flush=True)
            print("Current explored matching:", str(match_result), flush=True)
            print("Its axis mapping:", flush=True)
            for i, v in match_result.axis_map.items():
                print(i.var, ":", [x.var for x in v], flush=True)
            for mapping_id in range(len(all_mappings[match_id])):
                record = all_mappings[match_id][mapping_id]
                print("Current explored mapping:", str(record), flush=True)

                # transform compute
                new_state = app.apply(record, drop_output=drop_output)
                # prepare tune log file
                record_key = record.as_key()
                current_log_file = os.path.join(
                    schedule_log_dir, "mapping_" + str(record_key) + "_" + schedule_log_file
                )
                if record_key in schedule_context_cache:
                    sch_ctx = schedule_context_cache[record_key]
                else:
                    if str(target) == "cuda":
                        if not enable_split_K:
                            if use_shared_store:
                                raise NotImplementedError()
                            else:
                                if enable_perf_model:
                                    schedule_gen = CUDAScheduleGeneratorV3(
                                        match_result,
                                        new_state,
                                        log_file=current_log_file,
                                        arch=get_cuda_compute_version(measure_opt.dev_id),
                                    )
                                    if os.path.exists(current_log_file) and os.path.isfile(
                                        current_log_file
                                    ):
                                        schedule_gen.load_from_file(current_log_file)
                                    sc_info = schedule_gen.get_schedule_compute_info()
                                    schedule_app = CUDAScheduleApplierV3(match_result, sc_info)
                                else:
                                    schedule_gen = CUDAScheduleGeneratorV2(
                                        match_result,
                                        new_state,
                                        log_file=current_log_file,
                                        arch=get_cuda_compute_version(measure_opt.dev_id),
                                    )
                                    if os.path.exists(current_log_file) and os.path.isfile(
                                        current_log_file
                                    ):
                                        schedule_gen.load_from_file(current_log_file)
                                    sc_info = schedule_gen.get_schedule_compute_info()
                                    schedule_app = CUDAScheduleApplierV2(match_result, sc_info)
                        else:
                            if enable_perf_model:
                                raise NotImplementedError()
                            else:
                                schedule_gen = CUDAScheduleGeneratorSplitK(
                                    match_result,
                                    new_state,
                                    log_file=current_log_file,
                                    arch=get_cuda_compute_version(measure_opt.dev_id),
                                )
                                if os.path.exists(current_log_file) and os.path.isfile(
                                    current_log_file
                                ):
                                    schedule_gen.load_from_file(current_log_file)
                                sc_info = schedule_gen.get_schedule_compute_info()
                                schedule_app = CUDAScheduleApplierSplitK(match_result, sc_info)
                        checker = CUDAProgramChecker(
                            arch=get_cuda_compute_version(measure_opt.dev_id)
                        )
                    elif str(target) == "opencl":
                        schedule_gen = MaliScheduleGenerator(
                            match_result, new_state, log_file=current_log_file
                        )
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = MaliScheduleApplier(match_result, sc_info)
                        # TODO: write a checker for MALI GPU
                        checker = MaliProgramChecker(arch="g76")
                    elif str(target) == "llvm -mcpu=skylake-avx512":
                        schedule_gen = LLVMScheduleGenerator(
                            match_result, new_state, log_file=current_log_file
                        )
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = LLVMScheduleApplier(match_result, sc_info)
                        # TODO: write a checker for CPU
                        checker = EmptyChecker()
                    elif str(target).startswith("tenet"):
                        target = str(target)
                        parts = target.split(" ")
                        assert len(parts) > 1
                        if parts[1] == "cuda":
                            schedule_gen = CUDAScheduleGeneratorTenet(
                                match_result,
                                new_state,
                                log_file=current_log_file,
                                arch=get_cuda_compute_version(measure_opt.dev_id),
                            )
                            if os.path.exists(current_log_file) and os.path.isfile(
                                current_log_file
                            ):
                                schedule_gen.load_from_file(current_log_file)
                            sc_info = schedule_gen.get_schedule_compute_info()
                            schedule_app = CUDAScheduleApplierTenet(match_result, sc_info)
                            checker = CUDAProgramChecker(
                                arch=get_cuda_compute_version(measure_opt.dev_id)
                            )
                        else:
                            schedule_gen = TenetScheduleGenerator(
                                match_result, new_state, log_file=current_log_file
                            )
                            if os.path.exists(current_log_file) and os.path.isfile(
                                current_log_file
                            ):
                                schedule_gen.load_from_file(current_log_file)
                            sc_info = schedule_gen.get_schedule_compute_info()
                            schedule_app = TenetScheduleApplier(match_result, sc_info)
                            # TODO: write a checker for TENET
                            checker = EmptyChecker()
                    else:
                        raise RuntimeError("Do not support target: %s" % target)

                    # tune loop
                    schedule_trials = tune_trials[mapping_id]
                    if schedule_trials and not pure_test:
                        # this returns a generator
                        if enable_perf_model:
                            generate_schedule = find_optimized_parameters_v3(
                                match_result,
                                schedule_gen,
                                schedule_app,
                                measure_opt,
                                checker,
                                schedule_trials,  # policy="random",
                                builder=builder,
                                runner=runner,
                                verbose=verbose_schedule,
                                search_group_size=search_group_size,
                                build_parallel=build_parallel,
                                run_parallel=run_parallel,
                                perf_percentage=perf_percentage,
                            )
                        else:
                            generate_schedule = find_optimized_parameters_v2(
                                match_result,
                                schedule_gen,
                                schedule_app,
                                measure_opt,
                                checker,
                                schedule_trials,  # policy="random",
                                builder=builder,
                                runner=runner,
                                verbose=verbose_schedule,
                                search_group_size=search_group_size,
                                build_parallel=build_parallel,
                                run_parallel=run_parallel,
                            )
                    else:
                        generate_schedule = None

                    # create new schedule context
                    sch_ctx = ScheduleContext(
                        schedule_gen, schedule_app, sc_info, checker, generate_schedule
                    )
                    schedule_context_cache[record_key] = sch_ctx

                if sch_ctx.generate_schedule is not None:
                    value, params = next(sch_ctx.generate_schedule)
                try:
                    entry = sch_ctx.schedule_gen.get_best_entry()
                    # we store 1/time_cost in file
                    params, value = entry.record, entry.value
                except Exception as e:
                    params = None
                    value = 1 / MAX_FLOAT

                # record the best value of current mapping
                best_values_of_mappings.append(value)

                # record the best
                if value > best_value:
                    best_value = value
                    best_ctx = sch_ctx
                    best_params = params
                    best_mapping = record
                    if transform_dump:
                        print("Dump IR after transform:", flush=True)
                        new_target_dag = new_state.target_dag
                        new_inputs = new_target_dag.get_inputs()
                        sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
                        print(
                            tvm.lower(
                                sch, new_inputs + list(new_target_dag.tensors), simple_mode=True
                            ),
                            flush=True,
                        )

                print(f"Best record value:{best_value} (larger is better)", flush=True)
                print(
                    f"Round {round+1}, Match {match_id+1}, Mapping {mapping_id+1}: {value}/{best_value}({1/best_value*1e3} ms), {str(record)}, {str(params)}",
                    flush=True,
                )
            if not pure_test:
                # redistribute weights according to current best value
                max_value = max(best_values_of_mappings)
                exp_scores = [math.exp(x - max_value) for x in best_values_of_mappings]
                sum_exp_scores = sum(exp_scores)
                new_weights = [x / sum_exp_scores for x in exp_scores]
                delta_weights = [new_weights[i] - weights[i] for i in range(len(weights))]
                new_updates = [
                    delta_weights[i] + momentum * updates[i] for i in range(len(updates))
                ]
                new_weights = [weights[i] + new_updates[i] for i in range(len(new_updates))]
                exp_scores = [math.exp(x) for x in new_weights]
                sum_exp_scores = sum(exp_scores)
                new_weights = [x / sum_exp_scores for x in exp_scores]
                # update into global context
                mapping_weights[match_id] = new_weights
                weights_updates[match_id] = new_updates
                print("New weights", new_weights, flush=True)

        if not pure_test:
            print("Show mapping exploration summary:", flush=True)
            for k, v in schedule_context_cache.items():
                print(
                    f"mapping {str(k)}: explored {v.schedule_gen.num_entries()} schedules",
                    flush=True,
                )
    end = time.time()
    if not pure_test:
        print(f"Mapping exploration uses time {(end - beg)} s.", flush=True)
    return AutoTensorizeResult(
        best_ctx.schedule_gen,
        best_ctx.schedule_app,
        best_params,
        1 / best_value,
        mapping=best_mapping,
    )
