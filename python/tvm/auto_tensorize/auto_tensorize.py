import os
from .utils import bi_product
from .search.measure import MAX_FLOAT
import tvm
import tvm._ffi
from .search import pebble_local_builder_build, pebble_local_runner_run
from .tensorization_phases import get_match_results, TransformGenerator, TransformApplier
from .tensorization_phases import CUDAScheduleGenerator, CUDAScheduleApplier, CUDAScheduleGeneratorV2, CUDAScheduleApplierV2
from .tensorization_phases import CUDAScheduleGeneratorMultiReduce, \
    CUDAScheduleApplierMultiReduce
from .tensorization_phases import MaliScheduleGenerator, MaliScheduleApplier
from .search import CUDAProgramChecker, MaliProgramChecker, find_optimized_parameters, find_optimized_parameters_v2
from .target import get_cuda_compute_version
from .policy import first_fit, best_fit, all_fit


class AutoTensorizeResult(object):
    def __init__(self, sch_gen, sch_app, params, perf):
        self.sch_gen = sch_gen
        self.sch_app = sch_app
        self.params = params
        self.perf = perf

    def defined(self):
        return ((self.sch_gen is not None)
                and (self.sch_app is not None)
                and (self.params is not None)
                and (self.perf is not None))


def auto_tensorize_compute(target_dag, target,
                           log_file,
                           measure_opt,
                           verbose=False,
                           transform_dump=False,
                           transform_policy="all_fit"):
    # refactor target
    measure_opt.target = target
    match_results = get_match_results(target_dag, target)

    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target: %s" %
              target, flush=True)
        return None, None
    elif verbose:
        print("Matched results:", flush=True)
        for m in match_results:
            print(str(m), flush=True)

    if transform_policy == "all_fit":
        match_result, record = all_fit(match_results)
    elif transform_policy == "first_fit":
        match_result, record = first_fit(match_results)
    elif transform_policy == "best_fit":
        match_result, record = best_fit(match_results)
    else:
        raise RuntimeError("Unknown transform policy: %s" % transform_policy)
    if verbose:
        print("Selected:", str(match_result), flush=True)
        print("Axis map:", flush=True)
        for k, v in match_result.axis_map.items():
            print(k, ":", v, flush=True)
    app = TransformApplier(match_result, verbose=transform_dump)
    new_state = app.apply(record)

    if transform_dump:
        print("Dump IR after transform:", flush=True)
        new_target_dag = new_state.target_dag
        new_inputs = new_target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
        print(tvm.lower(
            sch, new_inputs + list(new_target_dag.tensors), simple_mode=True),
            flush=True)

    return match_result, new_state


def auto_tensorize_schedule(target_dag, target,
                            log_file,
                            measure_opt,
                            match_result,
                            new_state,
                            trials=200,
                            builder=pebble_local_builder_build,
                            runner=pebble_local_runner_run,
                            verbose=False,
                            search_batch=16):
    if match_result is None or new_state is None:
        return AutoTensorizeResult(None, None, None, None)
    if str(target) == "cuda":
        schedule_gen = CUDAScheduleGenerator(
            match_result, new_state, log_file=log_file)
        if os.path.exists(log_file) and os.path.isfile(log_file):
            schedule_gen.load_from_file(log_file)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = CUDAScheduleApplier(match_result, sc_info)
        checker = CUDAProgramChecker(
            arch=get_cuda_compute_version(measure_opt.dev_id))
    elif str(target) == "opencl":
        schedule_gen = MaliScheduleGenerator(
            match_result, new_state, log_file=log_file)
        if os.path.exists(log_file) and os.path.isfile(log_file):
            schedule_gen.load_from_file(log_file)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = MaliScheduleApplier(match_result, sc_info)
        # TODO: write a checker for MALI GPU
        checker = MaliProgramChecker(arch="g76")
    else:
        raise RuntimeError("Do not support target: %s" % target)

    # use tuning to find params
    if trials:
        value, params = find_optimized_parameters(
            match_result, schedule_gen, schedule_app,
            measure_opt, checker, trials,  # policy="random",
            builder=builder,
            runner=runner,
            verbose=verbose,
            batch_size=search_batch)

    entry = schedule_gen.get_best_entry()
    # we store 1/time_cost in file
    params, value = entry.record, 1 / entry.value
    # print("Evaluation only:", params, value, flush=True)

    return AutoTensorizeResult(
        schedule_gen,
        schedule_app,
        params,
        value
    )


def auto_tensorize(target_dag, target,
                   log_file,
                   measure_opt,
                   trials=200,
                   builder=pebble_local_builder_build,
                   runner=pebble_local_runner_run,
                   verbose=False,
                   transform_dump=False,
                   transform_policy="all_fit",
                   search_batch=16):
    match_result, new_state = auto_tensorize_compute(
        target_dag,
        target,
        log_file,
        measure_opt,
        verbose,
        transform_dump,
        transform_policy)

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
        search_batch
    )


def get_schedule(sch_app, params):
    target_dag = sch_app.target_dag
    inputs = target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])

    args = inputs + list(target_dag.tensors)
    sch = sch_app.apply(sch, params)
    return sch, args


def auto_tensorize_v2(target_dag, target,
                      log_file,
                      measure_opt,
                      trials=200,
                      builder=pebble_local_builder_build,
                      runner=pebble_local_runner_run,
                      verbose=False,
                      transform_dump=False):
    # refactor target
    measure_opt.target = target
    match_results = get_match_results(target_dag, target)

    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target: %s" %
              target, flush=True)
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

    gen = TransformGenerator(match_result)
    record = gen.get(policy="random")
    # here is transform policy
    record.unfold_choice = (
        [1 for _ in record.unfold_choice[0]], record.unfold_choice[1])
    app = TransformApplier(match_result, verbose=transform_dump)
    new_state = app.apply(record)

    if transform_dump:
        print("Dump IR after transform:", flush=True)
        new_target_dag = new_state.target_dag
        new_inputs = new_target_dag.get_inputs()
        sch = tvm.te.create_schedule([x.op for x in new_target_dag.tensors])
        print(tvm.lower(
            sch, new_inputs + list(new_target_dag.tensors), simple_mode=True),
            flush=True)

    if str(target) == "cuda":
        schedule_gen = CUDAScheduleGeneratorMultiReduce(
            match_result, new_state, log_file=log_file)
        if os.path.exists(log_file) and os.path.isfile(log_file):
            schedule_gen.load_from_file(log_file)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = CUDAScheduleApplierMultiReduce(
            match_result, sc_info)
        checker = CUDAProgramChecker(
            arch=get_cuda_compute_version(measure_opt.dev_id))
    else:
        raise RuntimeError("Do not support target: %s" % target)

    # use tuning to find params
    if trials:
        value, params = find_optimized_parameters(
            match_result, schedule_gen, schedule_app,
            measure_opt, checker, trials,  # policy="random",
            builder=builder,
            runner=runner,
            verbose=verbose)
    else:
        entry = schedule_gen.get_best_entry()
        # we store 1/time_cost in file
        params, value = entry.record, 1 / entry.value

    return AutoTensorizeResult(
        schedule_gen,
        schedule_app,
        params,
        value
    )


def auto_tensorize_v3(
        target_dag, target,
        transform_log_file,
        schedule_log_file,
        measure_opt,
        trials=200,
        schedule_trials=20,
        builder=pebble_local_builder_build,
        runner=pebble_local_runner_run,
        verbose=False,
        verbose_schedule=False,
        transform_dump=False,
        search_batch=5,
        desired_compute_key=None,
        desired_shape_key=None):

    measure_opt.target = target
    match_results = get_match_results(target_dag, target)

    if len(match_results) == 0:
        print("This workload has no matched intrinsic for target: %s" %
              target, flush=True)
        return None, None
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
        return None, None
    shape_key_match_results = []
    if desired_shape_key is not None:
        for m in compute_key_match_results:
            if m.shape_key == desired_shape_key:
                shape_key_match_results.append(m)
    else:
        shape_key_match_results = compute_key_match_results
    if len(shape_key_match_results) == 0:
        print("No match result matches desired shape key:", desired_shape_key, flush=True)
        return None, None

    match_result = shape_key_match_results[0]
    if verbose:
        print("Selected:", str(match_result), flush=True)
        print("Axis map:", flush=True)
        for k, v in match_result.axis_map.items():
            print(k, ":", v, flush=True)
    
    gen = TransformGenerator(
        match_result, log_file=transform_log_file, allow_repeat=True)
    if os.path.exists(transform_log_file) and os.path.isfile(transform_log_file):
        gen.load_from_file(transform_log_file)
    app = TransformApplier(match_result, verbose=transform_dump)

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
    for it in range(iterations):
        if not pure_test:
            record = gen.get_next(policy="random")
        else:
            try:
                entry = gen.get_best_entry()
                record = entry.record
            except Exception as e:
                raise RuntimeError("Can't get previous results for test mode.")
        print(f"Choose transform: {record}", flush=True)
        new_state = app.apply(record)

        record_key = str(record)
        if record_key in schedule_context_cache:
            sch_ctx = schedule_context_cache[record_key]
        else:
            current_log_file = record_key + "_" + schedule_log_file
            if str(target) == "cuda":
                schedule_gen = CUDAScheduleGeneratorV2(
                match_result, new_state, log_file=current_log_file)
                if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                    schedule_gen.load_from_file(current_log_file)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = CUDAScheduleApplierV2(match_result, sc_info)
                checker = CUDAProgramChecker(
                    arch=get_cuda_compute_version(measure_opt.dev_id))
            elif str(target) == "opencl":
                schedule_gen = MaliScheduleGenerator(
                    match_result, new_state, log_file=current_log_file)
                if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                    schedule_gen.load_from_file(current_log_file)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = MaliScheduleApplier(match_result, sc_info)
                # TODO: write a checker for MALI GPU
                checker = MaliProgramChecker(arch="g76")
            else:
                raise RuntimeError("Do not support target: %s" % target)

            # use tuning to find params
            if schedule_trials:
                generate_schedule = find_optimized_parameters_v2(
                    match_result, schedule_gen, schedule_app,
                    measure_opt, checker, schedule_trials,  # policy="random",
                    builder=builder,
                    runner=runner,
                    verbose=verbose_schedule,
                    batch_size=search_batch)
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

        print(f"Iteration: {it+1}: {value}/{best_value}, {str(record)}, {str(params)}", flush=True)

        if (it + 1) % 10 == 0:
            print("Show transformation explore summary:", flush=True)
            for k, v in schedule_context_cache.items():
                print(f"{str(k)}: {v.schedule_gen.num_entries()}", flush=True)

    return AutoTensorizeResult(
        best_ctx.schedule_gen,
        best_ctx.schedule_app,
        best_params,
        1 / best_value
    )
