import os
import tvm
from .search import pebble_local_builder_build, pebble_local_runner_run
from .tensorization_phases import get_match_results, TransformGenerator, TransformApplier
from .tensorization_phases import CUDAScheduleGenerator, CUDAScheduleApplier
from .search import CUDAProgramChecker, find_optimized_parameters




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


def auto_tensorize(target_dag, target,
        log_file,
        measure_opt,
        trials=200,
        builder=pebble_local_builder_build,
        runner=pebble_local_runner_run,
        verbose=False):
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

    gen = TransformGenerator(match_result)
    record = gen.get(policy="random")
    # here is transform policy
    record.unfold_choice = (
        [1 for _ in record.unfold_choice[0]], record.unfold_choice[1])
    app = TransformApplier(match_result)
    new_state = app.apply(record)

    if str(target) == "cuda":
        schedule_gen = CUDAScheduleGenerator(
            match_result, new_state, log_file=log_file)
        if os.path.exists(log_file) and os.path.isfile(log_file):
            schedule_gen.load_from_file(log_file)
        sc_info = schedule_gen.get_schedule_compute_info()
        schedule_app = CUDAScheduleApplier(match_result, sc_info)
        checker = CUDAProgramChecker()
    else:
        raise RuntimeError("Do not support target: %s" % target)
    
    # use tuning to find params
    value, params = find_optimized_parameters(
        match_result, schedule_gen, schedule_app,
        measure_opt, checker, trials,  # policy="random",
        builder=builder,
        runner=runner,
        verbose=verbose)

    return AutoTensorizeResult(
        schedule_gen,
        schedule_app,
        params,
        value
    )


def get_schedule(sch_app, params):
    target_dag = sch_app.target_dag
    inputs = target_dag.get_inputs()
    sch = tvm.te.create_schedule([x.op for x in target_dag.tensors])

    args = inputs + list(target_dag.tensors)
    sch = sch_app.apply(sch, params)
    return sch, args
