import tvm
import sys
import os
import tvm._ffi
import numpy as np
import multiprocessing
import multiprocessing.pool
import psutil
import signal
import queue
import copy
import json
import time
import math

from functools import reduce
from .schedule_state import RealScheduleState
from .hardware_config import get_hardware_config
from .schedule_merge import schedule_cuda_merge
from .schedule_allredcue import schedule_cuda_allreduce
from .schedule_buffer_output import schedule_cuda_buffer_output
from .schedule_tiling_and_binding import schedule_cuda_tiling_and_binding
from .schedule_buffer_input import schedule_cuda_buffer_input, create_buffer
from .schedule_unroll import schedule_cuda_unroll
from .utils import tile_axis, tile_axes, reorder_spatial_and_reduce_axes
from ..utils import to_tuple, to_int, can_to_int, to_int_or_None, ASSERT, ERROR

from tvm import auto_tensorize as at, tg
from tvm import auto_scheduler
from tvm.auto_scheduler.cost_model import RandomModel, XGBModel
from tvm.auto_scheduler.search_policy import SketchPolicy

from collections import OrderedDict


def interpret_cuda_schedule(sch, tensors, subgraph, multi_entity, hd_config, debug=sys.stdout):
    op_to_id = {}
    op_to_state = {}
    for i, op in enumerate(subgraph.operation_list):
        op_to_id[op] = i
        op_to_state[op] = RealScheduleState("cuda")
    # reverse order, from output to input
    for op in reversed(subgraph.operation_list):
        schedule_cuda_merge(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_buffer_output(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        create_buffer(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_allreduce(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_tiling_and_binding(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_buffer_input(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )
        schedule_cuda_unroll(
            op, op_to_id, op_to_state, sch, tensors, subgraph, multi_entity, hd_config, debug=debug
        )


def interpret_llvm_schedule(sch, tensors, subgraph, multi_entity, hd_config):
    return


@tvm._ffi.register_func("tg.autoschedule.interpret")
def interpret(sch, tensors, subgraph, target, multi_entity):
    with open("trace_debug_autoschedule.log", "a") as debug:
        if str(target.kind) == "cuda":
            hd_config = get_hardware_config("default_cuda", "cuda")
            interpret_cuda_schedule(sch, tensors, subgraph, multi_entity, hd_config, debug=debug)
        elif str(target.kind) == "llvm":
            hd_config = get_hardware_config("default_llvm", "llvm")
            interpret_llvm_schedule(sch, tensors, subgraph, multi_entity, hd_config)
        else:
            ERROR("Currently no support for target", target.kind, type(target.kind))
        return


def set_interpret(func):
    tvm._ffi.register_func("tg.autoschedule.interpret", func, True)


@tvm._ffi.register_func("tg.autoschedule.auto_tensorize_cuda")
def auto_tensorize_cuda(sch, tensors, log_file, trials):
    target = "cuda"
    measure_opt = at.MeasureOptions(target=target, timeout=10, number=200, min_repeat_ms=500)
    target_dag = at.compute_dag_from_tensors(tensors)
    result = at.auto_tensorize(
        target_dag, target, log_file, measure_opt, trials=trials, transform_dump=True
    )
    if result.defined():
        sch, args = at.get_schedule(result.sch_app, result.params)
        return {sch: args}
    else:
        return {sch: []}


@tvm._ffi.register_func("tg.autoschedule.build_func")
def build_func(sch, args, target, target_host, name, binds):
    binds = {x: y for x, y in binds.items()}
    return tvm.build(sch, args, target, target_host, name, binds)


class TGAutoScheduleContext(object):
    def __init__(self, name, top_log_dir, subgraph, measure_option, verbose=False):
        self.measure_option = measure_option
        self.target = tvm.target.Target(measure_option.target)
        self.name = name
        self.subgraph = subgraph
        self.log_name = os.path.join(top_log_dir, "tg:" + name + ".log")
        self.logger = open(self.log_name, "a")
        self.best_perf = 1e-10
        self.best_result = None
        self.verbose = verbose
        self.result = None
        self.counter = 0
        self.total_trials = 0

        if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
            with open(self.log_name, "r") as fin:
                print("Loading from %s..." % self.log_name, flush=True)
                for line in fin:
                    obj = json.loads(line)
                    entity = obj["entity"]
                    perf = obj["perf"]
                    entity = tg.string_to_multi_schedule_entity(entity)
                    result = tg.get_schedule_result_from_entity(name, subgraph, self.target, entity)
                    if perf > self.best_perf:
                        self.best_perf = perf
                        self.best_result = result
                    # feedback
                    tg.get_schedule_result(
                        self.name,
                        self.subgraph,
                        self.target,
                        self.measure_option.dev_id,
                        self.measure_option.timeout,
                        perf,
                        True,
                        result,
                    )

    def __del__(self):
        self.logger.close()

    def get_new_schedule(self):
        ret = None
        while ret is None:
            try:
                ret = tg.get_schedule_result(
                    self.name,
                    self.subgraph,
                    self.target,
                    self.measure_option.dev_id,
                    self.measure_option.timeout,
                )
            except Exception as e:
                if self.verbose:
                    print(e)
                pass
        return ret

    def count(self):
        self.counter = (self.counter + 1) % 16
        if self.counter == 0:
            print("\n", flush=True)
            print("Currently best performance:", 1 / self.best_perf, flush=True)

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s by %d trials..." % (self.log_name, trials), flush=True)
        self.total_trials += trials
        search_group_size = 10
        iterations = (trials + search_group_size - 1) // search_group_size
        beg = time.time()
        for i in range(iterations):
            schs = []
            args_lst = []
            results = []
            for j in range(search_group_size):
                # this one is get new schedule
                result = self.get_new_schedule()
                # measure current result
                results.append(result)
                sch = result.schedule
                args = result.tensors
                schs.append(sch)
                args_lst.append(args)

            timecosts = at.evaluate_schedules(schs, args_lst, self.measure_option)

            for result, timecost in zip(results, timecosts):
                # timecost = 1.0
                perf = 1.0 / (timecost + 1e-10)
                if perf > self.best_perf:
                    self.best_perf = perf
                    self.best_result = result
                    entity = tg.multi_schedule_entity_to_string(result.schedule_entities)
                    log = {"entity": entity, "perf": perf}
                    print(json.dumps(log), file=self.logger, flush=True)
                    print(".B", end="", flush=True)
                elif timecost != at.MAX_FLOAT:
                    print(".N", end="", flush=True)
                self.count()
                # this one is feedback
                tg.get_schedule_result(
                    self.name,
                    self.subgraph,
                    self.target,
                    self.measure_option.dev_id,
                    self.measure_option.timeout,
                    perf,
                    True,
                    result,
                )
        end = time.time()
        print("Schedule cost %f seconds" % (end - beg))

        sch, args, perf = self.get_best_schedule()
        return sch, args

    def get_best_schedule(self):
        if self.best_result is not None:
            return self.best_result.schedule, self.best_result.tensors, 1 / self.best_perf
        else:
            return None, None, at.MAX_FLOAT

    def get_measure_opt(self):
        return self.measure_option


class AnsorAutoScheduleContext(object):
    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.measure_option = measure_option
        self.target_dag = at.compute_dag_from_tensors([x.output(0) for x in subgraph.root_ops])
        task_name = name

        inputs = self.target_dag.get_inputs()
        args = inputs + list(self.target_dag.tensors)

        self.total_trials = 0

        def task_func():
            return args

        registered_func = auto_scheduler.register_workload(task_name, f=task_func)

        target = tvm.target.Target(measure_option.target)

        self.task = auto_scheduler.create_task(
            task_name,
            (),
            target,
            hardware_params=auto_scheduler.HardwareParams(
                1024,  # cores
                16,  # vector bytes
                1024,  # cache line bytes
            ),
        )

        task_name = self.task.workload_key[2:-2]
        self.log_name = os.path.join(top_log_dir, "ansor:" + task_name + ".log")

        # self.measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        #   priority=measure_option.priority,
        #   timeout=measure_option.timeout,
        #   number=measure_option.number,
        #   repeat=measure_option.repeat,
        #   min_repeat_ms=measure_option.min_repeat_ms,
        #   cooldown_interval=measure_option.cooldown_interval,
        #   enable_cpu_cache_flush=measure_option.enable_cpu_cache_flush)

        self.runner = auto_scheduler.LocalRunner(
            timeout=measure_option.timeout,
            number=measure_option.number,
            repeat=measure_option.repeat,
            min_repeat_ms=measure_option.min_repeat_ms,
            cooldown_interval=measure_option.cooldown_interval,
            enable_cpu_cache_flush=measure_option.enable_cpu_cache_flush,
        )

    def auto_schedule(self, trials, model="xgb"):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s by %d trials..." % (self.log_name, trials), flush=True)
        self.total_trials += trials
        sch, args = None, None
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=trials,
            # runner=self.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(self.log_name)],
        )

        if model == "random":
            cost_model = RandomModel()
        elif model == "xgb":
            cost_model = XGBModel()
        else:
            raise RuntimeError("Unsupported model: %s" % model)
        if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
            cost_model.update_from_file(self.log_name)
            search_policy = auto_scheduler.SketchPolicy(
                self.task,
                cost_model,
                init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(self.log_name)],
            )
        else:
            search_policy = SketchPolicy(self.task, cost_model)
        sch, args = auto_scheduler.auto_schedule(
            self.task, search_policy=search_policy, tuning_options=tune_option
        )

        return sch, args

    def get_best_schedule(self):
        try:
            inp, res = auto_scheduler.load_best(self.log_name, self.task.workload_key)
            sch, args = self.task.compute_dag.apply_steps_from_state(inp.state)
            return sch, args, np.mean([float(x) * 1e3 for x in res.costs])
        except:
            return None, None, at.MAX_FLOAT

    def get_measure_opt(self):
        return self.measure_option


class AutoTensorizeContext(object):
    @classmethod
    def can_use(cls, name, top_log_dir, subgraph, measure_option):
        target_dag = at.compute_dag_from_tensors([x.output(0) for x in subgraph.root_ops])
        measure_option = measure_option
        task_name = name
        log_name = os.path.join(top_log_dir, "at:" + task_name + ".log")
        match_result, new_state = at.auto_tensorize_compute(
            target_dag, measure_option.target, log_name, measure_option
        )
        return match_result is not None and new_state is not None

    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.target_dag = at.compute_dag_from_tensors([x.output(0) for x in subgraph.root_ops])
        self.measure_option = measure_option
        task_name = name
        self.log_name = os.path.join(top_log_dir, "at:" + task_name + ".log")
        self.match_result, self.new_state = at.auto_tensorize_compute(
            self.target_dag, measure_option.target, self.log_name, measure_option
        )

        self.total_trials = 0

        assert self.match_result is not None
        assert self.new_state is not None

        if str(self.measure_option.target) == "cuda":
            self.schedule_gen = at.CUDAScheduleGeneratorV2(
                self.match_result, self.new_state, log_file=self.log_name
            )
            if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
                self.schedule_gen.load_from_file(self.log_name)
            sc_info = self.schedule_gen.get_schedule_compute_info()
            self.schedule_app = at.CUDAScheduleApplierV2(self.match_result, sc_info)
            self.checker = at.CUDAProgramChecker(
                arch=at.get_cuda_compute_version(self.measure_option.dev_id)
            )
        else:
            raise RuntimeError("Do not support target: %s" % self.measure_option.target)

        self.builder = at.pebble_local_builder_build
        self.runner = at.pebble_local_runner_run

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s..." % self.log_name, flush=True)
        self.total_trials += trials
        if trials:
            value, params = at.find_optimized_parameters(
                self.match_result,
                self.schedule_gen,
                self.schedule_app,
                self.measure_option,
                self.checker,
                trials,  # policy="random",
                builder=self.builder,
                runner=self.runner,
                verbose=False,
            )

        sch, args, perf = self.get_best_schedule()
        return sch, args

    def get_best_schedule(self):
        if self.schedule_gen.has_entry():
            entry = self.schedule_gen.get_best_entry()
            # we store 1/time_cost in file
            params = entry.record
            sch, args = at.get_schedule(self.schedule_app, params)
            return sch, args, 1 / entry.value * 1e3
        else:
            return None, None, at.MAX_FLOAT

    def get_measure_opt(self):
        return self.measure_option


class AutoTensorizeContextV2(object):
    @classmethod
    def can_use(cls, name, top_log_dir, subgraph, measure_option):
        target_dag = at.compute_dag_from_tensors([x.output(0) for x in subgraph.root_ops])
        measure_option = measure_option
        match_results = at.get_match_results(target_dag, measure_option.target)
        return len(match_results) > 0

    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.target_dag = at.compute_dag_from_tensors([x.output(0) for x in subgraph.root_ops])
        self.measure_option = measure_option
        task_name = name
        self.log_dir = os.path.join(top_log_dir, "at-" + task_name)
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        self.log_name = os.path.join(self.log_dir, "at:" + task_name + ":transform" + ".log")
        match_results = at.get_match_results(self.target_dag, measure_option.target)

        self.total_trials = 0

        assert len(match_results) > 0
        self.match_result = match_results[0]

        self.gen = at.MappingGenerator(self.match_result, log_file=self.log_name, allow_repeat=True)
        if os.path.exists(self.log_name) and os.path.isfile(self.log_name):
            self.gen.load_from_file(self.log_name)
        self.app = at.MappingApplier(self.match_result, verbose=True, strict=False)

        class ScheduleContext:
            def __init__(self, schedule_gen, schedule_app, sc_info, checker, generate_schedule):
                self.schedule_gen = schedule_gen
                self.schedule_app = schedule_app
                self.sc_info = sc_info
                self.checker = checker
                self.generate_schedule = generate_schedule

        self.schedule_ctx_cls = ScheduleContext

        self.schedule_context_cache = {}
        self.best_value = 1 / at.MAX_FLOAT
        self.best_ctx = None
        self.best_params = None
        self.pure_test = False
        self.drop_output = False
        self.enable_split_K = False
        self.use_shared_store = False

        self.builder = at.pebble_local_builder_build
        self.runner = at.pebble_local_runner_run

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s..." % self.log_name, flush=True)
        self.total_trials += trials
        schedule_trials = 20
        iterations = trials // schedule_trials

        if iterations == 0:
            iterations = 1
            schedule_trials = 0
            self.pure_test = True
            print("Pure testing mode...", flush=True)
        beg = time.time()
        for it in range(iterations):
            if not self.pure_test:
                feasible = False
                while not feasible:
                    record = self.gen.get_next(policy="random")
                    try:
                        tmp_app = at.MappingApplier(self.match_result, strict=False)
                        tmp_app.apply(record, drop_output=self.drop_output)
                        feasible = True
                    except RuntimeError as e:
                        print("Catch an infeasible mapping:", flush=True)
                        print(record, flush=True)
            else:
                try:
                    entry = self.gen.get_best_entry()
                    record = entry.record
                except Exception as e:
                    raise RuntimeError("Can't get previous results for test mode.")
            print(f"Choose transform: {record}", flush=True)
            new_state = self.app.apply(record, drop_output=self.drop_output)

            record_key = record.as_key()
            if record_key in self.schedule_context_cache:
                sch_ctx = self.schedule_context_cache[record_key]
            else:
                current_log_file = os.path.join(
                    self.log_dir, "at:" + self.name + ":mapping:" + str(record_key) + ".log"
                )
                if str(self.measure_option.target) == "cuda":
                    if not self.enable_split_K:
                        if self.use_shared_store:
                            raise NotImplementedError()
                            # schedule_gen = at.CUDAScheduleGeneratorV3(
                            #     self.match_result, new_state, log_file=current_log_file)
                            # if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            #     schedule_gen.load_from_file(current_log_file)
                            # sc_info = schedule_gen.get_schedule_compute_info()
                            # schedule_app = at.CUDAScheduleApplierV3(self.match_result, sc_info)
                        else:
                            schedule_gen = at.CUDAScheduleGeneratorV2(
                                self.match_result, new_state, log_file=current_log_file
                            )
                            if os.path.exists(current_log_file) and os.path.isfile(
                                current_log_file
                            ):
                                schedule_gen.load_from_file(current_log_file)
                            sc_info = schedule_gen.get_schedule_compute_info()
                            schedule_app = at.CUDAScheduleApplierV2(self.match_result, sc_info)
                    else:
                        schedule_gen = at.CUDAScheduleGeneratorSplitK(
                            self.match_result, new_state, log_file=current_log_file
                        )
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = at.CUDAScheduleApplierSplitK(self.match_result, sc_info)
                    checker = at.CUDAProgramChecker(
                        arch=at.get_cuda_compute_version(self.measure_option.dev_id)
                    )
                elif str(self.measure_option.target) == "opencl":
                    schedule_gen = at.MaliScheduleGenerator(
                        self.match_result, new_state, log_file=current_log_file
                    )
                    if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                        schedule_gen.load_from_file(current_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = at.MaliScheduleApplier(self.match_result, sc_info)
                    # TODO: write a checker for MALI GPU
                    checker = at.MaliProgramChecker(arch="g76")
                else:
                    raise RuntimeError("Do not support target: %s" % self.measure_option.target)

                # use tuning to find params
                if schedule_trials:
                    generate_schedule = at.find_optimized_parameters_v2(
                        self.match_result,
                        schedule_gen,
                        schedule_app,
                        self.measure_option,
                        checker,
                        schedule_trials,  # policy="random",
                        builder=self.builder,
                        runner=self.runner,
                        verbose=False,
                        search_group_size=10,
                    )
                else:
                    generate_schedule = None

                sch_ctx = self.schedule_ctx_cls(
                    schedule_gen, schedule_app, sc_info, checker, generate_schedule
                )
                self.schedule_context_cache[record_key] = sch_ctx

            if sch_ctx.generate_schedule is not None:
                value, params = next(sch_ctx.generate_schedule)
            try:
                entry = sch_ctx.schedule_gen.get_best_entry()
                # we store 1/time_cost in file
                params, value = entry.record, entry.value
                # print("Evaluation only:", params, value, flush=True)
                if not self.pure_test:
                    self.gen.feedback(record, value)
            except Exception as e:
                params = None
                value = 1 / at.MAX_FLOAT

            # record the best
            if value > self.best_value:
                self.best_value = value
                self.best_ctx = sch_ctx
                self.best_params = params

            print(
                f"Iteration: {it+1}: {value}/{self.best_value}, {str(record)}, {str(params)}",
                flush=True,
            )

            if (it + 1) % 10 == 0:
                print("Show transformation explore summary:", flush=True)
                for k, v in self.schedule_context_cache.items():
                    print(f"{str(k)}: {v.schedule_gen.num_entries()}", flush=True)

        end = time.time()
        print(f"Tensorize use time {(end - beg)} s.", flush=True)
        sch, args, perf = self.get_best_schedule()
        return sch, args

    def get_best_schedule(self):
        tmp_schedule_context_cache = {}
        if self.gen.has_entry():
            best_transform = self.gen.get_best_entry()
            transform = best_transform.record
            transform_key = transform.as_key()
            best_log_file = os.path.join(
                self.log_dir, "at:" + self.name + ":mapping:" + str(transform_key) + ".log"
            )
            if transform_key in self.schedule_context_cache:
                schedule_gen = self.schedule_context_cache[transform_key].schedule_gen
                schedule_app = self.schedule_context_cache[transform_key].schedule_app
            elif transform_key in tmp_schedule_context_cache:
                schedule_gen, schedule_app = tmp_schedule_context_cache[transform_key]
            else:
                new_state = self.app.apply(transform, drop_output=self.drop_output)
                if str(self.measure_option.target) == "cuda":
                    if not self.enable_split_K:
                        if self.use_shared_store:
                            raise NotImplementedError()
                        else:
                            schedule_gen = at.CUDAScheduleGeneratorV2(
                                self.match_result, new_state, log_file=best_log_file
                            )
                            if os.path.exists(best_log_file) and os.path.isfile(best_log_file):
                                schedule_gen.load_from_file(best_log_file)
                            sc_info = schedule_gen.get_schedule_compute_info()
                            schedule_app = at.CUDAScheduleApplierV2(self.match_result, sc_info)
                    else:
                        schedule_gen = at.CUDAScheduleGeneratorSplitK(
                            self.match_result, new_state, log_file=best_log_file
                        )
                        if os.path.exists(best_log_file) and os.path.isfile(best_log_file):
                            schedule_gen.load_from_file(best_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = at.CUDAScheduleApplierSplitK(self.match_result, sc_info)
                    checker = at.CUDAProgramChecker(
                        arch=at.get_cuda_compute_version(self.measure_option.dev_id)
                    )
                elif str(self.measure_option.target) == "opencl":
                    schedule_gen = at.MaliScheduleGenerator(
                        self.match_result, new_state, log_file=best_log_file
                    )
                    if os.path.exists(best_log_file) and os.path.isfile(best_log_file):
                        schedule_gen.load_from_file(best_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = at.MaliScheduleApplier(self.match_result, sc_info)
                    # TODO: write a checker for MALI GPU
                    checker = at.MaliProgramChecker(arch="g76")
                else:
                    raise RuntimeError("Do not support target: %s" % self.measure_option.target)
                tmp_schedule_context_cache[transform_key] = (schedule_gen, schedule_app)

            if schedule_gen.has_entry():
                entry = schedule_gen.get_best_entry()
                # we store 1/time_cost in file
                params = entry.record
                sch, args = at.get_schedule(schedule_app, params)
                return sch, args, 1 / entry.value * 1e3
            else:
                return None, None, at.MAX_FLOAT
        else:
            return None, None, at.MAX_FLOAT

    def get_measure_opt(self):
        return self.measure_option


class AutoTensorizeContextV3(object):
    @classmethod
    def can_use(cls, name, top_log_dir, subgraph, measure_option):
        target_dag = at.compute_dag_from_tensors([x.output(0) for x in subgraph.root_ops])
        measure_option = measure_option
        match_results = at.get_match_results(target_dag, measure_option.target)
        return len(match_results) > 0

    def __init__(self, name, top_log_dir, subgraph, measure_option):
        self.name = name
        self.subgraph = subgraph
        self.target_dag = at.compute_dag_from_tensors([x.output(0) for x in subgraph.root_ops])
        self.measure_option = measure_option
        task_name = name
        self.log_dir = os.path.join(top_log_dir, "at-" + task_name)
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        self.log_name = os.path.join(self.log_dir, "at:" + task_name + ":transform" + ".log")
        match_results = at.get_match_results(self.target_dag, measure_option.target)

        self.total_trials = 0

        assert len(match_results) > 0

        self.drop_output = False
        self.search_group_size = 5
        self.repeat_rounds = 2
        self.enable_split_K = False
        self.use_shared_store = False
        self.enable_perf_model = False

        self.builder = at.pebble_local_builder_build
        self.runner = at.pebble_local_runner_run

        self.all_matches = []
        self.all_mappings = []
        self.appliers = []
        self.mapping_weights = []
        self.weights_updates = []
        self.momentum = 0.8
        # use all_fit logic to choose the one with minimum padding
        match_result, _ = at.policy.all_fit(match_results)
        match_results = [match_result]
        self.total_matchings = 0
        self.total_mappings = 0
        for match_result in match_results:
            transform_strict = True
            self.all_matches.append(match_result)
            gen = at.MappingGenerator(match_result)
            mappings = gen.get_all()
            # filter out infeasible mappings
            feasible_mappings = []
            tmp_app = at.MappingApplier(match_result, strict=transform_strict)
            for mapping in mappings:
                try:
                    tmp_app.apply(mapping, drop_output=self.drop_output)
                    feasible_mappings.append(mapping)
                except RuntimeError as e:
                    pass
            if len(feasible_mappings) == 0:
                # relax
                transform_strict = False
            else:
                mappings = feasible_mappings
            # record the feasible mappings
            self.all_mappings.append(mappings)
            self.total_matchings += 1
            assert len(mappings) > 0
            self.total_mappings += len(mappings)
            self.mapping_weights.append([1.0 / len(mappings) for m in mappings])
            self.weights_updates.append([0.0 for m in mappings])
            app = at.MappingApplier(match_result, verbose=False, strict=transform_strict)
            self.appliers.append(app)
        assert self.total_mappings > 0

        class ScheduleContext:
            def __init__(self, schedule_gen, schedule_app, sc_info, checker, generate_schedule):
                self.schedule_gen = schedule_gen
                self.schedule_app = schedule_app
                self.sc_info = sc_info
                self.checker = checker
                self.generate_schedule = generate_schedule

        self.schedule_ctx_cls = ScheduleContext

        # global context for overall exploration
        self.schedule_context_cache = {}
        self.best_value = 1 / at.MAX_FLOAT
        self.best_ctx = None
        self.best_params = None

    def auto_schedule(self, trials):
        print("#############################################", flush=True)
        print(self.subgraph.tag, flush=True)
        print("Autoscheduling %s..." % self.log_name, flush=True)
        self.total_trials += trials

        if trials < self.total_mappings * self.repeat_rounds * self.search_group_size:
            print(
                f"[Warning] Too few trials, expect at least {self.total_mappings * self.repeat_rounds * self.search_group_size} trials.",
                flush=True,
            )
            trials = self.total_mappings * self.repeat_rounds * self.search_group_size
            print(
                f"Increase trials to {self.total_mappings * self.repeat_rounds * self.search_group_size}.",
                flush=True,
            )
        else:
            print("Total trials:", trials, flush=True)
        trials_per_matching = trials // self.repeat_rounds // self.total_matchings

        print("Num rounds:", self.repeat_rounds, flush=True)
        print("Num matching:", self.total_matchings, flush=True)
        print("Num mapping:", self.total_mappings, flush=True)
        print("Initial trials per matching:", trials_per_matching, flush=True)

        beg = time.time()
        for round in range(self.repeat_rounds):
            for match_id in range(self.total_matchings):
                match_result = self.all_matches[match_id]
                app = self.appliers[match_id]
                weights = self.mapping_weights[match_id]
                updates = self.weights_updates[match_id]
                tune_trials = [math.ceil(trials_per_matching * x) for x in weights]
                best_values_of_mappings = []
                print("Original weights", weights, flush=True)
                print("Original trials for each mapping", tune_trials, flush=True)
                print("Current explored matching:", str(match_result), flush=True)
                print("Its axis mapping:", flush=True)
                for i, v in match_result.axis_map.items():
                    print(i.var, ":", [x.var for x in v], flush=True)
                for mapping_id in range(len(self.all_mappings[match_id])):
                    record = self.all_mappings[match_id][mapping_id]
                    print("Current explored mapping:", str(record), flush=True)

                    # transform compute
                    new_state = app.apply(record, drop_output=self.drop_output)
                    # prepare tune log file
                    record_key = record.as_key()
                    current_log_file = os.path.join(
                        self.log_dir, "at:" + self.name + ":mapping:" + str(record_key) + ".log"
                    )
                    if record_key in self.schedule_context_cache:
                        sch_ctx = self.schedule_context_cache[record_key]
                    else:
                        schedule_gen, schedule_app, checker, sc_info = self._get_schedule_ctx(
                            match_result, new_state, current_log_file
                        )

                        # tune loop
                        schedule_trials = tune_trials[mapping_id]
                        if schedule_trials:
                            # this returns a generator
                            if self.enable_perf_model:
                                generate_schedule = at.find_optimized_parameters_v3(
                                    match_result,
                                    schedule_gen,
                                    schedule_app,
                                    self.measure_option,
                                    checker,
                                    schedule_trials,  # policy="random",
                                    builder=self.builder,
                                    runner=self.runner,
                                    verbose=False,
                                    search_group_size=self.search_group_size,
                                    build_parallel=1,
                                    run_parallel=1,
                                    perf_percentage=0.5,
                                )
                            else:
                                generate_schedule = at.find_optimized_parameters_v2(
                                    match_result,
                                    schedule_gen,
                                    schedule_app,
                                    self.measure_option,
                                    checker,
                                    schedule_trials,  # policy="random",
                                    builder=self.builder,
                                    runner=self.runner,
                                    verbose=False,
                                    search_group_size=self.search_group_size,
                                    build_parallel=1,
                                    run_parallel=1,
                                )
                        else:
                            generate_schedule = None

                        # create new schedule context
                        sch_ctx = self.schedule_ctx_cls(
                            schedule_gen, schedule_app, sc_info, checker, generate_schedule
                        )
                        self.schedule_context_cache[record_key] = sch_ctx

                    if sch_ctx.generate_schedule is not None:
                        value, params = next(sch_ctx.generate_schedule)
                    try:
                        entry = sch_ctx.schedule_gen.get_best_entry()
                        # we store 1/time_cost in file
                        params, value = entry.record, entry.value
                    except Exception as e:
                        params = None
                        value = 1 / at.MAX_FLOAT

                    # record the best value of current mapping
                    best_values_of_mappings.append(value)

                    # record the best
                    if value > self.best_value:
                        self.best_value = value
                        self.best_ctx = sch_ctx
                        self.best_params = params

                    print(f"Best record value:{self.best_value} (larger is better)", flush=True)
                    print(
                        f"Round {round+1}, Match {match_id+1}, Mapping {mapping_id+1}: {value}/{self.best_value}({1/self.best_value*1e3} ms), {str(record)}, {str(params)}",
                        flush=True,
                    )

                # redistribute weights according to current best value
                max_value = max(best_values_of_mappings)
                exp_scores = [math.exp(x - max_value) for x in best_values_of_mappings]
                sum_exp_scores = sum(exp_scores)
                new_weights = [x / sum_exp_scores for x in exp_scores]
                delta_weights = [new_weights[i] - weights[i] for i in range(len(weights))]
                new_updates = [
                    delta_weights[i] + self.momentum * updates[i] for i in range(len(updates))
                ]
                new_weights = [weights[i] + new_updates[i] for i in range(len(new_updates))]
                exp_scores = [math.exp(x) for x in new_weights]
                sum_exp_scores = sum(exp_scores)
                new_weights = [x / sum_exp_scores for x in exp_scores]
                # update into global context
                self.mapping_weights[match_id] = new_weights
                self.weights_updates[match_id] = new_updates
                print("New weights", new_weights, flush=True)

            print("Show mapping exploration summary:", flush=True)
            for k, v in self.schedule_context_cache.items():
                print(
                    f"mapping {str(k)}: explored {v.schedule_gen.num_entries()} schedules",
                    flush=True,
                )
        end = time.time()
        print(f"Tensorize use time {(end - beg)} s.", flush=True)
        sch, args, perf = self.get_best_schedule()
        return sch, args

    def _get_schedule_ctx(self, match_result, new_state, current_log_file):
        target = self.measure_option.target
        if str(target) == "cuda":
            if not self.enable_split_K:
                if self.use_shared_store:
                    raise NotImplementedError()
                else:
                    if self.enable_perf_model:
                        schedule_gen = at.CUDAScheduleGeneratorV3(
                            match_result,
                            new_state,
                            log_file=current_log_file,
                            arch=at.get_cuda_compute_version(self.measure_option.dev_id),
                            verbose_init=False,
                        )
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = at.CUDAScheduleApplierV3(match_result, sc_info)
                    else:
                        schedule_gen = at.CUDAScheduleGeneratorV2(
                            match_result,
                            new_state,
                            log_file=current_log_file,
                            arch=at.get_cuda_compute_version(self.measure_option.dev_id),
                            verbose_init=False,
                        )
                        if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                            schedule_gen.load_from_file(current_log_file)
                        sc_info = schedule_gen.get_schedule_compute_info()
                        schedule_app = at.CUDAScheduleApplierV2(match_result, sc_info)
            else:
                if self.enable_perf_model:
                    raise NotImplementedError()
                else:
                    schedule_gen = at.CUDAScheduleGeneratorSplitK(
                        match_result,
                        new_state,
                        log_file=current_log_file,
                        arch=at.get_cuda_compute_version(self.measure_option.dev_id),
                        verbose_init=False,
                    )
                    if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                        schedule_gen.load_from_file(current_log_file)
                    sc_info = schedule_gen.get_schedule_compute_info()
                    schedule_app = at.CUDAScheduleApplierSplitK(match_result, sc_info)
            checker = at.CUDAProgramChecker(
                arch=at.get_cuda_compute_version(self.measure_option.dev_id), verbose_init=False
            )
        elif str(target) == "opencl":
            schedule_gen = at.MaliScheduleGenerator(
                match_result, new_state, log_file=current_log_file, verbose_init=False
            )
            if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                schedule_gen.load_from_file(current_log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = at.MaliScheduleApplier(match_result, sc_info)
            # TODO: write a checker for MALI GPU
            checker = at.MaliProgramChecker(arch="g76", verbose_init=False)
        elif str(target) == "llvm -mcpu=skylake-avx512":
            schedule_gen = at.LLVMScheduleGenerator(
                match_result, new_state, log_file=current_log_file, verbose_init=False
            )
            if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                schedule_gen.load_from_file(current_log_file)
            sc_info = schedule_gen.get_schedule_compute_info()
            schedule_app = at.LLVMScheduleApplier(match_result, sc_info)
            # TODO: write a checker for CPU
            checker = at.EmptyChecker()
        elif str(target).startswith("tenet"):
            target = str(target)
            parts = target.split(" ")
            assert len(parts) > 1
            if parts[1] == "cuda":
                schedule_gen = at.CUDAScheduleGeneratorTenet(
                    match_result,
                    new_state,
                    log_file=current_log_file,
                    arch=at.get_cuda_compute_version(self.measure_option.dev_id),
                    verbose_init=False,
                )
                if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                    schedule_gen.load_from_file(current_log_file)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = at.CUDAScheduleApplierTenet(match_result, sc_info)
                checker = at.CUDAProgramChecker(
                    arch=at.get_cuda_compute_version(self.measure_option.dev_id), verbose_init=False
                )
            else:
                schedule_gen = at.TenetScheduleGenerator(
                    match_result, new_state, log_file=current_log_file, verbose_init=False
                )
                if os.path.exists(current_log_file) and os.path.isfile(current_log_file):
                    schedule_gen.load_from_file(current_log_file)
                sc_info = schedule_gen.get_schedule_compute_info()
                schedule_app = at.TenetScheduleApplier(match_result, sc_info)
                # TODO: write a checker for TENET
                checker = at.EmptyChecker()
        else:
            raise RuntimeError("Do not support target: %s" % target)
        return schedule_gen, schedule_app, checker, sc_info

    def get_best_schedule(self):
        best_sch = None
        best_args = None
        best_cost = at.MAX_FLOAT
        tmp_schedule_context_cache = {}
        for match_id in range(self.total_matchings):
            match_result = self.all_matches[match_id]
            app = self.appliers[match_id]
            for mapping_id in range(len(self.all_mappings[match_id])):
                record = self.all_mappings[match_id][mapping_id]

                # transform compute
                new_state = app.apply(record, drop_output=self.drop_output)
                # prepare tune log file
                record_key = record.as_key()
                current_log_file = os.path.join(
                    self.log_dir, "at:" + self.name + ":mapping:" + str(record_key) + ".log"
                )
                if record_key in self.schedule_context_cache:
                    schedule_gen = self.schedule_context_cache[record_key].schedule_gen
                    schedule_app = self.schedule_context_cache[record_key].schedule_app
                elif record_key in tmp_schedule_context_cache:
                    schedule_gen, schedule_app = tmp_schedule_context_cache[record_key]
                elif (os.path.exists(current_log_file) and os.path.isfile(current_log_file)):
                    schedule_gen, schedule_app, checker, sc_info = self._get_schedule_ctx(
                        match_result, new_state, current_log_file
                    )
                    # create new tmp schedule context
                    tmp_schedule_context_cache[record_key] = (schedule_gen, schedule_app)
                else:
                    continue
                if schedule_gen.has_entry():
                    entry = schedule_gen.get_best_entry()
                    # we store 1/time_cost in file
                    params = entry.record
                    sch, args = at.get_schedule(schedule_app, params)
                    if 1 / entry.value < best_cost:
                        best_sch = sch
                        best_args = args
                        best_cost = 1 / entry.value
        return best_sch, best_args, best_cost

    def get_measure_opt(self):
        return self.measure_option


class AutoScheduleGraphDispatch(object):
    working_set = {}
    results = {}

    @classmethod
    def add_task(
        cls, name, top_log_dir, subgraph, measure_option, scheduler_option="auto_tensorize_v3"
    ):
        use_at = 0
        next_id = len(AutoScheduleGraphDispatch.working_set)
        if scheduler_option == "auto_tensorize_v3" or scheduler_option == "auto_tensorize":
            if AutoTensorizeContextV3.can_use(name, top_log_dir, subgraph, measure_option):
                ctx = AutoTensorizeContextV3(name, top_log_dir, subgraph, measure_option)
                use_at = 1
            else:
                # fallback to TG
                print("Fallback to TG")
                ctx = TGAutoScheduleContext(name, top_log_dir, subgraph, measure_option)
        elif scheduler_option == "auto_tensorize_v2":
            if AutoTensorizeContextV2.can_use(name, top_log_dir, subgraph, measure_option):
                ctx = AutoTensorizeContextV2(name, top_log_dir, subgraph, measure_option)
                use_at = 1
            else:
                # fallback to TG
                print("Fallback to TG")
                ctx = TGAutoScheduleContext(name, top_log_dir, subgraph, measure_option)
        elif scheduler_option == "tg":
            ctx = TGAutoScheduleContext(name, top_log_dir, subgraph, measure_option)
        elif scheduler_option == "ansor":
            ctx = AnsorAutoScheduleContext(name, top_log_dir, subgraph, measure_option)
        else:
            raise RuntimeError("Unknown scheduler: %s" % scheduler_option)
        AutoScheduleGraphDispatch.working_set[next_id] = ctx
        sch, args, perf = ctx.get_best_schedule()
        # if sch is not None:
        #   perf = at.evaluate_schedule(
        #     sch, args, ctx.get_measure_opt(), new_process=True)
        # else:
        #   perf = at.MAX_FLOAT
        AutoScheduleGraphDispatch.results[next_id] = (sch, args, perf)
        return next_id, ctx, use_at

    @classmethod
    def remove_task(cls, task_id):
        if task_id in AutoScheduleGraphDispatch.working_set:
            del AutoScheduleGraphDispatch.working_set[task_id]

    @classmethod
    def auto_schedule(cls, selected_ids, trials_lst):
        for tid, trials in zip(selected_ids, trials_lst):
            if not trials:
                continue
            if tid in AutoScheduleGraphDispatch.working_set:
                ctx = AutoScheduleGraphDispatch.working_set[tid]
                # if isinstance(ctx, TGAutoScheduleContext):
                ctx.auto_schedule(trials)
                sch, args, perf = ctx.get_best_schedule()
                # if sch is not None:
                #   perf = at.evaluate_schedule(
                #     sch, args, ctx.get_measure_opt(), new_process=True)
                # else:
                #   perf = at.MAX_FLOAT
                AutoScheduleGraphDispatch.results[tid] = (sch, args, perf)

    @classmethod
    def query_schedule(cls, tid):
        if tid in AutoScheduleGraphDispatch.results:
            ctx = AutoScheduleGraphDispatch.working_set[tid]
            sch, args, perf = ctx.get_best_schedule()
            # if sch is not None:
            #   perf = at.evaluate_schedule(
            #     sch, args, ctx.get_measure_opt(), new_process=True)
            # else:
            #   perf = at.MAX_FLOAT
            print(
                "Query subgraph task id: %s, perf=%f ms after %d tuning"
                % (str(tid), perf, ctx.total_trials),
                flush=True,
            )
            return (sch, args, perf)
        else:
            return (None, None, at.MAX_FLOAT)


class AutoScheduleMultiGraphContext(object):
    def __init__(
        self,
        name,
        tir_multi_graph,
        measure_option,
        scheduler_option="auto_tensorize_v3",
        gamma=0.02,
        trials=100,
        policy="equal",
    ):
        self.tir_multi_graph = tir_multi_graph
        self.performance_trace = {}
        self.schedules = {}
        self.contexts = {}
        self.graph_tag_to_tid = {}
        self.C = {}
        self.alpha = {}
        self.beta = {}
        self.X = {}
        self.gamma = gamma
        self.use_at_set = set()
        self.subgraph_count = {}
        self.log_dir = name
        if not (os.path.exists(self.log_dir) and os.path.isdir(self.log_dir)):
            os.mkdir(self.log_dir)
        graphs = tg.get_graphs_from_tir_multi_graph(tir_multi_graph)
        graphs = OrderedDict(sorted([(x.value, y) for x, y in graphs.items()], key=lambda x: x[0]))
        for key, subgraph in graphs.items():
            new_name = name + ":subgraph" + str(key)
            if subgraph.tag in self.graph_tag_to_tid:
                self.subgraph_count[subgraph.tag] += 1
                continue
            else:
                self.subgraph_count[subgraph.tag] = 1
            tid, ctx, use_at = AutoScheduleGraphDispatch.add_task(
                new_name, self.log_dir, subgraph, measure_option, scheduler_option=scheduler_option
            )
            if use_at:
                self.use_at_set.add(subgraph.tag)
            sch, args, perf = AutoScheduleGraphDispatch.query_schedule(tid)
            self.performance_trace[tid] = [perf]
            self.C[tid] = perf
            self.alpha[tid] = perf / (32)
            self.beta[tid] = 1.0
            self.X[tid] = self.calculate_X(tid)
            self.schedules[tid] = (sch, args)
            self.contexts[tid] = ctx
            self.graph_tag_to_tid[subgraph.tag] = tid
        self.L = len(self.graph_tag_to_tid) * trials
        self.trials = trials
        self.policy = policy

    def calculate_X(self, tid):
        raw = math.sqrt(self.C[tid] / (self.alpha[tid] + 1e-10))
        return raw

    def select_next_tasks(self):
        # this is the decision part, currently use the simple decision
        ret = []
        trials = []
        sum_X = reduce(lambda x, y: x + y, self.X.values(), 0.0)

        for tid, lst in self.performance_trace.items():
            if self.policy == "equal":
                ret.append(tid)
                trials.append(self.trials)
            elif self.policy == "rebalance":
                ret.append(tid)
                raw = int(max(1, min(self.X[tid] * self.L / sum_X, self.L)))
                trials.append(raw)
                diff = 2 * (self.C[tid] / (self.alpha[tid] * raw) + self.beta[tid] - lst[-1])
                self.alpha[tid] = max(
                    1e-5,
                    self.alpha[tid]
                    + self.gamma * diff * self.C[tid] / (raw * self.alpha[tid] * self.alpha[tid]),
                )
                self.beta[tid] = max(1e-5, self.beta[tid] - self.gamma * diff)
                self.C[tid] = lst[-1]
                self.X[tid] = self.calculate_X(tid)

        return ret, trials

    def auto_schedule(self):
        tids, trials = self.select_next_tasks()
        AutoScheduleGraphDispatch.auto_schedule(tids, trials)
        for k, lst in self.performance_trace.items():
            sch, args, perf = AutoScheduleGraphDispatch.query_schedule(k)
            self.schedules[k] = (sch, args)
            lst[-1] = perf  # only reserve one

    def get_schedules(self):
        total = 0
        mapped = 0
        for k, v in self.subgraph_count.items():
            total += v
            if k in self.use_at_set:
                mapped += v
        print(
            "[NOTICE] totally",
            total,
            "subgraphs, mapped",
            mapped,
            "subgraphs, ratio=",
            mapped / total * 100.0,
            "%",
        )
        ret = {}
        graphs = tg.get_graphs_from_tir_multi_graph(self.tir_multi_graph)
        graphs = OrderedDict({x.value: y for x, y in graphs.items()})
        for key, subgraph in graphs.items():
            tid = self.graph_tag_to_tid[subgraph.tag]
            sch, args = self.schedules[tid]
            ret[key] = tg.ScheduleTensors(sch, args)
        return ret

    def ready(self):
        graphs = tg.get_graphs_from_tir_multi_graph(self.tir_multi_graph)
        graphs = OrderedDict({x.value: y for x, y in graphs.items()})
        for key, subgraph in graphs.items():
            tid = self.graph_tag_to_tid[subgraph.tag]
            sch, args = self.schedules[tid]
            if sch is None or args is None:
                return False
        return True


class AutoScheduleMultiGraphDispatch(object):
    working_set = {}

    @classmethod
    def add_graph_task(
        cls,
        name,
        tir_multi_graph,
        measure_option,
        scheduler_option="auto_tensorize_v3",
        trials=100,
        policy="equal",
    ):
        next_id = len(AutoScheduleMultiGraphDispatch.working_set)
        AutoScheduleMultiGraphDispatch.working_set[next_id] = AutoScheduleMultiGraphContext(
            name,
            tir_multi_graph,
            measure_option,
            scheduler_option=scheduler_option,
            trials=trials,
            policy=policy,
        )
        return next_id

    @classmethod
    def auto_schedule(cls, tid):
        assert tid in cls.working_set
        ctx = cls.working_set[tid]
        ctx.auto_schedule()

    @classmethod
    def get_schedules(cls, tid):
        assert tid in cls.working_set
        ctx = cls.working_set[tid]
        return ctx.get_schedules()

    @classmethod
    def ready(cls, tid):
        assert tid in cls.working_set
        ctx = cls.working_set[tid]
        return ctx.ready()
