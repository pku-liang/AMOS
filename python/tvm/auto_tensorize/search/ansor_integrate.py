import tvm
from functools import reduce
from tvm import auto_scheduler
from tvm.auto_scheduler.cost_model import RandomModel, XGBModel
from tvm.auto_scheduler.search_policy import SketchPolicy


def establish_task_ansor(
    schedule_gen, schedule_app,
        measure_opt, task_name):
    target_dag = schedule_app.target_dag
    inputs = target_dag.get_inputs()
    args = inputs + list(target_dag.tensors)
    def task_func():
        return args

    registered_func = auto_scheduler.register_workload(
        task_name, f=task_func)

    target = tvm.target.Target(measure_opt.target)

    task = auto_scheduler.create_task(
        task_name, (), target, recipe=schedule_gen.recipe_stage)

    return task


def find_optimized_params_ansor(task, measure_opt, trials, model="random"):
    task_name = task.workload_key
    log_name = task_name + ".log"

    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        priority=measure_opt.priority,
        timeout=measure_opt.timeout,
        number=measure_opt.number,
        repeat=measure_opt.repeat,
        min_repeat_ms=measure_opt.min_repeat_ms,
        cooldown_interval=measure_opt.cooldown_interval,
        enable_cpu_cache_flush=measure_opt.enable_cpu_cache_flush)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=trials,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_name)],
    )

    if model == "random":
        cost_model = RandomModel()
    elif model == "xgb":
        cost_model = XGBModel()
    else:
        raise RuntimeError("Unsupported model: %s" % model)
    search_policy = SketchPolicy(task, cost_model)
    sch, args = auto_scheduler.auto_schedule(
        task, search_policy=search_policy, tuning_options=tune_option)

    return log_name


def get_schedule_ansor(task, log_name):
    inp, res = auto_scheduler.load_best(log_name, task.workload_key)
    sch, args = task.compute_dag.apply_steps_from_state(inp.state)
    return sch, args