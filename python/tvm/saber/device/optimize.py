import time
import multiprocessing as multi
from pebble import concurrent
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired
from .measure import (
    MAX_FLOAT,
    pebble_local_builder_build_shape_oblivious,
    pebble_local_runner_run_shape_oblivious,
    pebble_rpc_runner_run_shape_oblivious
)


def serial_minimize(
        device_impl,
        generator,
        measure_opt,
        trials=100,
        batch_size=1,
        policy=""
):
    best_value = 1 / MAX_FLOAT
    best_params = None
    if generator.has_entry():
        top1 = generator.topk(k=1)[0]
        best_value = top1.value
        best_params = top1.record
    batch_num = (trials + batch_size - 1) // batch_size
    print("Total search tirals:", trials,
          "\nbatch size:", batch_size,
          "\nbatch num:", batch_num, flush=True)
    tic = time.time()
    for b in range(batch_num):
        print("Search round:", b, flush=True)
        generator.refresh()
        params_lst = []
        for i in range(batch_size):
            if b * batch_size + i < trials:
                # params = generator.get(policy=policy)
                params = generator.get_next(policy=policy)
                # print(str(params))
                params_lst.append(params)
        assert params_lst
        for params in params_lst:
            res = device_impl(params)
            value = 1 / res  # use absolute performance
            if value > 1 / MAX_FLOAT:  # valid results
                generator.feedback(params, value)
            if value > best_value:
                best_value = value
                best_params = params
        print("Current minimal cost: ", 1/best_value, flush=True)
        if best_params is not None:
            print("Current best params:\n", best_params.to_json(), flush=True)
    toc = time.time()
    print("Search %d trials costs %f seconds" % (trials, toc - tic), flush=True)
    return best_value, best_params


def parallel_maximize(
    compile_impl,
    evaluate_impl,
    agg_func,
    generator_lst,
    measure_opt,
    checker,
    iterations=200,
    walk_length_per_iter=10,
    report_period=10,
    builder=pebble_local_builder_build_shape_oblivious,
    build_parallel=1,
    runner=pebble_local_runner_run_shape_oblivious,
    verbose=False,
    homogenerous=True,
):
    # sch_impl, args, vars, tensor_lst, var_value_lst, agg_func = device_impl()
    if measure_opt.use_rpc:
        runner = pebble_rpc_runner_run_shape_oblivious
    print("Total search iteraions:", iterations, flush=True)
    tic = time.time()
    for it in range(iterations):
        gen_ctx = []
        for i, gen in enumerate(generator_lst):
            gen.refresh()
            for j in range(walk_length_per_iter):
                params = gen.get_next()
                gen_ctx.append(params)
        build_results = builder(
            compile_impl, gen_ctx, measure_opt, checker, n_parallel=build_parallel)
        run_results = runner(
            build_results, gen_ctx, evaluate_impl, measure_opt)
        for i, gen in enumerate(generator_lst):
            for params, res in zip(
                gen_ctx[i*walk_length_per_iter:(i+1)*walk_length_per_iter],
                run_results[i*walk_length_per_iter:(i+1)*walk_length_per_iter]):
                if verbose:
                    print(res.costs)
                feedback = agg_func([x.value for x in res.costs])
                # genenrator use max heap
                if homogenerous:
                    # share information among generators
                    for g in generator_lst:
                        g.feedback(params, feedback)
                else:
                    gen.feedback(params, feedback)
        if (it + 1) % report_period == 0:
            top1 = None
            top1_value = -MAX_FLOAT
            for gen in generator_lst:
                gtop1 = gen.topk(k=1)[0]
                best_value = gtop1.value
                best_params = gtop1.record
                if best_value > top1_value:
                    top1_value = best_value
                    top1 = best_params
            print("Round", it + 1, flush=True)
            print("Current best result:", top1_value, flush=True)
            print("Best params:", top1, flush=True)

    toc = time.time()
    print("Search %d trials costs %f seconds" % (trials, toc - tic), flush=True)
    top1 = None
    top1_value = -MAX_FLOAT
    for gen in generator_lst:
        gtop1 = gen.topk(k=1)[0]
        best_value = gtop1.value
        best_params = gtop1.record
        if best_value > top1_value:
            top1_value = best_value
            top1 = best_params
    return top1_value, top1
