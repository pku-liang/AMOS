import time
from .measure import MAX_FLOAT


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