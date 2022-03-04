import numpy as np
import time
import heapq
from .measure import *
from .record import Entry
from ..utils import *
import queue
import logging
import json
import sys
import os


class ParamGenerator(object):
    def get(self, *args, **kwargs):
        raise NotImplementedError()

    def feedback(self, *args, **kwargs):
        raise NotImplementedError()


class FlipFlopParamGenerator(ParamGenerator):
    pass


class CDParamGenerator(ParamGenerator):
    def init_Q_table(self):
        self.Q_table = {}
        visited = set()
        q = queue.Queue()
        for x in self.choices:
            q.put(x)
            visited.add(self.to_hashable(x))
        while not q.empty():
            x = q.get()
            entry = {}
            for d in self.directions:
                des = self.move_towards_direction(x, d)
                if self.valid(des):
                    # initial random value
                    entry[self.to_hashable(d)] = (des, np.random.random())
                    if self.to_hashable(des) not in visited:
                        q.put(des)
                        visited.add(self.to_hashable(des))
            self.Q_table[self.to_hashable(x)] = entry

    def feedback(self, init, direction, reward):
        pass

    def map_to_hidden(self, factors):
        raise NotImplementedError()

    def map_from_hidden(self, init):
        raise NotImplementedError()

    def move_towards_direction(self, init, d):
        raise NotImplementedError()

    def valid(self, init):
        raise NotImplementedError()

    def to_hashable(self, value):
        if isinstance(value, list):
            ret = []
            for v in value:
                new_v = self.to_hashable(v)
                ret.append(new_v)
            return tuple(ret)
        return value

    def get_random_direction(self, init):
        choices = []
        for d, (des, q_value) in self.Q_table[self.to_hashable(init)].items():
            choices.append((d, des))
        choice = np.random.randint(0, len(choices))
        return choices[choice]

    def get_q_direction(self, init, eps=0.01):
        # if np.random.random() < eps:
        #     return self.get_random_direction(init)
        # max_choice = -1
        # max_q = -1
        # max_des = None
        # for d, (des, q_value) in self.Q_table[self.to_hashable(init)].items():
        #     if q_value > max_q:
        #         max_choice = d
        #         max_q = q_value
        #         max_des = des
        # return max_choice, max_des
        print("Warning: no implementation for get q direction.")
        return self.get_random_direction(init)

    def get(self, hint=None, policy="random"):
        if hint is None:
            choice = np.random.randint(0, len(self.choices))
            hint = self.choices[choice]
        else:
            hint = self.map_to_hidden(hint)
        if policy == "random":
            direction, des = self.get_random_direction(hint)
        elif policy == "q":
            direction, des = self.get_q_direction(hint)
        else:
            raise RuntimeError("Unknown policy: %s" % policy)
        return self.map_from_hidden(des), direction

    def get_all(self):
        ret = []
        for choice in self.choices:
            ret.append((self.map_from_hidden(choice), -1))
        return ret

    def diameter(self):
        raise NotImplementedError()

    def get_directions_from(self, init, may_be_self):
        ret = []
        if may_be_self != self:
            return ret
        for d in self.directions:
            if self.to_hashable(d) in self.Q_table[self.to_hashable(init)]:
                ret.append(d)
        return ret

    def get_next_via_direction(self, init, d):
        if self.to_hashable(d) not in self.Q_table[self.to_hashable(init)]:
            raise RuntimeError("Invalid direction")
        return (self.Q_table[self.to_hashable(init)][self.to_hashable(d)][0], d)

    def get_next(self, init, may_be_self):
        # init = self.map_to_hidden(init)
        # ds = self.get_directions_from(init, may_be_self)
        # for d in ds:
        #     try:
        #         ret = self.get_next_via_direction(init, d)
        #         yield self.map_from_hidden(ret[0]), ret[1]
        #     except RuntimeError as e:
        #         pass
        if self == may_be_self:
            yield self.get()


class EntryGenerator(object):
    def get(self, *args, **kwargs):
        raise NotImplementedError()

    def feedback(self, *args, **kwargs):
        raise NotImplementedError()


class SAEntryGenerator(EntryGenerator):
    def __init__(
        self,
        eps,
        record_cls,
        steps=1,
        log_file="sa_entry_generator_record.log",
        allow_repeat=False,
        topk=20,
    ):
        self.eps = eps
        self.entries = []
        self.visited = {}
        self.record_cls = record_cls
        self.steps = steps
        self.log_file = log_file
        self.allow_repeat = allow_repeat
        self.topk_num = topk
        self.init_logger()
        self.last_choice = None
        self.last_value = 0.0
        self.gen = self._get_next(self.allow_repeat)

    def init_logger(self):
        if self.log_file is not None:
            print("Logging to %s..." % self.log_file, flush=True)
            self.logger = open(self.log_file, "a")
        else:
            print("Logging to %s..." % "devnull", flush=True)
            self.logger = open(os.devnull, "w")

    def init_param_generator(self, *args):
        raise NotImplementedError()

    def init_score_table(self, *args):
        raise NotImplementedError()

    def calculate_p(self, x, best):
        return np.exp((x - best) / (2 * (best + 1e-5)))

    def greedy(self, cnt):
        p = np.random.random()
        q = self.eps / (cnt // 100 + 1)
        return p > q

    def sa_select_entry(self, max_num=20):
        assert len(self.entries) > 0
        topk = heapq.nsmallest(min(max_num, len(self.entries)), self.entries)
        cand = topk
        best_value = cand[0].value
        ps = list(map(lambda x: self.calculate_p(x.value, best_value), cand))

        num_cand = len(cand)
        for i in range((max_num + 3) // 4):
            choice = np.random.randint(0, num_cand)
            if np.random.random() < ps[choice]:
                return cand[choice]
        # no chosen, return the best
        return cand[0]

    def topk(self, k=1):
        topk = heapq.nsmallest(min(k, len(self.entries)), self.entries)
        return topk

    def has_entry(self):
        return len(self.entries) > 0

    def num_entries(self):
        return len(self.entries)

    def get(self, policy="random", repeat=False, max_trial=100):
        for i in range(max_trial):
            if policy == "random" or not self.entries:
                record = self.get_record(policy="random")
            elif policy == "q":
                if self.greedy(i + 1):
                    entry = self.sa_select_entry(max_num=self.topk_num)
                    record = self.get_record(entry=entry, policy="q")
                else:
                    record = self.get_record(policy="random")
            elif policy == "greedy":
                return self.entries[0]
            else:
                raise RuntimeError("Unknown policy: %s" % policy)
            if str(record) not in self.visited:
                if self.valid(record):
                    self.visited[str(record)] = 0.0
                    return record
            elif repeat:
                self.feedback(record, self.visited[str(record)])
                return record
            else:
                self.feedback(record, self.visited[str(record)])
        print("It seems hard to find new candidates...", flush=True)
        return self.entries[0].record

    def get_all(self):
        raise NotImplementedError()

    def update_score_table(self, value):
        if self.last_choice is not None:
            i = self.last_choice
            if value > self.last_value:
                self.score_table[i] += 1
                self.score_table[i] = min(1.0, self.score_table[i])
            elif value == self.last_value:
                self.score_table[i] += 0.5
                self.score_table[i] = min(1.0, self.score_table[i])
            else:
                self.score_table[i] -= 1
                self.score_table[i] = max(0.0, self.score_table[i])
            self.score_table = softmax(self.score_table)

    def feedback(self, record, value, log_to_file=True):
        entry = Entry(record, value)
        self.visited[str(record)] = value
        heapq.heappush(self.entries, entry)
        # self.feedback_value(entry, value)
        self.update_score_table(value)
        # store the record
        log = json.dumps(entry.to_json())
        if log_to_file:
            print(log, file=self.logger, flush=True)

    def record_from_json(self, obj):
        raise NotImplementedError()

    def clear(self, log_file):
        self.entries = []
        self.visited = {}
        self.last_choice = None
        self.last_value = 0.0
        self.gen = self._get_next(repeat=self.allow_repeat)
        self.init_score_table()
        self.log_file = log_file
        self.logger.close()
        self.init_logger()

    def load_from_file(self, file_name, clear=False):
        if clear:
            print("Clearing...")
            self.clear(file_name)
        print("Loading from file %s..." % file_name, flush=True)
        # assert file_name != self.log_file, "Please do not use the same log file."
        assert not self.entries, "Please clear the generator first (be caution!)."
        count = 0
        best = 0.0
        with open(file_name, "r") as fin:
            for line in fin:
                count += 1
                obj = json.loads(line)
                record = self.record_from_json(obj["record"])
                value = obj["value"]
                best = max(value, best)
                self.feedback(record, value, False)
        print(
            "Load %d entries! The best known is %f ms" % (count, 1 / (best + 1e-10) * 1e3),
            flush=True,
        )

    def get_best_entry(self):
        assert self.entries
        return self.entries[0]

    def get_record(self, entry=None, policy="random"):
        raise NotImplementedError()

    def feedback_value(self, entry, value):
        raise NotImplementedError()

    def valid(self, record):
        return True

    def get_generators(self):
        raise NotImplementedError()

    def get_records_mutate_one_generator(self, record, to_mutate, steps):
        raise NotImplementedError()

    def _get_next(self, repeat=False):
        count = 0
        while True:
            if not self.entries:
                self.last_choice = None
                self.last_value = 0.0
                count += 1
                yield self.get(repeat=repeat)
            else:
                if self.greedy(count):
                    entry = self.sa_select_entry(max_num=self.topk_num)
                    record = entry.record
                    self.last_value = entry.value
                    # select one generator
                    has_output = False
                    for i, gen_x in enumerate(self.get_generators()):
                        # if np.random.random() > self.score_table[i]:
                        #     continue
                        self.last_choice = i
                        for next_record in self.get_records_mutate_one_generator(
                            record, gen_x, self.steps
                        ):
                            if str(next_record) not in self.visited:
                                if self.valid(next_record):
                                    has_output = True
                                    self.visited[str(next_record)] = 0.0
                                    count += 1
                                    yield next_record
                    # fallback
                    if not has_output:
                        self.last_choice = None
                        self.last_value = 0.0
                        count += 1
                        yield self.get(repeat=repeat)
                else:
                    self.last_choice = None
                    self.last_value = 0.0
                    count += 1
                    yield self.get(repeat=repeat)

    def refresh(self):
        self.gen = self._get_next(repeat=self.allow_repeat)

    def get_next(self, policy=""):
        if policy:
            return self.get(policy=policy)
        return next(self.gen)


def find_optimized_parameters(
    match_results,
    schedule_gen,
    schedule_app,
    measure_opt,
    checker,
    trials,
    search_group_size=16,
    policy="",
    builder=tg_parallel_builder_build,
    runner=pebble_local_runner_run,
    verbose=False,
    build_parallel=1,
    run_parallel=1,
):
    best_value = 1 / MAX_FLOAT
    best_params = None
    if schedule_gen.has_entry():
        top1 = schedule_gen.topk(k=1)[0]
        best_value = top1.value
        best_params = top1.record
    if measure_opt.use_rpc:
        assert 0
        runner = pebble_rpc_runner_run
    search_group_num = (trials + search_group_size - 1) // search_group_size
    print(
        "Total search tirals:",
        trials,
        "\nbatch size:",
        search_group_size,
        "\nbatch num:",
        search_group_num,
        flush=True,
    )
    tic = time.time()
    for b in range(search_group_num):
        print("Search round:", b, flush=True)
        schedule_gen.refresh()
        params_lst = []
        for i in range(search_group_size):
            if b * search_group_size + i < trials:
                # params = schedule_gen.get(policy=policy)
                params = schedule_gen.get_next(policy=policy)
                # my_params = {
                # params.from_json(my_params)
                # print(str(params))
                params_lst.append(params)
        assert params_lst
        build_results = builder(
            schedule_app, params_lst, measure_opt, checker, n_parallel=build_parallel
        )
        run_results = runner(build_results, measure_opt, n_parallel=run_parallel)
        for params, res in zip(params_lst, run_results):
            if verbose:
                print(res)
            # use absolute performance
            value = 1 / np.mean([x.value for x in res.costs])
            if value > 1 / MAX_FLOAT:  # valid results
                schedule_gen.feedback(params, value)
            if value > best_value:
                # print(np.mean([x.value for x in res.costs]))
                # cost = evaluate_params(
                #     schedule_app,
                #     params,
                #     measure_opt)
                # print("Re-evaluate: %f ms" % cost, flush=True)
                best_value = value
                best_params = params
        print("Current best timecost: ", 1 / best_value * 1e3, "ms", flush=True)
        if best_params is not None:
            print("Current best params:\n", best_params.to_json(), flush=True)
    toc = time.time()
    print("Search %d trials costs %f seconds" % (trials, toc - tic), flush=True)
    return best_value, best_params


def find_optimized_parameters_v2(
    match_results,
    schedule_gen,
    schedule_app,
    measure_opt,
    checker,
    trials,
    search_group_size=5,
    policy="",
    builder=tg_parallel_builder_build,
    runner=pebble_local_runner_run,
    verbose=False,
    build_parallel=1,
    run_parallel=1,
):
    best_value = 1 / MAX_FLOAT
    best_params = None
    if schedule_gen.has_entry():
        top1 = schedule_gen.topk(k=1)[0]
        best_value = top1.value
        best_params = top1.record
    if measure_opt.use_rpc:
        runner = pebble_rpc_runner_run
    search_group_num = (trials + search_group_size - 1) // search_group_size
    if verbose:
        print(
            "Total search tirals:",
            trials,
            "\nbatch size:",
            search_group_size,
            "\nbatch num:",
            search_group_num,
            flush=True,
        )
    tic = time.time()
    while True:
        for b in range(search_group_num):
            if verbose:
                print("Search round:", b, flush=True)
            schedule_gen.refresh()
            params_lst = []
            for i in range(search_group_size):
                if b * search_group_size + i < trials:
                    # params = schedule_gen.get(policy=policy)
                    params = schedule_gen.get_next(policy=policy)
                    # print(str(params))
                    params_lst.append(params)
            assert params_lst
            build_results = builder(
                schedule_app, params_lst, measure_opt, checker, n_parallel=build_parallel
            )
            run_results = runner(build_results, measure_opt, n_parallel=run_parallel)

            max_value = 1 / MAX_FLOAT
            for params, res in zip(params_lst, run_results):
                if verbose:
                    print(res)
                # use absolute performance
                value = 1 / np.mean([x.value for x in res.costs])
                max_value = max(max_value, value)
                if value > 1 / MAX_FLOAT:  # valid results
                    schedule_gen.feedback(params, value)
                if value > best_value:
                    # print(np.mean([x.value for x in res.costs]))
                    # cost = evaluate_params(
                    #     schedule_app,
                    #     params,
                    #     measure_opt)
                    # print("Re-evaluate: %f ms" % cost, flush=True)
                    best_value = value
                    best_params = params

            if verbose:
                print("Current best timecost: ", 1 / best_value * 1e3, "ms", flush=True)
            else:
                print(f"iteration={b+1}: {max_value}/{best_value}", flush=True)
            if best_params is not None and verbose:
                print("Current best params:\n", best_params.to_json(), flush=True)
        yield best_value, best_params
    toc = time.time()
    if verbose:
        print("Search %d trials costs %f seconds" % (trials, toc - tic), flush=True)
    return best_value, best_params
