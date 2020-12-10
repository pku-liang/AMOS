import numpy as np
import time
import heapq
from .measure import *
from .record import Entry



class ParamGenerator(object):
    def get(self, *args, **kwargs):
        raise NotImplementedError()

    def feedback(self, *args, **kwargs):
        raise NotImplementedError()


class FlipFlopParamGenerator(ParamGenerator):
    pass


class QLearningParamGenerator(ParamGenerator):
    def init_Q_table(self):
        self.Q_table = {}
        for x in self.choices:
            entry = {}
            for d in self.directions:
                des = self.move_towards_direction(x, d)
                if self.valid(des):
                    # initial random value
                    entry[self.to_hasable(d)] = (des, np.random.random())
            self.Q_table[self.to_hasable(x)] = entry

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

    def to_hasable(self, value):
        if isinstance(value, list):
            ret = []
            for v in value:
                new_v = self.to_hasable(v)
                ret.append(new_v)
            return tuple(ret)
        return value

    def get_random_direction(self, init):
        choices = []
        for d, (des, q_value) in self.Q_table[self.to_hasable(init)].items():
            choices.append((d, des))
        choice = np.random.randint(0, len(choices))
        return choices[choice]

    def get_q_direction(self, init, eps=0.01):
        if np.random.random() < eps:
            return self.get_random_direction(init)
        max_choice = -1
        max_q = -1
        max_des = None
        for d, (des, q_value) in self.Q_table[self.to_hasable(init)].items():
            if q_value > max_q:
                max_choice = d
                max_q = q_value
                max_des = des
        return max_choice, max_des

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



class EntryGenerator(object):
    def get(self, *args, **kwargs):
        raise NotImplementedError()
    
    def feedback(self, *args, **kwargs):
        raise NotImplementedError()


class SAEntryGenerator(EntryGenerator):
    def __init__(self, eps, record_cls):
        self.eps = eps
        self.entries = []
        self.visited = {}
        self.record_cls = record_cls

    def calculate_p(self, x, best):
        return np.exp((x - best) / 2 * (best + 1e-5))

    def greedy(self):
        return np.random.random() > self.eps

    def sa_select_entry(self, max_num=20):
        assert len(self.entries) > 0
        topk = heapq.nlargest(min(max_num, len(self.entries)), self.entries)
        cand = topk
        best_value = cand[0].value
        ps = list(map(lambda x: self.calculate_p(x.value, best_value), cand))

        num_cand = len(cand)
        for i in range(max_num):
            choice = np.random.randint(0, num_cand)
            if np.random.random() < ps[choice]:
                return cand[i]
        # no chosen, return the best
        return cand[0]

    def get(self, policy="random", repeat=False, max_trial=100):
        for i in range(max_trial):
            if policy == "random" or not self.entries:
                record = self.get_record(policy="random")
            elif policy == "q":
                if self.greedy():
                    entry = self.sa_select_entry()
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

    def feedback(self, record, value):
        entry = Entry(record, value)
        self.visited[str(record)] = value
        heapq.heappush(self.entries, entry)
        self.feedback_value(entry, value)

    def record_from_json(self, obj):
        raise NotImplementedError()

    def get_record(self, entry=None, policy="random"):
        raise NotImplementedError()

    def feedback_value(self, entry, value):
        raise NotImplementedError()

    def valid(self, record):
        return True


def find_optimized_parameters(
    match_results, schedule_gen, schedule_app,
        measure_opt, checker, trials, batch_size=32,
        policy="random", builder=tg_parallel_builder_build,
        runner=pebble_local_runner_run):
    best_value = 1 / MAX_FLOAT
    best_params = None
    if measure_opt.use_rpc:
        runner = pebble_rpc_runner_run
    batch_num = (trials + batch_size - 1) // batch_size
    print("Total search tirals:", trials,
          "\nbatch size:", batch_size,
          "\nbatch num:", batch_num, flush=True)
    tic = time.time()
    for b in range(batch_num):
        print("Search round:", b, flush=True)
        params_lst = []
        for i in range(batch_size):
            if b * batch_size + i < trials:
                params = schedule_gen.get(policy=policy)
                # print(str(params))
                params_lst.append(params)
        assert params_lst
        build_results = builder(
            schedule_app, params_lst, measure_opt, checker)
        run_results = runner(
            build_results, measure_opt)
        for params, res in zip(params_lst, run_results):
            # print(res)
            value = 1 / np.mean([x.value for x in res.costs])  # use absolute performance
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
        print("Current best timecost: ", 1/best_value*1e3, "ms", flush=True)
        if best_params is not None:
            print("Current best params:\n", best_params.to_json(), flush=True)
    toc = time.time()
    print("Search %d trials costs %f seconds" % (trials, toc - tic), flush=True)
    return best_value, best_params
        