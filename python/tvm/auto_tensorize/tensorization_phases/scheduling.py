import tvm
import numpy as np
from .utils import any_factor_split, remap_factors, get_directions, get_factor_lst
from .target import *


class State(object):
    def __init__(self):
        self.op_to_iters = {}
        self.is_reduce_axis = set()


class ParamGenerator(object):
    def feedback(self, init, direction, reward):
        raise NotImplementedError()

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
            return tuple(value)
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
        return self.map_from_hidden(des)


class SplitFactorGenerator(ParamGenerator):
    def __init__(self, extent, parts):
        assert isinstance(extent, int)
        factor_list = any_factor_split(extent, parts)
        self.choices, self.factor_map, dim, self.sum_val = remap_factors(factor_list)
        self.directions = get_directions(dim)
        self.reverse_map = {y:x for x, y in self.factor_map.items()}
        self.Q_table = {}
        for x in self.choices:
            entry = {}
            for d in self.directions:
                des = self.move_towards_direction(x, d)
                if self.valid(des):
                    # initial random value
                    entry[d] = (des, np.random.random())
            self.Q_table[tuple(x)] = entry

    def move_towards_direction(self, init, d):
        ret = []
        tmp_sum = 0
        for i, v in enumerate(d):
            ret.append(init[i] + v)
            tmp_sum += ret[-1]
        ret.append(self.sum_val - tmp_sum)
        return ret

    def map_to_hidden(self, factors):
        return [self.reverse_map[f] for f in factors]

    def map_from_hidden(self, init):
        return [self.factor_map[i] for i in init]

    def valid(self, init):
        for v in init:
            if not (0 <= v <= self.sum_val):
                return False
        return True

    def feedback(self, init, direction, reward):
        # TODO: finish the Q-learning feedback
        pass


class VectorizeLengthGenerator(ParamGenerator):
    def __init__(self, target, dtype):
        self.lengths = get_factor_lst(get_vector_length(target, dtype))
        self.choices = list(range(len(self.lengths)))
        self.length_map = {x:y for x, y in zip(self.choices, self.lengths)}
        self.reverse_map = {y:x for x, y in zip(self.choices, self.lengths)}
        self.directions = [0, 1, -1]
        self.Q_table = {}
        for x in self.choices:
            entry = {}
            for d in self.directions:
                des = self.move_towards_direction(x, d)
                if self.valid(des):
                    entry[d] = (des, np.random.random())
            self.Q_table[x] = entry

    def move_towards_direction(self, init, d):
        des = init + d
        return des

    def valid(self, init):
        return 0 <= init < len(self.lengths)

    def map_to_hidden(self, length):
        return self.reverse_map[length]

    def map_from_hidden(self, init):
        return self.length_map[init]

    def feedback(self, init, direction, reward):
        pass


class UnrollStepGenerator(ParamGenerator):
    def __init__(self, steps):
        self.steps = steps
        self.choices = list(range(len(self.steps)))
        self.length_map = {x:y for x, y in zip(self.choices, self.steps)}
        self.reverse_map = {y:x for x, y in zip(self.choices, self.steps)}
        self.directions = [0, 1, -1]
        self.Q_table = {}
        for x in self.choices:
            entry = {}
            for d in self.directions:
                des = self.move_towards_direction(x, d)
                if self.valid(des):
                    entry[d] = (des, np.random.random())
            self.Q_table[x] = entry

    def move_towards_direction(self, init, d):
        des = init + d
        return des

    def valid(self, init):
        return 0 <= init < len(self.steps)

    def map_to_hidden(self, length):
        return self.reverse_map[length]

    def map_from_hidden(self, init):
        return self.length_map[init]

    def feedback(self, init, direction, reward):
        pass


class ScheduleParams(object):
    def __init__(self):
        pass


class ScheduleStep(object):
    pass


class TilingStep(ScheduleStep):
    pass


class BindingStep(ScheduleStep):
    pass


class CacheReadStep(ScheduleStep):
    pass


class ComputeAtStep(ScheduleStep):
    pass


class UnrollStep(ScheduleStep):
    pass


class ScheduleResult(object):
    def __init__(self, schedule, schedule_steps):
        self.schedule = schedule
        self.schedule_steps = schedule_steps