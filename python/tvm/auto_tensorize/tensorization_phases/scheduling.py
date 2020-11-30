import tvm
import numpy as np
from .utils import any_factor_split, remap_factors, get_directions, get_factor_lst
from .target import *
from .compute_transform import TransformGenerator
from ..search import QLearningParamGenerator


class State(object):
    def __init__(self):
        self.op_to_iters = {}
        self.is_reduce_axis = set()


class SplitFactorGenerator(QLearningParamGenerator):
    def __init__(self, extent, parts):
        assert isinstance(extent, int)
        factor_list = any_factor_split(extent, parts)
        self.choices, self.factor_map, dim, self.sum_val = remap_factors(
            factor_list)
        self.directions = get_directions(dim)
        self.reverse_map = {y: x for x, y in self.factor_map.items()}
        self.init_Q_table()

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


class VectorizeLengthGenerator(QLearningParamGenerator):
    def __init__(self, target, dtype):
        self.lengths = get_factor_lst(get_vector_length(target, dtype))
        self.choices = list(range(len(self.lengths)))
        self.length_map = {x: y for x, y in zip(self.choices, self.lengths)}
        self.reverse_map = {y: x for x, y in zip(self.choices, self.lengths)}
        self.directions = [0, 1, -1]
        self.init_Q_table()

    def move_towards_direction(self, init, d):
        des = init + d
        return des

    def valid(self, init):
        return 0 <= init < len(self.lengths)

    def map_to_hidden(self, length):
        return self.reverse_map[length]

    def map_from_hidden(self, init):
        return self.length_map[init]


class UnrollStepGenerator(QLearningParamGenerator):
    def __init__(self, steps):
        self.steps = steps
        self.choices = list(range(len(self.steps)))
        self.length_map = {x: y for x, y in zip(self.choices, self.steps)}
        self.reverse_map = {y: x for x, y in zip(self.choices, self.steps)}
        self.directions = [0, 1, -1]
        self.init_Q_table()

    def move_towards_direction(self, init, d):
        des = init + d
        return des

    def valid(self, init):
        return 0 <= init < len(self.steps)

    def map_to_hidden(self, length):
        return self.reverse_map[length]

    def map_from_hidden(self, init):
        return self.length_map[init]


class ScheduleGenerator(object):
    def __init__(self, intrin_match_result, transform_state):
        self.target_dag = transform_state.target_dag
        


class ScheduleResult(object):
    def __init__(self, schedule, schedule_steps):
        self.schedule = schedule
        self.schedule_steps = schedule_steps
