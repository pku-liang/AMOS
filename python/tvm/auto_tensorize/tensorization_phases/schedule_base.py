import tvm
from ..utils import *
from ..target import *
from ..search import CDParamGenerator, SAEntryGenerator


class ScheduleComputeInfo(object):
    def __init__(
        self, target_dag, main_op, output_op,
            main_op_id, output_op_id, recipe_stage,
            **kwargs):
        self.target_dag = target_dag
        self.main_op = main_op
        self.output_op = output_op
        self.main_op_id = main_op_id
        self.output_op_id = output_op_id
        self.recipe_stage = recipe_stage
        self.kwargs = kwargs


#####################################################
# Target independent parameter generator
#####################################################
class SplitFactorGenerator(CDParamGenerator):
    def __init__(self, extent, parts):
        assert isinstance(extent, int)
        factor_list = any_factor_split(extent, parts)
        self.choices, self.factor_map, dim, sum_val = remap_factors(
            factor_list)
        self.choice_set = set([tuple(x) for x in self.choices])
        self.directions = get_directions(dim)
        # self.directions = get_partial_directions(dim)
        self.reverse_map = {y: x for x, y in self.factor_map.items()}
        self.init_Q_table()

    def move_towards_direction(self, init, d):
        ret = []
        sum_val = reduce(lambda x, y: x + y, init, 0)
        tmp_sum = 0
        for i, v in enumerate(d):
            ret.append(init[i] + v)
            tmp_sum += ret[-1]
        ret.append(sum_val - tmp_sum)
        return ret

    def map_to_hidden(self, factors):
        return [self.reverse_map[f] for f in factors]

    def map_from_hidden(self, init):
        return [self.factor_map[i] for i in init]

    def valid(self, init):
        # for v in init:
        #     if not (0 <= v <= self.sum_val):
        #         return False
        # return True
        return tuple(init) in self.choice_set

    def diameter(self):
        return len(self.factor_map)


class VectorizeLengthGenerator(CDParamGenerator):
    def __init__(self, target, dtype):
        self.lengths = get_factor_lst(get_vector_length(target, dtype))
        self.choices = list(range(len(self.lengths)))
        self.length_map = {x: y for x, y in zip(self.choices, self.lengths)}
        self.reverse_map = {y: x for x, y in zip(self.choices, self.lengths)}
        self.directions = [1, -1]
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

    def diameter(self):
        return len(self.length_map)


class UnrollStepGenerator(CDParamGenerator):
    def __init__(self, steps):
        self.steps = steps
        self.choices = list(range(len(self.steps)))
        self.length_map = {x: y for x, y in zip(self.choices, self.steps)}
        self.reverse_map = {y: x for x, y in zip(self.choices, self.steps)}
        self.directions = [1, -1]
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

    def diameter(self):
        return len(self.length_map)


class InlineGenerator(CDParamGenerator):
    def __init__(self):
        self.choices = [0, 1]
        self.directions = [1, -1]
        self.init_Q_table()

    def move_towards_direction(self, init, d):
        des = init + d
        return des

    def valid(self, init):
        return 0 <= init <= 1

    def map_to_hidden(self, choice):
        return choice

    def map_from_hidden(self, init):
        return init

    def diameter(self):
        return 2


class SplitKGenerator(CDParamGenerator):
    def __init__(self, steps):
        self.steps = steps
        self.choices = list(range(len(self.steps)))
        self.length_map = {x: y for x, y in zip(self.choices, self.steps)}
        self.reverse_map = {y: x for x, y in zip(self.choices, self.steps)}
        self.directions = [1, -1]
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

    def diameter(self):
        return len(self.length_map)


#####################################################
# Target specific schedule generator and applier
#####################################################
""" We aim to design special scheduler for tensorize.
    So we only accept a particular structure of input DAG.
    The input DAG only has one intrinsic match point,
    the other nodes in the DAG should not contain reduction.
    For other kind of input DAG, a previous dispatcher should
    cut the DAG into proper sub-DAGs and assign those we don't
    handle to other schedulers such as Ansor, FlexTensor, or AutoTVM.
"""
class AcceleratorScheduleGenerator(SAEntryGenerator):
    def get_schedule_compute_info(self):
        raise NotImplementedError()
