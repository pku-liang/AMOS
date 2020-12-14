import tvm._ffi
import json
import heapq
import numpy as np
from tvm.runtime import Object
from ..capsule_base import ComputeDAG
from .intrin_match import IntrinMatchResult
from .. import _ffi_api
from ..search import CDParamGenerator, Entry, SAEntryGenerator
from .utils import bi_product, substitute_inputs, softmax
from functools import reduce


@tvm._ffi.register_object("auto_tensorize.TransformState")
class TransformState(Object):
    """
    Args:
    ---
    main_op_map Map for main op
    elem_op_map Map for elementwise op
    axis_map Map for axis
    reverse_axis_map Reverse map for axis
    target_dag Target compute dag
    intrin_dag Intrin compute dag
    """

    def __init__(self, main_op_map, elem_op_map, axis_map, target_dag, intrin_dag):
        self.__init_handle_by_constructor__(
            _ffi_api.TransformState, main_op_map, elem_op_map, axis_map, target_dag, intrin_dag
        )


@tvm._ffi.register_object("auto_tensorize.TransformRequest")
class TransformRequest(Object):
    """
    Args:
    ---
    name:
    axis_map:
    reverse_axis_map:
    time_loops:
    """

    def __init__(self, name, axis_map, reverse_axis_map, space_loops, time_loops, padding=False):
        self.__init_handle_by_constructor__(
            _ffi_api.TransformRequest,
            name,
            axis_map,
            reverse_axis_map,
            space_loops,
            time_loops,
            padding,
        )


def infer_range(vars_to_infer, original_vars, original_range_map):
    """Infer ranges for expressions

    Parameters
    ----------
    vars_to_infer:
    original_vars:
    original_range_map:

    Returns
    -------

    """
    range_map = _ffi_api.InferRange(vars_to_infer, original_vars, original_range_map)
    return range_map


def transform_main_op(init, request):
    """Infer ranges for expressions

    Parameters
    ----------
    init:
    request:

    Returns
    -------

    """
    n = _ffi_api.TransformMainOp(init, request)
    return n


# class UnfoldChoiceGenerator(CDParamGenerator):
#     def __init__(self, num_choices):
#         self.choices = bi_product(num_choices)
#         self.directions = []
#         # d = [0 for _ in range(num_choices)]
#         # self.directions.append(d)
#         for i in range(num_choices):
#             d = [0 for _ in range(num_choices)]
#             d[i] = 1
#             self.directions.append(d)
#         self.init_Q_table()

#     def map_to_hidden(self, factors):
#         return factors

#     def map_from_hidden(self, init):
#         return init

#     def move_towards_direction(self, init, d):
#         ret = []
#         for a, b in zip(init, d):
#             ret.append((a + b) % 2)
#         return ret

#     def valid(self, init):
#         return reduce(lambda x, y: x + y, init, 0) > 0


class UnfoldChoiceGenerator(CDParamGenerator):
    def __init__(self, axis_map):
        self.unfolds = []
        value_map = {}
        value = 1
        k_value = 1
        num_items = 0
        for k, lst in axis_map.items():
            tmp_num_items = len(lst)
            if num_items:
                assert num_items == tmp_num_items
            num_items = tmp_num_items
            for l in lst:
                if l not in value_map:
                    value_map[l] = value
                    value *= 2
            if k not in value_map:
                value_map[k] = k_value
                k_value *= 3
        choices = bi_product(num_items)
        visited = set()
        for bit_vec in choices:
            tmp_set = {}
            for ind, v in enumerate(bit_vec):
                if v:
                    for k, lst in axis_map.items():
                        if k not in tmp_set:
                            tmp_set[k] = set()
                        tmp_set[k].add(lst[ind])
            if tmp_set:
                unique_value = 0
                for k, s in tmp_set.items():
                    tmp_value = 0
                    for v in s:
                        tmp_value += value_map[v]
                    unique_value += tmp_value * value_map[k]
                if unique_value not in visited:
                    visited.add(unique_value)
                    self.unfolds.append(bit_vec)

        self.choices = list(range(len(self.unfolds)))
        self.reverse_map = {self.to_hashable(k): v for v, k in enumerate(self.unfolds)}

        self.directions = [1, -1]
        self.init_Q_table()

    def map_to_hidden(self, factors):
        return self.reverse_map[self.to_hashable(factors)]

    def map_from_hidden(self, init):
        return self.unfolds[init]

    def move_towards_direction(self, init, d):
        des = init + d
        return des

    def valid(self, init):
        return 0 <= init < len(self.choices)


class Record(object):
    def __init__(self, unfold_choice):
        # choice = (des, direction)
        self.unfold_choice = unfold_choice

    def to_json(self):
        return {"unfold": self.unfold_choice}

    def __str__(self):
        return json.dumps(self.to_json())


class TransformGenerator(SAEntryGenerator):
    def __init__(
        self, intrin_match_result, eps=1e-1, log_file="transform_schedule_generator.log", steps=1
    ):
        super(TransformGenerator, self).__init__(eps, Record, steps=steps, log_file=log_file)
        self.init_param_generator(intrin_match_result)
        self.init_score_table()

    def init_param_generator(self, intrin_match_result):
        assert isinstance(intrin_match_result, IntrinMatchResult)
        # match_point_num = -1
        # for k, v in intrin_match_result.axis_map.items():
        #     match_len = len(v)
        #     if match_point_num < 0:
        #         match_point_num = match_len
        #     assert match_point_num == match_len
        # self.unfold_gen = UnfoldChoiceGenerator(match_point_num)
        self.unfold_gen = UnfoldChoiceGenerator(intrin_match_result.axis_map)
        self.generator_lst = [self.unfold_gen]

    def init_score_table(self):
        self.score_table = softmax([0.5 for gen in self.generator_lst])

    def get_generators(self):
        return self.generator_lst

    def record_from_json(self, obj):
        return self.record_cls(obj["unfold"])

    def get_record(self, entry=None, policy="random"):
        if entry is None:
            return self.record_cls(self.unfold_gen.get(policy=policy))
        else:
            return self.record_cls(
                self.unfold_gen.get(hint=entry.record.unfold_choice[0], policy=policy)
            )

    def get_records_mutate_one_generator(self, record, to_mutate, steps):
        unfold = record.unfold_choice

        next_unfold = self.unfold_gen.get_next(unfold[0], to_mutate)

        has_mutate = False

        def helper(_gen, org_val):
            nonlocal has_mutate
            try:
                ret = next(_gen)
                has_mutate = True
            except StopIteration:
                ret = org_val
            return ret

        for s in range(steps):
            unfold = helper(next_unfold, unfold)
            if has_mutate:
                yield self.record_cls(unfold)
            has_mutate = False

    def feedback_value(self, entry, value):
        self.unfold_gen.feedback(*entry.record.unfold_choice, value)


class TransformApplier(object):
    def __init__(self, intrin_match_result):
        assert isinstance(intrin_match_result, IntrinMatchResult)
        self.init_state = TransformState(
            intrin_match_result.main_op_map,
            intrin_match_result.elem_op_map,
            intrin_match_result.axis_map,
            intrin_match_result.target_dag,
            intrin_match_result.intrin_dag,
        )

    def apply_unfold(self, record, state):
        # unfold
        intrin_main_op = None
        target_main_op = None
        for k, v in state.main_op_map.items():
            intrin_main_op = k
            target_main_op = v
        assert intrin_main_op is not None
        assert target_main_op is not None
        intrin_axis = list(intrin_main_op.axis) + list(intrin_main_op.reduce_axis)
        target_axis = list(target_main_op.axis) + list(target_main_op.reduce_axis)
        unfold_choice = record.unfold_choice[0]
        choices = []
        tmp = []
        for axis in intrin_axis:
            tmp.append(state.axis_map[axis])
        tmp = list(zip(*tmp))
        for i, v in enumerate(unfold_choice):
            if v == 1:
                choices.append(tmp[i])
        choices = list(zip(*choices))

        name = ".unfold"
        fwd_axis_map = {}
        rvs_axis_map = {}
        space_loops = []
        time_loops = []

        def flatten(axes, strides):
            ret = 0
            for a, s in zip(axes, strides):
                ret = ret + a * s
            return ret

        for axis, choice in zip(intrin_axis, choices):
            visited = set()
            unique_choice = []
            for c in choice:
                if c not in visited:
                    unique_choice.append(c)
                    visited.add(c)
            unique_stride = []
            stride = 1
            for c in reversed(unique_choice):
                unique_stride.append(stride)
                stride *= int(c.dom.extent)
            unique_stride = list(reversed(unique_stride))
            fwd_axis_map[axis] = flatten(unique_choice, unique_stride)
            for i, (a, s) in enumerate(zip(unique_choice, unique_stride)):
                if i > 0:
                    rvs_axis_map[a] = axis % unique_stride[i - 1] // s
                else:
                    rvs_axis_map[a] = axis // s
            space_loops.extend(unique_choice)

        visited = set()
        for axis in space_loops:
            visited.add(axis)

        for axis in target_axis:
            if axis not in visited:
                time_loops.append(axis)

        request = TransformRequest(name, fwd_axis_map, rvs_axis_map, space_loops, time_loops)
        unfold_state = transform_main_op(state, request)
        return unfold_state

    def apply_fold(self, record, state):
        # fold
        intrin_main_op = None
        target_main_op = None
        for k, v in state.main_op_map.items():
            intrin_main_op = k
            target_main_op = v
        assert intrin_main_op is not None
        assert target_main_op is not None
        intrin_axis = list(intrin_main_op.axis) + list(intrin_main_op.reduce_axis)
        target_axis = list(target_main_op.axis) + list(target_main_op.reduce_axis)

        choices = []
        tmp = []
        for axis in intrin_axis:
            tmp.append(state.axis_map[axis][-1])
        choices = list(tmp)

        name = ".fold"
        fwd_axis_map = {}
        rvs_axis_map = {}
        space_loops = []
        time_loops = []
        need_padding = False

        for axis, choice in zip(intrin_axis, choices):
            factor = int(axis.dom.extent)
            extent = int(choice.dom.extent)
            outer = (extent + factor - 1) // factor
            var = tvm.tir.IterVar([0, outer], axis.var.name + ".o", axis.iter_type)
            fwd_axis_map[axis] = choice % factor
            fwd_axis_map[var] = choice // factor
            rvs_axis_map[choice] = var * factor + axis
            space_loops.append(choice)
            time_loops.append(var)
            if extent < factor:
                need_padding = True

        visited = set()
        for axis in space_loops:
            visited.add(axis)

        for axis in target_axis:
            if axis not in visited:
                time_loops.append(axis)

        request = TransformRequest(
            name, fwd_axis_map, rvs_axis_map, space_loops, time_loops, padding=need_padding
        )
        fold_state = transform_main_op(state, request)
        return fold_state

    def apply(self, record):
        state = self.apply_unfold(record, self.init_state)
        state = self.apply_fold(record, state)
        return state
