import tvm._ffi
import json
import heapq
import numpy as np
from tvm.runtime import Object
from ..capsule_base import ComputeDAG
from .intrin_match import IntrinMatchResult
from .. import _ffi_api
from ..search import QLearningParamGenerator, Entry
from .utils import bi_product
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

    def __init__(self, main_op_map, elem_op_map,
                 axis_map, target_dag, intrin_dag):
        self.__init_handle_by_constructor__(
            _ffi_api.TransformState,
            main_op_map,
            elem_op_map,
            axis_map,
            target_dag,
            intrin_dag)


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

    def __init__(self, name, axis_map, reverse_axis_map,
                 space_loops, time_loops, padding=False):
        self.__init_handle_by_constructor__(
            _ffi_api.TransformRequest,
            name,
            axis_map,
            reverse_axis_map,
            space_loops,
            time_loops,
            padding)


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
    range_map = _ffi_api.InferRange(
        vars_to_infer, original_vars, original_range_map)
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
    n = _ffi_api.TransformMainOp(
        init, request)
    return n


def substitute_inputs(org_dag, op_map):
    """Infer ranges for expressions

    Parameters
    ----------
    org_dag: ComputeDAG
    op_map: dict of {Operation: Operation}

    Returns
    -------
    ComputeDAG
    """
    n = _ffi_api.SubstituteInputs(
        org_dag, op_map)
    return n


class UnfoldChoiceGenerator(QLearningParamGenerator):
    def __init__(self, num_choices):
        self.choices = bi_product(num_choices)
        self.directions = []
        # d = [0 for _ in range(num_choices)]
        # self.directions.append(d)
        for i in range(num_choices):
            d = [0 for _ in range(num_choices)]
            d[i] = 1
            self.directions.append(d)
        self.init_Q_table()

    def map_to_hidden(self, factors):
        return factors

    def map_from_hidden(self, init):
        return init

    def move_towards_direction(self, init, d):
        ret = []
        for a, b in zip(init, d):
            ret.append((a + b) % 2)
        return ret

    def valid(self, init):
        return reduce(lambda x, y: x + y, init, 0) > 0


class Record(object):
    def __init__(self, unfold_choice):
        # choice = (des, direction)
        self.unfold_choice = unfold_choice

    def to_json(self):
        return {"unfold": self.unfold_choice}

    def __str__(self):
        return json.dumps(self.to_json())


class TransformGenerator(object):
    def __init__(self, intrin_match_result, eps=1e-1):
        assert isinstance(intrin_match_result, IntrinMatchResult)
        match_point_num = -1
        for k, v in intrin_match_result.axis_map.items():
            match_len = len(v)
            if match_point_num < 0:
                match_point_num = match_len
            assert match_point_num == match_len
        self.unfold_gen = UnfoldChoiceGenerator(match_point_num)
        self.eps = eps

        self.record_cls = Record
        self.entries = []
        self.visited = {}

    def calculate_p(self, x, best):
        return np.exp((x - best) / 2 * (best + 1e-5))

    def greedy(self):
        return np.random.random() > self.eps

    def sa_select_entry(self, max_num=20):
        assert len(self.entries) > 0
        cand = heapq.nlargest(min(max_num, len(self.entries)), self.entries)
        best_value = cand[0].value
        ps = list(map(lambda x: self.calculate_p(x.value, best_value), cand))
        num_cand = len(cand)
        for i in range(max_num):
            choice = np.random.randint(0, num_cand)
            if np.random.random() < ps[choice]:
                return cand[i]
        # no chosen, return the best
        return cand[0]

    def record_from_json(self, obj):
        return self.record_cls(obj["unfold"])

    def get(self, policy="random", repeat=False, max_trial=100):
        for i in range(max_trial):
            if policy == "random" or not self.entries:
                record = self.record_cls(
                    self.unfold_gen.get(policy="random"))
            elif policy == "q":
                if self.greedy():
                    entry = self.sa_select_entry()
                    record = self.record_cls(
                        self.unfold_gen.get(
                            hint=entry.record.unfold_choice[0], policy="q"))
                else:
                    record = self.record_cls(
                        self.unfold_gen.get(policy="random"))
            elif policy == "greedy":
                return self.entries[0]
            if str(record) not in self.visited:
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
        self.unfold_gen.feedback(*entry.record.unfold_choice, value)


class TransformApplier(object):
    def __init__(self, intrin_match_result):
        assert isinstance(intrin_match_result, IntrinMatchResult)
        self.init_state = TransformState(
            intrin_match_result.main_op_map,
            intrin_match_result.elem_op_map,
            intrin_match_result.axis_map,
            intrin_match_result.target_dag,
            intrin_match_result.intrin_dag
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
        intrin_axis = (list(intrin_main_op.axis)
                       + list(intrin_main_op.reduce_axis))
        target_axis = (list(target_main_op.axis)
                       + list(target_main_op.reduce_axis))
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
                    rvs_axis_map[a] = axis % unique_stride[i-1] // s
                else:
                    rvs_axis_map[a] = axis // s
            space_loops.extend(unique_choice)

        visited = set()
        for axis in space_loops:
            visited.add(axis)

        for axis in target_axis:
            if axis not in visited:
                time_loops.append(axis)

        request = TransformRequest(
            name,
            fwd_axis_map,
            rvs_axis_map,
            space_loops,
            time_loops)
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
        intrin_axis = (list(intrin_main_op.axis)
                       + list(intrin_main_op.reduce_axis))
        target_axis = (list(target_main_op.axis)
                       + list(target_main_op.reduce_axis))

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
            var = tvm.tir.IterVar(
                [0, outer], axis.var.name + ".o", axis.iter_type)
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
            name,
            fwd_axis_map,
            rvs_axis_map,
            space_loops,
            time_loops,
            padding=need_padding)
        fold_state = transform_main_op(state, request)
        return fold_state

    def apply(self, record):
        state = self.apply_unfold(record, self.init_state)
        state = self.apply_fold(record, state)
        return state
