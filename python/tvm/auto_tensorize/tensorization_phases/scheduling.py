import tvm
import numpy as np
from .utils import any_factor_split, remap_factors, get_directions, get_factor_lst
from .target import *
from .compute_transform import TransformGenerator, substitute_inputs
from ..search import QLearningParamGenerator
from ..capsule_base import construct_dag
from ..recipe import OperationRole, RecipeStage
from functools import reduce


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


def reconstruct_dag_as_intrin(
        target_dag, main_op, recipe, compute_key, shape_key):
    inputs = list(main_op.input_tensors)
    outputs = [main_op.output(0)]
    # TODO: consider elem op in dag construction
    input_names, output_names, nodes, read_graph, feed_graph = \
        construct_dag(
            recipe, compute_key, shape_key, inputs, outputs, [], outputs)
    print("input names:")
    print(input_names)
    print("output names:")
    print(output_names)
    print("nodes:")
    print(nodes)
    print("read graph:")
    print(read_graph)
    print("feed graph:")
    print(feed_graph)
    output_tensors = reduce(
        lambda x, y: x + y, [nodes[x] for x in output_names], [])
    output = output_tensors[0]
    replace_map = {main_op: output.op}
    result_dag = substitute_inputs(target_dag, replace_map)
    return (result_dag,
            (input_names, output_names, nodes, read_graph, feed_graph))


class ScheduleGenerator(object):
    def __init__(self, intrin_match_result, transform_state):
        recipe = intrin_match_result.recipe
        compute_key = intrin_match_result.compute_key
        shape_key = intrin_match_result.shape_key

        target_main_op = None
        for k, v in transform_state.main_op_map.items():
            target_main_op = v
        assert target_main_op is not None
        self.target_dag, info = reconstruct_dag_as_intrin(
            transform_state.target_dag,
            target_main_op,
            recipe,
            compute_key,
            shape_key)
        (_, _, nodes, _, _) = info

        def cond(cur):
            if cur in recipe.capsules:
                return True
            return False
        capsule_names, read_graph, feed_graph = recipe.serialize_dag(
            cond1=cond)

        operation_role = {}
        capsule_map = {}
        reserve_inner_axis_count = {}
        main_op_reserve_reduce_axis = []
        main_op_reserve_reduce_axis_factor = []
        # These two are not useful for CPUs
        load_from_shared = {}
        store_to_shared = {}
        for name in capsule_names:
            op = nodes[name].op
            capsule_map[op] = name
            spatial_axis, reduce_axis = \
                recipe.get_capsule_compute_reserve_axis(
                    compute_key, shape_key, name)
            reserve_inner_axis_count[op] = len(spatial_axis)
            if name not in read_graph:
                operation_role[op] = OperationRole.load_op
                load_from_shared[op] = 1
            elif name not in feed_graph:
                operation_role[op] = OperationRole.output_op
                store_to_shared[op] = 0
            elif name == recipe.main_capsule_name:
                operation_role[op] = OperationRole.main_op
                for i, red in enumerate(reduce_axis):
                    main_op_reserve_reduce_axis.append(
                        len(op.reduce_axis) - len(reduce_axis) + i)
                    main_op_reserve_reduce_axis_factor.append(
                        int(red.dom.extent))

        self.recipe_stage = RecipeStage(
            operation_role,
            recipe.target,
            recipe.get_name(),
            compute_key,
            shape_key,
            capsule_map,
            reserve_inner_axis_count,
            main_op_reserve_reduce_axis,
            main_op_reserve_reduce_axis_factor,
            load_from_shared,
            store_to_shared,
            recipe.scope
        )


class ScheduleResult(object):
    def __init__(self, schedule, schedule_steps):
        self.schedule = schedule
        self.schedule_steps = schedule_steps
