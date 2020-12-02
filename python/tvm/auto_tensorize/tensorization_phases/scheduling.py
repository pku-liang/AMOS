import tvm
import numpy as np
from .utils import any_factor_split, remap_factors, get_directions, get_factor_lst
from .target import *
from .compute_transform import TransformGenerator, substitute_inputs
from ..search import QLearningParamGenerator
from ..capsule_base import construct_dag
from ..recipe import OperationRole, RecipeStage, InstructionScope
from functools import reduce


class State(object):
    def __init__(self):
        self.op_to_iters = {}
        self.is_reduce_axis = set()


#####################################################
# Target independent parameter generator
#####################################################
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


#####################################################
# Target specific parameter generator
#####################################################
class CUDAKernelParamGenerator(QLearningParamGenerator):
    pass


####################################################
# tools
####################################################
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


def can_inline(op, dag):
    """
    op: tvm.te.Operation
    dag: ComputeDAG
    """
    if op not in dag.feed_graph:
        return False
    if not isinstance(op, tvm.te.ComputeOp):
        return False
    if len(op.reduce_axis) > 0:
        return False


#####################################################
# Target specific schedule generator
#####################################################
class ScheduleGenerator(object):
    pass


class CUDAScheduleGenerator(ScheduleGenerator):
    def __init__(self, intrin_match_result, transform_state):
        self.intrin_match_result = intrin_match_result
        # get match recipe info
        recipe = intrin_match_result.recipe
        compute_key = intrin_match_result.compute_key
        shape_key = intrin_match_result.shape_key
        # get main op
        target_main_op = None
        for k, v in transform_state.main_op_map.items():
            target_main_op = v
        assert target_main_op is not None
        # insert intrinsic dag
        self.target_dag, info = reconstruct_dag_as_intrin(
            transform_state.target_dag,
            target_main_op,
            recipe,
            compute_key,
            shape_key)
        # nodes: dict {capsule name : new tensor}
        (_, _, nodes, _, _) = info
        # get new main op
        self.main_op = nodes[recipe.main_capsule_name][0].op
        # get main op id
        self.main_op_id = 0
        for op in self.target_dag.op_lst:
            if op == self.main_op:
                break
            self.main_op_id += 1
        ###################################
        # fill the recipe stage info
        # analyze the intrinsic dag
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

        load_from_shared = {}
        store_to_shared = {}
        self.output_op = None
        for name in capsule_names:
            op = nodes[name][0].op
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
                self.output_op = op
            elif name == recipe.main_capsule_name:
                operation_role[op] = OperationRole.main_op
                for i, red in enumerate(reduce_axis):
                    main_op_reserve_reduce_axis.append(
                        len(op.reduce_axis) - len(reduce_axis) + i)
                    main_op_reserve_reduce_axis_factor.append(
                        int(red.dom.extent))
        assert self.output_op is not None
        # construct recipe stage
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
        # the state during schedule
        # this is only used to guide primitive combination
        self.state = {
            "inlined": set(),
            "main_op_reduce_axis": [],
            "output_op_axis": [],
            "last_op_axis": []
        }
        # the parameters during schedule
        # this is only used to provide paramters
        self.params = {
            "vectoirze": None,
            "spatial_factors": [],
            "reduce_factors": [],
            "last_factors": []
        }
        # some constants
        self.warp_size = 32
        self.bx = tvm.te.thread_axis("blockIdx.x")
        self.ty = tvm.te.thread_axis("threadIdx.y")
        self.tx = tvm.te.thread_axis("threadIdx.x")
        self.obx = tvm.te.thread_axis("blockIdx.x")
        self.oty = tvm.te.thread_axis("threadIdx.y")
        self.otx = tvm.te.thread_axis("threadIdx.x")
        self.reduce_tiling_parts = 3
        self.spatial_tiling_parts = 3

    def initialize_state(self):
        raise NotImplementedError()

    def initialize_parameters(self):
        raise NotImplementedError()

    def get_main_op_outermost_last_reduce_axis(self):
        assert len(self.state["main_op_reduce_axis"]) > 0
        assert isinstance(self.state["main_op_reduce_axis"][0], list)
        assert len(self.state["main_op_reduce_axis"][0]) > 0
        return self.state["main_op_reduce_axis"][0][-1]

    def get_output_op_third_innermost_last_axis(self):
        assert len(self.state["output_op_axis"]) > 2
        assert isinstance(self.state["output_op_axis"][-3], list)
        assert len(self.state["output_op_axis"][-3]) > 0
        return self.state["output_op_axis"][-3][-1]

    def get_last_op_second_innermost_last_axis(self):
        assert len(self.state["last_op_axis"]) > 0
        assert isinstance(self.state["last_op_axis"][-1], list)
        assert len(self.state["last_op_axis"][-1]) > 0
        return self.state["last_op_axis"][-1][-2]

    def get_main_op_warp_numbers(self):
        assert len(self.state["spatial_factors"]) > 0
        ret = 1
        for part in self.state["spatial_factors"]:
            assert len(part) > 0
            ret *= part[-1]
        return ret

    def get_main_op_reduce_axis_factors(self, number):
        assert len(self.state["reduce_factors"]) >= number
        return self.state["reduce_factors"][:number]

    def get_output_op_axis_factors(self, number):
        assert len(self.state["spatial_factors"]) >= number
        return self.state["spatial_factors"][:number]

    def get_last_op_axis_factors(self, number):
        assert len(self.state["last_factors"]) >= number
        return self.state["last_factors"][:number]

    def get_last_op_warp_numbers(self):
        assert len(self.state["last_factors"]) > 0
        ret = 1
        for part in self.state["last_factors"]:
            assert len(part) > 0
            ret *= part[-1]
        return ret

    def get_vectorize_length(self):
        assert self.params["vectorize"] is not None
        return self.params["vectorize"]

    def check_parameter_ready(self):
        raise NotImplementedError()

    def inline(self, op_id, op, sch, X):
        if op in self.recipe_stage.operation_role:
            pass
        else:
            if can_inline(op, self.target_dag):
                sch[X(op)].compute_inline()
                self.state["inlined"].add(op)

    def cache_read(self, op_id, op, sch, X):
        if op in self.recipe_stage.operation_role:
            return
        if op in self.state["inlined"]:
            return
        if len(op.reduce_axis) == 0:
            return
        if not op in self.target_dag.feed_graph:
            return
        do_cache_read_for_load = False
        do_cache_read_for_last = False
        consumers = self.target_dag.feed_graph[op]
        if len(consumers) <= 0:
            return
        if consumers[0] in self.recipe_stage.operation_role:
            if self.recipe_stage.operation_role[consumers[0]] == OperationRole.load_op:
                if len(consumers) == 1:
                    do_cache_read_for_load = True
        if consumers[0] == self.target_dag.op_lst[-1]:
            # the last op
            if len(consumers) == 1:
                do_cache_read_for_last = True
        
        # can't do both
        assert not (do_cache_read_for_load and do_cache_read_for_last)
        
        if do_cache_read_for_load:
            S = sch.cache_read(X(op).output(0), "shared", [X(x) for x in consumers])
            axis = self.get_main_op_outermost_last_reduce_axis()
            # compute at to main op
            sch[S].compute_at(sch[X(self.main_op)], axis)
            warp_num = self.get_main_op_warp_numbers()
            vec_len = self.get_vectorize_length()
            fused = sch[S].fuse(*sch[S].op.axis)
            fused, vectorized = sch[S].split(fused, factor=vec_len)
            fused, thread_level = sch[S].split(fused, factor=self.warp_size)
            fused, warp_level = sch[S].split(fused, factor=warp_num)
            sch[S].bind(thread_level, self.tx)
            sch[S].bind(warp_level, self.ty)
            sch[S].vectorize(vectorized)

        if do_cache_read_for_last:
            last_ops = [X(x) for x in consumers]
            S = sch.cache_read(X(op).output(0), "shared", last_ops)
            axis = self.get_last_op_second_innermost_last_axis()
            # compute at to last op
            sch[S].compute_at(last_ops[0], axis)
            warp_num = self.get_last_op_warp_numbers()
            fused = sch[S].fuse(*sch[S].op.axis)
            fused, thread_level = sch[S].split(fused, factor=self.warp_size)
            fused, warp_level = sch[S].split(fused, factor=warp_num)
            sch[S].bind(thread_level, self.otx)
            sch[S].bind(warp_level, self.oty)

    def set_scope(self, op_id, op, sch, X):
        if op in self.recipe_stage.operation_role:
            # do not set scope for output op
            if self.recipe_stage.operation_role[op] != OperationRole.output_op:
                # only handle register level
                sch[X(op)].set_scope("local")

    def tiling(self, op_id, op, sch, X):
        # only tiling for 3 ops: main, output, last
        if op == self.main_op:
            # prepare spatial axis
            axis = sch[X(op)].op.axis
            reserve_spatial_num = self.recipe_stage.reserve_inner_axis_count[op]
            spatial_axis_split_parts = [
                axis[:-reserve_spatial_num], axis[-reserve_spatial_num:]]

            all_reduce_axis = sch[X(op)].op.reduce_axis
            reserve_reduce_axis = []
            split_reduce_axis = []
            for i, iv in enumerate(all_reduce_axis):
                if i in self.recipe_stage.main_op_reserve_reduce_axis:
                    reserve_reduce_axis.append(iv)
                else:
                    split_reduce_axis.append(iv)
            pos = self.get_output_op_third_innermost_last_axis()
            sch[X(op)].compute_at(sch[X(self.output_op)], pos)
            
            reduce_axis_split_parts = []
            reduce_axis_split_factors = self.get_main_op_reduce_axis_factors(
                len(split_reduce_axis)
            )
            for iv, factors in zip(split_reduce_axis, reduce_axis_split_factors):
                part = []
                for f in reversed(factors[1:]):
                    iv, inner = sch[X(op)].split(iv, factor=f)
                    part.append(inner)
                part.append(iv)
                part = list(reversed(part))
                reduce_axis_split_parts.append(part)
            reordered_reduce_axis = [list(x) for x in zip(*reduce_axis_split_parts)]
            reordered_reduce_axis.append(reserve_reduce_axis)
            ordered_axis = reordered_reduce_axis[:-1] + \
                           [spatial_axis_split_parts[0]] + \
                           [reordered_reduce_axis[-1]] + \
                           [spatial_axis_split_parts[1]]
            ordered_axis = reduce(lambda x, y: x + y, ordered_axis, [])
            sch[X(op)].reorder(*ordered_axis)
            self.state["main_op_reduce_axis"] = reordered_reduce_axis
        elif op == self.output_op:
            axis = sch[X(op)].op.axis
            reserve_spatial_num = self.recipe_stage.reserve_inner_axis_count[op]
            split_spatial_axis = axis[:-reserve_spatial_num]
            reserve_spatial_axis = axis[-reserve_spatial_num:]
            spatial_axis_split_factors = self.get_output_op_axis_factors(
                len(split_spatial_axis)
            )
            spatial_axis_split_parts = []
            for iv, factors in zip(split_spatial_axis, spatial_axis_split_factors):
                part = []
                for f in reversed(factors[1:]):
                    iv, inner = sch[X(op)].split(iv, factor=f)
                    part.append(inner)
                part.append(iv)
                part = list(reversed(part))
                spatial_axis_split_parts.append(part)
            reordered_spatial_axis = [list(x) for x in zip(*spatial_axis_split_parts)]
            reordered_spatial_axis.append(reserve_spatial_axis)
            # reorder
            ordered_axis = reduce(lambda x, y: x + y, reordered_spatial_axis, [])
            sch[X(op)].reorder(*ordered_axis)
            # fuse and bind
            assert len(reordered_spatial_axis) > 3
            fused_axis = [sch[X(op)].fuse(*part) for part in reordered_spatial_axis[:-2]]
            final_axis = [[x] for x in fused_axis]
            sch[X(op)].bind(fused_axis[0], self.bx)
            sch[X(op)].bind(fused_axis[-1], self.ty)
            # thread level intrinsic, still bind to thread x
            if self.recipe_stage.instruction_scope == InstructionScope.thread:
                fused = sch[X(op)].fuse(*reordered_spatial_axis[-2])
                outer, inner = sch[X(op)].split(fused, nparts=self.warp_size)
                sch[X(op)].bind(outer, self.tx)
                final_axis.append([outer, inner])
                final_axis.append(reordered_spatial_axis[-1])
            else:
                final_axis.append(reordered_spatial_axis[-2])
                final_axis.append(reordered_spatial_axis[-1])
            self.state["output_op_axis"] = final_axis
        elif op == self.target_dag.op_lst[-1]:
            # last op
            axis = sch[X(op)].op.axis
            fused = sch[X(op)].fuse(*axis)
            fused, thread_level = sch[X(op)].split(fused, factor=self.warp_size)
            split_factors = self.get_last_op_axis_factors(1)
            split_parts = []
            for f in reversed(split_factors[0][1:]):
                fused, inner = sch[X(op)].split(fused, factor=f)
                split_parts.append(inner)
            split_parts.append(fused)
            split_parts = list(reversed(split_parts))
            sch[X(op)].bind(split_parts[0], self.obx)
            sch[X(op)].bind(split_parts[-1], self.oty)
            sch[X(op)].bind(thread_level, self.otx)
            self.state["last_op_axis"] = split_parts + [thread_level]

    def compute_at(self, op_id, op, sch, X):
        pass

    def unroll(self, op_id, op, sch, X):
        pass
    
    def schedule(self, sch, mapping_func=lambda x: x):
        X = mapping_func
        
        # initialize parameters
        self.initialize_parameters()
        # check if parameters are ready
        self.check_parameter_ready()
        # initialize state
        self.initialize_state()

        dag = self.target_dag
        for op_id, op in enumerate(reversed(dag.op_lst)):
            if not isinstance(op, tvm.te.ComputeOp):
                continue
            else:
                self.inline(op, op_id, sch, X)
                self.cache_read(op, op_id, sch, X)
                self.set_scope(op, op_id, sch, X)
                self.tiling(op, op_id, sch, X)
                self.compute_at(op, op_id, sch, X)
                self.unroll(op, op_id, sch, X)            


def auto_schedule(intrin_match_result, transform_state):
    """We aim to design special scheduler for tensorize.
       So we only accept a particular structure of input DAG.
       The input DAG only has one intrinsic match point,
       the other nodes in the DAG should not contain reduction.
       For other kind of input DAG, a previous dispatcher should
       cut the DAG into proper sub-DAGs and assign those we don't
       handle to other schedulers such as Ansor, FlexTensor, or AutoTVM.
    """
    # TODO: add a checker to check if we can schedule the given DAG
    target_main_op = None
    for k, v in transform_state.main_op_map.items():
        target_main_op = v
    assert target_main_op is not None
    for op in transform_state.target_dag.op_lst:
        if len(op.reduce_axis) > 0 and op != target_main_op:
            raise RuntimeError(
                "We do not support scheduling for reduce op which is not main op.")
    recipe = intrin_match_result.recipe
    if recipe.target == "cuda":
        pass
    else:
        raise RuntimeError("Target not supported: %s" % recipe.target)


class ScheduleResult(object):
    def __init__(self, schedule, schedule_steps):
        self.schedule = schedule
        self.schedule_steps = schedule_steps
