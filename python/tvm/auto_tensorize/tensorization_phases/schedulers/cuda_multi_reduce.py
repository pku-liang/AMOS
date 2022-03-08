import json
from ...utils import *
from ...target import *
from ...search import CDParamGenerator, Entry, SAEntryGenerator
from ...hw_abs_dag import OperationRole, HwAbsDAGStage, InstructionScope
from ..schedule_base import *
from functools import reduce
from collections import OrderedDict


class CUDAStateMultiReduce(object):
    def __init__(
            self, inlined, main_op_reduce_axis,
            output_op_axis, left_op_spatial_axis_map,
            left_op_reduce_axis_map, tensorize_iter, op_to_id):
        self.inlined = inlined
        self.main_op_reduce_axis = main_op_reduce_axis
        self.output_op_axis = output_op_axis
        self.left_op_spatial_axis_map = left_op_spatial_axis_map
        self.left_op_reduce_axis_map = left_op_reduce_axis_map
        self.tensorize_iter = tensorize_iter
        self.op_to_id = op_to_id


def empty_cuda_state_multi_reduce():
    return CUDAStateMultiReduce(set(), [], [], {}, {}, {}, {})


class CUDAParamsMultiReduce(object):
    def __init__(
            self, vectorize, spatial_factors, reduce_factors,
            left_spatial_factors_map, left_reduce_factors_map,
            output_unroll_step, left_op_unroll_step):
        self.vectorize = vectorize
        self.spatial_factors = spatial_factors
        self.reduce_factors = reduce_factors
        self.left_spatial_factors_map = left_spatial_factors_map
        self.left_reduce_factors_map = left_reduce_factors_map
        self.output_unroll_step = output_unroll_step
        self.left_op_unroll_step = left_op_unroll_step

    def to_json(self):
        ret = {
            "vectorize": self.vectorize,
            "spatial_factors": self.spatial_factors,
            "reduce_factors": self.reduce_factors,
            "left_spatial_factors_map": self.left_spatial_factors_map,
            "left_reduce_factors_map": self.left_reduce_factors_map,
            "output_unroll_step": self.output_unroll_step,
            "left_op_unroll_step": self.left_op_unroll_step
        }
        return ret

    def from_json(self, obj):
        self.vectorize = obj["vectorize"]
        self.spatial_factors = obj["spatial_factors"]
        self.reduce_factors = obj["reduce_factors"]
        self.left_spatial_factors_map = OrderedDict(
            {int(x): y for x, y in obj["left_spatial_factors_map"].items()})
        self.left_reduce_factors_map = OrderedDict(
            {int(x): y for x, y in obj["left_reduce_factors_map"].items()})
        self.output_unroll_step = obj["output_unroll_step"]
        self.left_unroll_step = obj["left_op_unroll_step"]

    def __str__(self):
        obj = self.to_json()
        new_obj = {}

        def handle(v):
            if isinstance(v, dict):
                return {x: handle(y) for x, y in v.items()}
            if isinstance(v, list):
                return [handle(x) for x in v]
            if isinstance(v, tuple) and len(v) == 2:
                return handle(v[0])
            return v

        for k, v in obj.items():
            new_obj[k] = handle(v)
        return json.dumps(new_obj)


def empty_cuda_params_multi_reduce():
    return CUDAParamsMultiReduce(
        None, [], [], OrderedDict(), OrderedDict(), None, None)


#####################################################
# Target specific parameter generator
#####################################################
class CUDAKernelParamGeneratorMultiReduce(CDParamGenerator):
    pass


def need_tiling(op, dag, threshhold=256):
    """
    Returns:
    [tile spatial, tile reduce]
    """
    assert len(dag.op_lst) > 0
    tile_spatial = True
    tile_reduce = True
    reduce_domain = reduce(
        lambda x, y: x * y, [
            int(iv.dom.extent) for iv in op.reduce_axis], 1)
    if can_inline(op, dag):
        tile_reduce = False
        tile_spatial = False
    elif reduce_domain <= threshhold:
        tile_reduce = False
    return (tile_spatial, tile_reduce)


class CUDAScheduleGeneratorMultiReduce(AcceleratorScheduleGenerator):
    def __init__(self, intrin_match_result, transform_state, eps=0.9,
            reduce_tiling=3, spatial_tiling=4,
            left_reduce_tiling=2, left_spatial_tiling=3, arch=70,
            log_file="cuda_schedule_generator.log", steps=1,
            verbose_init=True):
        super(CUDAScheduleGeneratorMultiReduce, self).__init__(eps, CUDAParamsMultiReduce,
            steps=steps, log_file=log_file, verbose_init=verbose_init)
        self.init_hw_abs_dag(intrin_match_result)
        nodes = self.init_target_dag(transform_state)
        self.init_hw_abs_dag_stage(nodes)
        # get main op id and output op id
        self.main_op_id = 0
        self.output_op_id = 0
        for i, op in enumerate(self.target_dag.op_lst):
            if op == self.main_op:
                self.main_op_id = i
            elif op == self.output_op:
                self.output_op_id = i
        # constants
        self.reduce_tiling_parts = reduce_tiling
        self.spatial_tiling_parts = spatial_tiling
        self.left_reduce_tiling_parts = left_reduce_tiling
        self.left_spatial_tiling_parts = left_spatial_tiling
        self.arch_info = CUDA(arch=arch)
        self.warp_size = self.arch_info.get_warp_size()
        # params generator
        self.init_param_generator()
        self.init_score_table()

    def init_hw_abs_dag(self, intrin_match_result):
        hw_abs_dag = intrin_match_result.hw_abs_dag
        compute_key = intrin_match_result.compute_key
        shape_key = intrin_match_result.shape_key
        self.hw_abs_dag = hw_abs_dag
        self.compute_key = compute_key
        self.shape_key = shape_key

    def init_target_dag(self, transform_state):
        # get main op
        target_main_op = None
        for k, v in transform_state.main_op_map.items():
            target_main_op = v
        assert target_main_op is not None
        # insert intrinsic dag
        self.target_dag, info = reconstruct_dag_as_intrin(
            transform_state.target_dag,
            target_main_op,
            self.hw_abs_dag,
            self.compute_key,
            self.shape_key)
        # nodes: dict {hw abs name : new tensor}
        (_, _, nodes, _, _) = info
        # get new main op
        self.main_op = nodes[self.hw_abs_dag.main_hw_abs_name][0].op
        return nodes

    def init_hw_abs_dag_stage(self, nodes):
        ###################################
        # fill the hw_abs_dag stage info
        # analyze the intrinsic dag
        def cond(cur):
            if cur in self.hw_abs_dag.hw_abs_dict:
                return True
            return False
        hw_abs_names, read_graph, feed_graph = self.hw_abs_dag.serialize_dag(
            cond1=cond)

        operation_role = {}
        hw_abs_map = {}
        reserve_inner_axis_count = {}
        main_op_reserve_reduce_axis = []
        main_op_reserve_reduce_axis_factor = []

        load_from_shared = {}
        store_to_shared = {}
        self.output_op = None
        for name in hw_abs_names:
            op = nodes[name][0].op
            hw_abs_map[op] = name
            spatial_axis, reduce_axis = \
                self.hw_abs_dag.get_hw_abs_compute_reserve_axis(
                    self.compute_key, self.shape_key, name)
            reserve_inner_axis_count[op] = len(spatial_axis)
            if name not in read_graph:
                operation_role[op] = OperationRole.load_op
                load_from_shared[op] = 1
            elif name not in feed_graph:
                operation_role[op] = OperationRole.output_op
                store_to_shared[op] = 0
                self.output_op = op
            elif name == self.hw_abs_dag.main_hw_abs_name:
                operation_role[op] = OperationRole.main_op
                for i, red in enumerate(reduce_axis):
                    main_op_reserve_reduce_axis.append(
                        len(op.reduce_axis) - len(reduce_axis) + i)
                    main_op_reserve_reduce_axis_factor.append(
                        int(red.dom.extent))
        assert self.output_op is not None
        # construct hw_abs_dag stage
        self.hw_abs_dag_stage = HwAbsDAGStage(
            operation_role,
            self.hw_abs_dag.target,
            self.hw_abs_dag.get_name(),
            self.compute_key,
            self.shape_key,
            hw_abs_map,
            reserve_inner_axis_count,
            main_op_reserve_reduce_axis,
            main_op_reserve_reduce_axis_factor,
            load_from_shared,
            store_to_shared,
            self.hw_abs_dag.scope
        )

    def init_param_generator(self):
        # for main op reduce split
        self.reduce_splits = []
        reserve_reduce_axis = set()
        for a in self.hw_abs_dag_stage.main_op_reserve_reduce_axis:
            reserve_reduce_axis.add(int(a))
        # skip the reserved reduce axis
        for i, iv in enumerate(self.main_op.reduce_axis):
            if i not in reserve_reduce_axis:
                gen = SplitFactorGenerator(
                    int(iv.dom.extent), self.reduce_tiling_parts)
                self.reduce_splits.append(gen)
        # for output op spatial split
        self.spatial_splits = []
        reserve_axis_count = int(self.hw_abs_dag_stage.reserve_inner_axis_count[self.output_op])
        # skip the reserved spatial axis
        for iv in self.output_op.axis[:-reserve_axis_count]:
            gen = SplitFactorGenerator(
                int(iv.dom.extent), self.spatial_tiling_parts)
            self.spatial_splits.append(gen)
        # for left op reduce/spatial axis
        self.left_reduce_splits = OrderedDict()
        self.left_spatial_splits = OrderedDict()
        self.unroll_left = OrderedDict()
        for op_id, op in enumerate(self.target_dag.op_lst):
            if op in self.hw_abs_dag_stage.operation_role:
                # skip tensorize related ops
                continue
            (tile_spatial, tile_reduce) = need_tiling(op, self.target_dag)
            if tile_spatial:
                total_spatial_extents = reduce(
                    lambda x, y: x * y, [
                        int(iv.dom.extent) for iv in op.axis], 1)
                total_spatial_extents = (
                    total_spatial_extents
                    + self.warp_size - 1) // self.warp_size
                sp_gen = SplitFactorGenerator(
                    total_spatial_extents, self.left_spatial_tiling_parts)
                self.left_spatial_splits[op_id] = sp_gen
            if tile_reduce:
                total_reduce_extents = reduce(
                    lambda x, y: x * y, [
                        int(iv.dom.extent) for iv in op.reduce_axis], 1)
                total_reduce_extents = (
                    total_reduce_extents
                    + self.warp_size - 1) // self.warp_size
                rd_gen = SplitFactorGenerator(
                    total_reduce_extents, self.left_reduce_tiling_parts)
                self.left_reduce_splits[op_id] = rd_gen

        self.unroll_left = UnrollStepGenerator([16, 64, 512, 1500])
        self.vectorize = VectorizeLengthGenerator(
            self.hw_abs_dag.target, self.main_op.input_tensors[0].dtype)
        self.unroll_output = UnrollStepGenerator([16, 64, 512, 1500])
        self.generator_lst = [
            self.vectorize,
            *self.spatial_splits,
            *self.reduce_splits,
            *list(self.left_spatial_splits.values()),
            *list(self.left_reduce_splits.values()),
            self.unroll_output,
            self.unroll_left
        ]

    def init_score_table(self):
        self.score_table = softmax([0.5 for gen in self.generator_lst])

    def get_generators(self):
        return self.generator_lst

    def get_schedule_compute_info(self):
        return ScheduleComputeInfo(
            self.target_dag,
            self.main_op,
            self.output_op,
            self.main_op_id,
            self.output_op_id,
            self.hw_abs_dag_stage,
            spatial_tiling = self.spatial_tiling_parts,
            reduce_tiling = self.reduce_tiling_parts,
            left_spatial_tiling = self.left_spatial_tiling_parts,
            left_reduce_tiling = self.left_reduce_tiling_parts
        )

    def valid(self, record):
        max_warps = self.arch_info.max_threads() // self.warp_size
        max_blocks = self.arch_info.max_blocks()
        warp_num = 1
        block_num = 1
        for factors in record.spatial_factors:
            warp_num *= factors[0][-2]
            block_num *= factors[0][0]
        if warp_num > max_warps:
            return False
        if block_num > max_blocks:
            return False
        for factors in record.left_spatial_factors_map.values():
            warp_num = factors[0][-2]
            if warp_num > max_warps:
                return False
            block_num = factors[0][0]
            if block_num > max_blocks:
                return False
        return True

    def record_from_json(self, obj):
        ret = empty_cuda_params_multi_reduce()
        ret.from_json(obj)
        return ret

    def get_record(self, entry=None, policy="random"):
        if entry is None:
            record = self.record_cls(
                self.vectorize.get(policy=policy),
                [gen.get(policy=policy) for gen in self.spatial_splits],
                [gen.get(policy=policy) for gen in self.reduce_splits],
                OrderedDict(
                    {x: gen.get(
                        policy=policy) for x, gen
                        in self.left_spatial_splits.items()}),
                OrderedDict(
                    {x: gen.get(
                        policy=policy) for x, gen
                        in self.left_reduce_splits.items()}),
                self.unroll_output.get(policy=policy),
                self.unroll_left.get(policy=policy))
        else:
            record = self.record_cls(
                self.vectorize.get(hint=entry.record.vectorize[0], policy="q"),
                [gen.get(hint=x[0], policy="q") for gen, x in zip(
                    self.spatial_splits, entry.record.spatial_factors)],
                [gen.get(hint=x[0], policy="q") for gen, x in zip(
                    self.reduce_splits, entry.record.reduce_factors)],
                OrderedDict(
                    {y: gen.get(
                        hint=x[0], policy="q") for (y, gen), x in zip(
                            self.left_spatial_splits.items(),
                            entry.record.left_spatial_factors_map)}),
                OrderedDict(
                    {y: gen.get(
                        hint=x[0], policy="q") for (y, gen), x in zip(
                            self.left_reduce_splits.items(),
                            entry.record.left_reduce_factors_map)}),
                self.unroll_output.get(
                    hint=entry.record.output_unroll_step[0], policy="q"),
                self.unroll_left.get(
                    hint=entry.record.left_op_unroll_step[0], policy="q")
                )
        return record

    def get_records_mutate_one_generator(
            self, record, to_mutate, steps):
        vec = record.vectorize
        spatial = record.spatial_factors
        reduce = record.reduce_factors
        left_spatial = record.left_spatial_factors_map.values()
        left_reduce = record.left_reduce_factors_map.values()
        unroll_output = record.output_unroll_step
        unroll_left = record.left_op_unroll_step

        next_vec = self.vectorize.get_next(
            vec[0], to_mutate)
        next_spatial = [
            gen.get_next(x[0], to_mutate) for gen, x in zip(
                self.spatial_splits, spatial)
        ]
        next_reduce = [
            gen.get_next(x[0], to_mutate) for gen, x in zip(
                self.reduce_splits, reduce)
        ]
        next_left_spatial = [
            gen.get_next(x[0], to_mutate) for gen, x in zip(
                self.left_spatial_splits.values(), left_spatial)
        ]
        next_left_reduce = [
            gen.get_next(x[0], to_mutate) for gen, x in zip(
                self.left_reduce_splits.values(), left_reduce)
        ]
        next_unroll_output = self.unroll_output.get_next(
            unroll_output[0], to_mutate
        )
        next_unroll_left = self.unroll_left.get_next(
            unroll_left[0], to_mutate
        )
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
            vec = helper(next_vec, vec)
            spatial = [
                helper(_gen, org_val) for _gen, org_val in zip(
                    next_spatial, spatial)
            ]
            reduce = [
                helper(_gen, org_val) for _gen, org_val in zip(
                    next_reduce, reduce)
            ]
            left_spatial = [
                helper(_gen, org_val) for _gen, org_val in zip(
                    next_left_spatial, left_spatial)
            ]
            left_reduce = [
                helper(_gen, org_val) for _gen, org_val in zip(
                    next_left_reduce, left_reduce)
            ]
            unroll_output = helper(next_unroll_output, unroll_output)
            unroll_left = helper(next_unroll_left, unroll_left)
            if has_mutate:
                yield self.record_cls(
                    vec,
                    spatial,
                    reduce,
                    OrderedDict({
                        x: y for x, y in zip(
                            self.left_spatial_splits.keys(), left_spatial)}),
                    OrderedDict({
                        x: y for x, y in zip(
                            self.left_reduce_splits.keys(), left_reduce)}),
                    unroll_output,
                    unroll_left)
            has_mutate = False

    def feedback_value(self, entry, value):
        self.vectorize.feedback(*entry.record.vectorize, value)
        for gen, factors in zip(
                self.spatial_splits, entry.record.spatial_factors):
            gen.feedback(*factors, value)
        for gen, factors in zip(
                self.reduce_splits, entry.record.reduce_factors):
            gen.feedback(*factors, value)
        for gen, factors in zip(
                self.left_spatial_splits.values(),
                entry.record.left_spatial_factors_map):
            gen.feedback(*factors, value)
        for gen, factors in zip(
                self.left_reduce_splits.values(),
                entry.record.left_reduce_factors_map):
            gen.feedback(*factors, value)
        self.unroll_output.feedback(*entry.record.output_unroll_step, value)
        self.unroll_left.feedback(*entry.record.left_op_unroll_step, value)


class CUDAScheduleApplierMultiReduce(object):
    def __init__(self, intrin_match_result, schedule_compute_info, arch=70):
        self.intrin_match_result = intrin_match_result
        # get match hw_abs_dag info
        hw_abs_dag = intrin_match_result.hw_abs_dag
        compute_key = intrin_match_result.compute_key
        shape_key = intrin_match_result.shape_key
        self.hw_abs_dag = hw_abs_dag
        self.compute_key = compute_key
        self.shape_key = shape_key

        self.target_dag = schedule_compute_info.target_dag
        self.main_op = schedule_compute_info.main_op
        self.output_op = schedule_compute_info.output_op
        self.main_op_id = schedule_compute_info.main_op_id
        self.output_op_id = schedule_compute_info.output_op_id
        self.hw_abs_dag_stage = schedule_compute_info.hw_abs_dag_stage
        # the state during schedule
        self.state = empty_cuda_state_multi_reduce()
        # the parameters during schedule
        self.params = empty_cuda_params_multi_reduce()
        # some constants
        self.warp_size = CUDA(arch=arch).get_warp_size()
        self.bx = tvm.te.thread_axis("blockIdx.x")
        self.ty = tvm.te.thread_axis("threadIdx.y")
        self.tx = tvm.te.thread_axis("threadIdx.x")
        self.get_vx = lambda _: tvm.te.thread_axis("vthread")
        self.get_bx = lambda _: tvm.te.thread_axis("blockIdx.x")
        self.get_ty = lambda _: tvm.te.thread_axis("threadIdx.y")
        self.get_tx = lambda _: tvm.te.thread_axis("threadIdx.x")
        # self.reduce_tiling_parts = schedule_compute_info.kwargs["reduce_tiling"]
        # self.spatial_tiling_parts = schedule_compute_info.kwargs["spatial_tiling"]

    def initialize_state(self):
        # self.state = {
        #     "inlined": set(),
        #     "main_op_reduce_axis": [],
        #     "output_op_axis": [],
        #     "last_op_axis": [],
        #     "tensorize_iter": {}
        # }
        self.state = empty_cuda_state_multi_reduce()

    def initialize_parameters(self, params):
        self.params = params

    def get_main_op_outermost_last_reduce_axis(self):
        if len(self.state.main_op_reduce_axis) > 0:
            assert isinstance(self.state.main_op_reduce_axis[0], list)
            assert len(self.state.main_op_reduce_axis[0]) > 0
            return self.state.main_op_reduce_axis[0][-1]
        else:
            # no reduce axis
            # print(self.state.main_op_reduce_axis)
            # print(self.main_op.body)
            raise RuntimeError("No reduce axis in main op.")

    def get_main_op_outermost_first_reduce_axis(self):
        if len(self.state.main_op_reduce_axis) > 0:
            assert isinstance(self.state.main_op_reduce_axis[0], list)
            assert len(self.state.main_op_reduce_axis[0]) > 0
            return self.state.main_op_reduce_axis[0][0]
        else:
            # no reduce axis
            # print(self.state.main_op_reduce_axis)
            # print(self.main_op.body)
            raise RuntimeError("No reduce axis in main op.")

    def get_main_op_second_outermost_last_reduce_axis(self):
        if len(self.state.main_op_reduce_axis) > 1:
            assert isinstance(self.state.main_op_reduce_axis[1], list)
            assert len(self.state.main_op_reduce_axis[1]) > 0
            return self.state.main_op_reduce_axis[1][-1]
        else:
            # no enough reduce axis
            return self.get_main_op_outermost_last_reduce_axis()

    def get_output_op_third_innermost_last_axis(self):
        assert len(self.state.output_op_axis) > 2
        assert isinstance(self.state.output_op_axis[-3], list)
        assert len(self.state.output_op_axis[-3]) > 0
        return self.state.output_op_axis[-3][-1]

    def get_output_op_outermost_last_axis(self):
        assert len(self.state.output_op_axis) > 1
        assert isinstance(self.state.output_op_axis[0], list)
        assert len(self.state.output_op_axis[0]) > 0
        return self.state.output_op_axis[0][-1]

    # def get_last_op_innermost_last_axis(self):
    #     assert len(self.state.last_op_axis) > 0
    #     assert isinstance(self.state.last_op_axis[-1], list)
    #     assert len(self.state.last_op_axis[-1]) > 0
    #     return self.state.last_op_axis[-1][-1]

    # def get_last_op_second_innermost_last_axis(self):
    #     assert len(self.state.last_op_axis) > 0
    #     assert isinstance(self.state.last_op_axis[-1], list)
    #     assert len(self.state.last_op_axis[-1]) > 1
    #     return self.state.last_op_axis[-1][-2]

    def get_left_op_second_innermost_axis(self, op):
        assert op in self.state.op_to_id
        op_id = self.state.op_to_id[op]
        assert op_id in self.state.left_op_spatial_axis_map
        op_axis = self.state.left_op_spatial_axis_map[op_id]
        assert len(op_axis) > 1
        return op_axis[-2]

    def get_left_op_outermost_reduce_axis(self, op):
        assert op in self.state.op_to_id
        op_id = self.state.op_to_id[op]
        assert op_id in self.state.left_op_reduce_axis_map
        op_reduce_axis = self.state.left_op_reduce_axis_map[op_id]
        assert len(op_reduce_axis) > 0
        return op_reduce_axis[0]

    # def get_last_op_outermost_last_axis(self):
    #     assert len(self.state.last_op_axis) > 0
    #     assert isinstance(
    #         self.state.last_op_axis[0], list), self.state.last_op_axis[0]
    #     assert len(self.state.last_op_axis[0]) > 0
    #     return self.state.last_op_axis[0][0]

    def get_left_op_outermost_axis(self, op):
        assert op in self.state.op_to_id
        op_id = self.state.op_to_id[op]
        assert op_id in self.state.left_op_spatial_axis_map
        op_spatial_axis = self.state.left_op_spatial_axis_map[op_id]
        assert len(op_spatial_axis) > 0
        return op_spatial_axis[0]

    def get_tensorize_iter(self, op):
        assert op in self.state.tensorize_iter
        return self.state.tensorize_iter[op]

    def get_main_op_warp_numbers(self):
        assert len(self.params.spatial_factors) > 0
        ret = 1
        for part in self.params.spatial_factors:
            assert len(part[0]) > 1
            ret *= part[0][-2]
        return ret

    def get_main_op_reduce_axis_factors(self, number):
        assert len(self.params.reduce_factors) >= number
        return [x[0] for x in self.params.reduce_factors[:number]]

    def get_output_op_axis_factors(self, number):
        assert len(self.params.spatial_factors) >= number, (
            len(self.params.spatial_factors), " vs. ", number)
        return [x[0] for x in self.params.spatial_factors[:number]]

    # def get_last_op_axis_factors(self, number):
    #     assert len(self.params.last_factors) >= number
    #     return [x[0] for x in self.params.last_factors[:number]]

    # def get_last_op_warp_numbers(self):
    #     assert len(self.params.last_factors) > 0
    #     ret = 1
    #     for part in self.params.last_factors:
    #         assert len(part) > 0
    #         ret *= part[0][-1]
    #     return ret

    def get_left_op_spatial_factors(self, op):
        assert op in self.state.op_to_id
        op_id = self.state.op_to_id[op]
        assert op_id in self.params.left_spatial_factors_map
        factors = self.params.left_spatial_factors_map[op_id][0]
        return factors

    def get_left_op_reduce_factors(self, op):
        assert op in self.state.op_to_id
        op_id = self.state.op_to_id[op]
        assert op_id in self.params.left_reduce_factors_map
        factors = self.params.left_reduce_factors_map[op_id][0]
        return factors

    def get_left_op_num_warps(self, op):
        assert op in self.state.op_to_id
        op_id = self.state.op_to_id[op]
        assert op_id in self.params.left_spatial_factors_map
        factors = self.params.left_spatial_factors_map[op_id][0]
        assert len(factors) > 0
        return factors[-2]

    def get_vectorize_length(self):
        assert self.params.vectorize is not None
        return self.params.vectorize[0]

    def get_output_op_unroll_step(self):
        assert self.params.output_unroll_step is not None
        return self.params.output_unroll_step[0]

    # def get_last_op_unroll_step(self):
    #     assert self.params.last_unroll_step is not None
    #     return self.params.last_unroll_step[0]

    def get_left_op_unroll_step(self):
        # assert op in self.state.op_to_id
        # op_id = self.state.op_to_id[op]
        # assert op_id in self.params.left_op_unroll_step
        unroll_step = self.params.left_op_unroll_step[0]
        return unroll_step

    def check_parameter_ready(self):
        return True

    def inline(self, op_id, op, sch, X):
        if op in self.hw_abs_dag_stage.operation_role:
            return
        else:
            if can_inline(op, self.target_dag):
                sch[X(op)].compute_inline()
                self.state.inlined.add(op)

    def cache_read(self, op_id, op, sch, X):
        if op in self.hw_abs_dag_stage.operation_role:
            return
        # if op in self.state.inlined:
        #     return
        if not op in self.target_dag.feed_graph:
            return
        do_cache_read_for_load = False
        do_cache_read_for_other = False
        consumers = self.target_dag.feed_graph[op]
        if len(consumers) <= 0:
            return
        if consumers[0] in self.hw_abs_dag_stage.operation_role:
            if self.hw_abs_dag_stage.operation_role[consumers[0]] == OperationRole.load_op:
                if len(consumers) == 1:
                    do_cache_read_for_load = True
        elif len(consumers[0].reduce_axis) > 0:
            # other op
            if len(consumers) == 1:
                do_cache_read_for_other = True

        # can't do both
        assert not (do_cache_read_for_load and do_cache_read_for_other)

        if do_cache_read_for_load:
            S = sch.cache_read(
                X(op).output(0), "shared", [X(x) for x in consumers])
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

        # if do_cache_read_for_other:
        #     tile_spatial, tile_reduce = need_tiling(
        #         consumers[0], self.target_dag)
        #     other_ops = [X(x) for x in consumers]
        #     # only use local buffer for these small reduce ops
        #     S = sch.cache_read(X(op).output(0), "local", other_ops)
        #     # use the original op as index for op id
        #     if not tile_reduce:
        #         axis = self.get_left_op_second_innermost_axis(consumers[0])
        #     else:
        #         axis = self.get_left_op_outermost_reduce_axis(consumers[0])
        #     # compute at to last op
        #     sch[S].compute_at(sch[other_ops[0]], axis)
        #     warp_num = self.get_left_op_num_warps(consumers[0])
        #     fused = sch[S].fuse(*sch[S].op.axis)
        #     fused, warp_level = sch[S].split(
        #         fused, factor=warp_num*self.warp_size)
        #     sch[S].bind(warp_level, self.get_tx(0))

    def set_scope(self, op_id, op, sch, X):
        if op in self.hw_abs_dag_stage.operation_role:
            # do not set scope for output op
            if self.hw_abs_dag_stage.operation_role[op] != OperationRole.output_op:
                # only handle register level
                sch[X(op)].set_scope("local")

    def tiling(self, op_id, op, sch, X):
        # only tiling for 3 kindes of ops: main, output, left
        if op == self.main_op:
            # prepare spatial axis
            axis = sch[X(op)].op.axis
            reserve_spatial_num = int(self.hw_abs_dag_stage.reserve_inner_axis_count[op])
            spatial_axis_split_parts = [
                axis[:-reserve_spatial_num], axis[-reserve_spatial_num:]]

            all_reduce_axis = sch[X(op)].op.reduce_axis
            reserve_reduce_axis = []
            split_reduce_axis = []
            tmp = set([int(x) for x in self.hw_abs_dag_stage.main_op_reserve_reduce_axis])
            for i, iv in enumerate(all_reduce_axis):
                if i in tmp:
                    reserve_reduce_axis.append(iv)
                else:
                    split_reduce_axis.append(iv)
            reserve_reduce_num = len(reserve_reduce_axis)
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
            assert len(reordered_reduce_axis) > 3, "No enough reduce axis split."
            ordered_axis = reordered_reduce_axis[:-2] + \
                           [spatial_axis_split_parts[0]] + \
                           reordered_reduce_axis[-2:-1] + \
                           [spatial_axis_split_parts[1]] + \
                           reordered_reduce_axis[-1:]
            ordered_axis = reduce(lambda x, y: x + y, ordered_axis, [])
            sch[X(op)].reorder(*ordered_axis)
            self.state.main_op_reduce_axis = reordered_reduce_axis
            self.state.tensorize_iter[op] = ordered_axis[
                -(reserve_spatial_num + reserve_reduce_num)]
        elif op == self.output_op:
            axis = sch[X(op)].op.axis
            reserve_spatial_num = int(self.hw_abs_dag_stage.reserve_inner_axis_count[op])
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
            assert len(reordered_spatial_axis) > 3, "No enough spatial axis split."
            fused_axis = [sch[X(op)].fuse(*part) for part in reordered_spatial_axis[:-2]]
            final_axis = [[x] for x in fused_axis]
            sch[X(op)].bind(fused_axis[0], self.bx)
            # the intermediate bind to vthread
            # for med_fused in fused_axis[1:-1]:
            #     sch[X(op)].bind(med_fused, tvm.te.thread_axis("vthread"))
            sch[X(op)].bind(fused_axis[-1], self.ty)
            # thread level intrinsic, still bind to thread x
            if self.hw_abs_dag_stage.instruction_scope == InstructionScope.thread:
                fused = sch[X(op)].fuse(*reordered_spatial_axis[-2])
                outer, inner = sch[X(op)].split(fused, nparts=self.warp_size)
                sch[X(op)].bind(outer, self.tx)
                final_axis.append([outer, inner])
                final_axis.append(reordered_spatial_axis[-1])
            else:
                final_axis.append(reordered_spatial_axis[-2])
                final_axis.append(reordered_spatial_axis[-1])
            self.state.output_op_axis = final_axis
            self.state.tensorize_iter[op] = final_axis[-1][-2]
        elif op not in self.hw_abs_dag_stage.operation_role:
            # other op
            tile_spatial, tile_reduce = need_tiling(op, self.target_dag)
            if tile_spatial:
                axis = sch[X(op)].op.axis
                fused = sch[X(op)].fuse(*axis)
                split_factors = self.get_left_op_spatial_factors(op)
                split_factors[-2] *= self.warp_size
                split_parts = []
                for f in reversed(split_factors[1:]):
                    fused, inner = sch[X(op)].split(fused, factor=f)
                    split_parts.append(inner)
                split_parts.append(fused)
                split_parts = list(reversed(split_parts))
                sch[X(op)].bind(split_parts[0], self.get_bx(0))
                sch[X(op)].bind(split_parts[-2], self.get_tx(0))
                self.state.left_op_spatial_axis_map.update({
                    op_id: split_parts})
                spatial_axis = split_parts
            else:
                spatial_axis = list(sch[X(op)].op.axis)
            if tile_reduce:
                axis = sch[X(op)].op.reduce_axis
                fused = sch[X(op)].fuse(*axis)
                split_factors = self.get_left_op_reduce_factors(op)
                split_parts = []
                for f in reversed(split_factors[1:]):
                    fused, inner = sch[X(op)].split(fused, factor=f)
                    split_parts.append(inner)
                split_parts.append(fused)
                split_parts = list(reversed(split_parts))
                self.state.left_op_reduce_axis_map.update({
                    op_id: split_parts})
                reduce_axis = split_parts
            else:
                reduce_axis = list(sch[X(op)].op.reduce_axis)
            # interleave
            ordered_axis = (spatial_axis[:2] + reduce_axis[:-1] +
                            spatial_axis[2:] + reduce_axis[-1:])
            sch[X(op)].reorder(*ordered_axis)

    def compute_at(self, op_id, op, sch, X):
        if not op in self.hw_abs_dag_stage.operation_role:
            return
        if op_id < self.main_op_id:
            # compute at to main op
            axis = self.get_main_op_second_outermost_last_reduce_axis()
            sch[X(op)].compute_at(sch[X(self.main_op)], axis)
            reserve_spatial_num = int(self.hw_abs_dag_stage.reserve_inner_axis_count[op])
            self.state.tensorize_iter[op] = sch[X(op)].op.axis[-reserve_spatial_num]
        elif self.main_op_id < op_id < self.output_op_id:
            # compute at to output op
            axis = self.get_output_op_third_innermost_last_axis()
            sch[X(op)].compute_at(sch[X(self.output_op)], axis)
            reserve_spatial_num = int(self.hw_abs_dag_stage.reserve_inner_axis_count[op])
            self.state.tensorize_iter[op] = sch[X(op)].op.axis[-reserve_spatial_num]

    def unroll(self, op_id, op, sch, X):
        if op == self.output_op:
            axis = self.get_output_op_outermost_last_axis()
            step = self.get_output_op_unroll_step()
            sch[X(op)].pragma(axis, "auto_unroll_max_step", step)
            # sch[X(op)].pragma(axis, "unroll_explicit", 0)
        elif op == self.main_op:
            axis = self.get_main_op_outermost_first_reduce_axis()
            # reuse output op unroll step
            step = self.get_output_op_unroll_step()
            sch[X(op)].pragma(axis, "auto_unroll_max_step", step)
            # sch[X(op)].pragma(axis, "unroll_explicit", 0)
        elif op not in self.hw_abs_dag_stage.operation_role:
            # other op
            tile_spatial, tile_reduce = need_tiling(op, self.target_dag)
            if tile_reduce:
                axis = self.get_left_op_outermost_axis(op)
                step = self.get_left_op_unroll_step()
                sch[X(op)].pragma(axis, "auto_unroll_max_step", step)
                # sch[X(op)].pragma(axis, "unroll_explicit", 0)
            else:
                for axis in sch[X(op)].op.reduce_axis:
                    sch[X(op)].unroll(axis)

    def tensorize(self, op_id, op, sch, X):
        if not op in self.hw_abs_dag_stage.operation_role:
            return
        intrin = self.hw_abs_dag.get_intrinsic(
            self.compute_key, self.shape_key, self.hw_abs_dag_stage.hw_abs_key[op])
        axis = self.get_tensorize_iter(op)
        sch[X(op)].tensorize(axis, intrin)

    def apply(self, sch, params, mapping_func=lambda x: x):
        X = mapping_func
        primitives = [
            self.inline,
            self.cache_read,
            self.set_scope,
            self.tiling,
            self.compute_at,
            self.unroll,
            self.tensorize
        ]

        # initialize parameters
        self.initialize_parameters(params)
        # check if parameters are ready
        self.check_parameter_ready()
        # initialize state
        self.initialize_state()

        dag = self.target_dag
        total_op = len(dag.op_lst)
        for op_id, op in enumerate(reversed(dag.op_lst)):
            self.state.op_to_id.update({op: total_op - op_id - 1})
            if not isinstance(op, tvm.te.ComputeOp):
                continue
            else:
                for prim in primitives:
                    prim(total_op - op_id - 1, op, sch, X)
        return sch
