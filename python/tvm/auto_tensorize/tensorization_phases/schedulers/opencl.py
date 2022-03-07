import json
from ...utils import *
from ...target import *
from ...search import CDParamGenerator, Entry, SAEntryGenerator
from ..compute_transform import TransformState
from ...hw_abstraction.hw_abs_base import ComputeDAG
from ...hw_abs_dag import OperationRole, HwAbsDAGStage, InstructionScope
from ..schedule_base import *


class MaliState(object):
    def __init__(
            self, inlined, main_op_reduce_axis,
            output_op_axis, tensorize_iter):
        self.inlined_ops = inlined
        self.main_op_reduce_axis_parts = main_op_reduce_axis
        self.output_op_axis_parts = output_op_axis
        self.tensorized_axis = tensorize_iter


def empty_mali_state():
    return MaliState(set(), [], [], {})


class MaliParams(object):
    def __init__(
            self, vectorize, spatial_factors, reduce_factors,
            output_unroll_step):
        self.vectorize = vectorize
        self.spatial_factors = spatial_factors
        self.reduce_factors = reduce_factors
        self.output_unroll_step = output_unroll_step

    def to_json(self):
        ret = {
            "vectorize": self.vectorize,
            "spatial_factors": self.spatial_factors,
            "reduce_factors": self.reduce_factors,
            "output_unroll_step": self.output_unroll_step,
        }
        return ret

    def from_json(self, obj):
        self.vectorize = obj["vectorize"]
        self.spatial_factors = obj["spatial_factors"]
        self.reduce_factors = obj["reduce_factors"]
        self.output_unroll_step = obj["output_unroll_step"]

    def __str__(self):
        obj = self.to_json()
        new_obj = {}

        def handle(v):
            if isinstance(v, list):
                return [handle(x) for x in v]
            if isinstance(v, tuple) and len(v) == 2:
                return handle(v[0])
            return v

        for k, v in obj.items():
            new_obj[k] = handle(v)
        return json.dumps(new_obj)


def empty_mali_params():
    return MaliParams(None, [], [], None)


class MaliScheduleGenerator(AcceleratorScheduleGenerator):
    def __init__(self, intrin_match_result,
                 transform_state: TransformState,
                 eps=0.9, reduce_tiling=3, spatial_tiling=4, last_tiling=3,
                 arch="g76", log_file="mali_schedule_generator.log", steps=1):
        super(MaliScheduleGenerator, self).__init__(eps, MaliParams,
                                                    steps=steps,
                                                    log_file=log_file)
        self._init_hw_abs_dag(intrin_match_result)
        nodes = self._init_target_dag(transform_state)
        self._init_hw_abs_dag_stage(nodes)
        # get main op id and output op id
        self.main_op_id = -1
        self.output_op_id = -1
        for i, op in enumerate(self.target_dag.op_lst):
            if op == self.main_op:
                self.main_op_id = i
            elif op == self.output_op:
                self.output_op_id = i
        # constants
        self.n_reduce_tiling_parts = reduce_tiling
        self.n_spatial_tiling_parts = spatial_tiling
        self.n_last_op_tiling_parts = last_tiling
        self.arch_info = Mali(arch=arch)
        self.warp_size = self.arch_info.get_warp_size()
        # params generator
        self.init_param_generator()
        self.init_score_table()

    def _init_hw_abs_dag(self, intrin_match_result):
        hw_abs_dag = intrin_match_result.hw_abs_dag
        compute_key = intrin_match_result.compute_key
        shape_key = intrin_match_result.shape_key
        self.hw_abs_dag = hw_abs_dag
        self.compute_key = compute_key
        self.shape_key = shape_key

    def _init_target_dag(self, transform_state):
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
        # nodes: dict {hw_abs name : new tensor}
        (_, _, nodes, _, _) = info
        # get new main op
        self.main_op = nodes[self.hw_abs_dag.main_hw_abs_name][0].op
        return nodes

    def _init_hw_abs_dag_stage(self, nodes):
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
            filtered_axis = []
            for axis in spatial_axis:
                if int(axis.dom.extent) > 1:
                    filtered_axis.append(axis)
            reserve_inner_axis_count[op] = len(filtered_axis)
            if name == self.hw_abs_dag.main_hw_abs_name:
                operation_role[op] = OperationRole.main_op
                for i, red in enumerate(reduce_axis):
                    main_op_reserve_reduce_axis.append(
                        len(op.reduce_axis) - len(reduce_axis) + i)
                    main_op_reserve_reduce_axis_factor.append(
                        int(red.dom.extent))
            elif name not in read_graph:
                operation_role[op] = OperationRole.load_op
                load_from_shared[op] = 1
            elif name not in feed_graph:
                operation_role[op] = OperationRole.output_op
                store_to_shared[op] = 0
                self.output_op = op
        assert self.output_op is None, "Why OpenCL needs output op"
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
        self.reduce_splits = []
        reserve_reduce_axis = set()
        for a in self.hw_abs_dag_stage.main_op_reserve_reduce_axis:
            reserve_reduce_axis.add(int(a))
        for i, iv in enumerate(self.main_op.reduce_axis):
            if i not in reserve_reduce_axis:
                gen = SplitFactorGenerator(
                    int(iv.dom.extent), self.n_reduce_tiling_parts)
                self.reduce_splits.append(gen)
        self.spatial_splits = []
        for iv in self.main_op.axis:
            gen = SplitFactorGenerator(
                int(iv.dom.extent), self.n_spatial_tiling_parts)
            self.spatial_splits.append(gen)
        self.vectorize = VectorizeLengthGenerator(
            self.hw_abs_dag.target, self.main_op.input_tensors[0].dtype)
        self.unroll_output = UnrollStepGenerator([16, 64, 512, 1500])
        self.generator_lst = [
            self.vectorize,
            *self.spatial_splits,
            *self.reduce_splits,
            self.unroll_output,
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
            spatial_tiling=self.n_spatial_tiling_parts,
            reduce_tiling=self.n_reduce_tiling_parts,
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
        return True

    def record_from_json(self, obj):
        return self.record_cls(
            obj["vectorize"],
            obj["spatial_factors"],
            obj["reduce_factors"],
            obj["output_unroll_step"])

    def get_record(self, entry=None, policy="random"):
        if entry is None:
            record = self.record_cls(
                self.vectorize.get(policy=policy),
                [gen.get(policy=policy) for gen in self.spatial_splits],
                [gen.get(policy=policy) for gen in self.reduce_splits],
                self.unroll_output.get(policy=policy))
        else:
            record = self.record_cls(
                self.vectorize.get(hint=entry.record.vectorize[0], policy="q"),
                [gen.get(hint=x[0], policy="q") for gen, x in zip(
                    self.spatial_splits, entry.record.spatial_factors)],
                [gen.get(hint=x[0], policy="q") for gen, x in zip(
                    self.reduce_splits, entry.record.reduce_factors)],
                self.unroll_output.get(
                    hint=entry.record.output_unroll_step[0], policy="q"),
            )
        return record

    def get_records_mutate_one_generator(
            self, record, to_mutate, steps):
        vec = record.vectorize
        spatial = record.spatial_factors
        reduce = record.reduce_factors
        unroll_output = record.output_unroll_step

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
        next_unroll_output = self.unroll_output.get_next(
            unroll_output[0], to_mutate
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
                helper(_gen, org_val) for _gen, org_val in
                zip(next_spatial, spatial)
            ]
            reduce = [
                helper(_gen, org_val) for _gen, org_val in
                zip(next_reduce, reduce)
            ]
            unroll_output = helper(next_unroll_output, unroll_output)
            if has_mutate:
                yield self.record_cls(
                    vec,
                    spatial,
                    reduce,
                    unroll_output)
            has_mutate = False

    def feedback_value(self, entry, value):
        self.vectorize.feedback(*entry.record.vectorize, value)
        for gen, factors in zip(self.spatial_splits,
                                entry.record.spatial_factors):
            gen.feedback(*factors, value)
        for gen, factors in zip(self.reduce_splits,
                                entry.record.reduce_factors):
            gen.feedback(*factors, value)
        self.unroll_output.feedback(*entry.record.output_unroll_step, value)


class MaliScheduleApplier(object):
    def __init__(self, intrin_match_result, schedule_compute_info, arch="g76"):
        self.intrin_match_result = intrin_match_result
        # get match hw_abs_dag info
        hw_abs_dag = intrin_match_result.hw_abs_dag
        compute_key = intrin_match_result.compute_key
        shape_key = intrin_match_result.shape_key
        self.hw_abs_dag = hw_abs_dag
        self.compute_key = compute_key
        self.shape_key = shape_key

        self.target_dag: ComputeDAG = schedule_compute_info.target_dag
        self.main_op = schedule_compute_info.main_op
        self.output_op = schedule_compute_info.output_op
        # we use last_op to substitute output_op
        # and refer to last_op as output_op
        assert self.output_op is None
        self.main_op_id = schedule_compute_info.main_op_id
        self.output_op_id = schedule_compute_info.output_op_id
        self._last_op_id = len(self.target_dag.op_lst) - 1
        self._last_op = self.target_dag.op_lst[-1]

        self.hw_abs_dag_stage = schedule_compute_info.hw_abs_dag_stage
        # the state during schedule
        self.state = empty_mali_state()
        # the parameters during schedule
        self.params = empty_mali_params()
        # some constants
        self.warp_size = Mali(arch=arch).get_warp_size()
        self.bx = tvm.te.thread_axis("blockIdx.x")
        self.ty = tvm.te.thread_axis("threadIdx.y")
        self.tx = tvm.te.thread_axis("threadIdx.x")
        self.obx = tvm.te.thread_axis("blockIdx.x")
        self.oty = tvm.te.thread_axis("threadIdx.y")
        self.otx = tvm.te.thread_axis("threadIdx.x")
        self.n_reduce_tiling_parts = schedule_compute_info.kwargs[
            "reduce_tiling"]
        self.n_spatial_tiling_parts = schedule_compute_info.kwargs[
            "spatial_tiling"]
        # self.n_last_op_tiling_parts = schedule_compute_info.kwargs[
        #     "last_tiling"]

    def _reset_state(self):
        # self.state = {
        #     "inlined": set(),
        #     "main_op_reduce_axis": [],
        #     "output_op_axis": [],
        #     "last_op_axis": [],
        #     "tensorize_iter": {}
        # }
        self.state = empty_mali_state()

    def _reset_parameters(self, params):
        self.params = params

    def get_main_op_outermost_last_reduce_axis(self):
        if len(self.state.main_op_reduce_axis_parts) > 0:
            assert isinstance(self.state.main_op_reduce_axis_parts[0], list)
            assert len(self.state.main_op_reduce_axis_parts[0]) > 0
            return self.state.main_op_reduce_axis_parts[0][-1]
        else:
            # no reduce axis
            # print(self.state.main_op_reduce_axis)
            # print(self.main_op.body)
            raise RuntimeError("No reduce axis in main op.")

    def get_main_op_outermost_first_reduce_axis(self):
        if len(self.state.main_op_reduce_axis_parts) > 0:
            assert isinstance(self.state.main_op_reduce_axis_parts[0], list)
            assert len(self.state.main_op_reduce_axis_parts[0]) > 0
            return self.state.main_op_reduce_axis_parts[0][0]
        else:
            # no reduce axis
            # print(self.state.main_op_reduce_axis)
            # print(self.main_op.body)
            raise RuntimeError("No reduce axis in main op.")

    def get_main_op_second_outermost_last_reduce_axis(self):
        if len(self.state.main_op_reduce_axis_parts) > 1:
            assert isinstance(self.state.main_op_reduce_axis_parts[1], list)
            assert len(self.state.main_op_reduce_axis_parts[1]) > 0
            return self.state.main_op_reduce_axis_parts[1][-1]
        else:
            # no enough reduce axis
            return self.get_main_op_outermost_last_reduce_axis()

    def get_output_op_second_innermost_last_axis(self):
        assert len(self.state.output_op_axis_parts) > 1
        assert isinstance(self.state.output_op_axis_parts[-2], list)
        assert len(self.state.output_op_axis_parts[-2]) > 0
        return self.state.output_op_axis_parts[-2][-1]

    def get_output_op_outermost_last_axis(self):
        assert len(self.state.output_op_axis_parts) > 1
        assert isinstance(self.state.output_op_axis_parts[0], list)
        assert len(self.state.output_op_axis_parts[0]) > 0
        return self.state.output_op_axis_parts[0][-1]

    def get_output_op_innermost_last_axis(self):
        assert len(self.state.output_op_axis_parts) > 1
        assert isinstance(self.state.output_op_axis_parts[-1], list)
        assert len(self.state.output_op_axis_parts[-1]) > 0
        return self.state.output_op_axis_parts[-1][-1]

    # def get_last_op_innermost_last_axis(self):
    #     assert len(self.state.last_op_axis_parts) > 0
    #     assert isinstance(self.state.last_op_axis_parts[-1], list)
    #     assert len(self.state.last_op_axis_parts[-1]) > 0
    #     return self.state.last_op_axis_parts[-1][-1]

    # def get_last_op_second_innermost_last_axis(self):
    #     assert len(self.state.last_op_axis_parts) > 0
    #     assert isinstance(self.state.last_op_axis_parts[-1], list)
    #     assert len(self.state.last_op_axis_parts[-1]) > 1
    #     return self.state.last_op_axis_parts[-1][-2]

    # def get_last_op_outermost_last_axis(self):
    #     assert len(self.state.last_op_axis_parts) > 0
    #     assert isinstance(
    #         self.state.last_op_axis_parts[0], list), \
    #         self.state.last_op_axis_parts[0]
    #     assert len(self.state.last_op_axis_parts[0]) > 0
    #     return self.state.last_op_axis_parts[0][0]

    def get_tensorized_axis(self, op):
        assert op in self.state.tensorized_axis
        return self.state.tensorized_axis[op]

    def get_output_op_warp_numbers(self):
        assert len(self.params.spatial_factors) > 0
        ret = 1
        for part in self.params.spatial_factors:
            assert len(part[0]) > 1
            ret *= part[0][-3]
        return ret

    def get_main_op_reduce_axis_factors(self, number):
        assert len(self.params.reduce_factors) >= number
        return [x[0] for x in self.params.reduce_factors[:number]]

    def get_output_op_axis_factors(self, number):
        assert len(self.params.spatial_factors) >= number, (
            self.params.to_json(), number)
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

    def get_vectorize_length(self):
        assert self.params.vectorize is not None
        return self.params.vectorize[0]

    def get_output_op_unroll_step(self):
        assert self.params.output_unroll_step is not None
        return self.params.output_unroll_step[0]

    # def get_last_op_unroll_step(self):
    #     assert self.params.last_unroll_step is not None
    #     return self.params.last_unroll_step[0]

    @staticmethod
    def _split_axes(stage: tvm.te.Stage, axes, factors):
        axis_split_parts = []
        for iv, factors in zip(axes, factors):
            part = []
            for f in reversed(factors[1:]):
                iv, inner = stage.split(iv, factor=f)
                part.append(inner)
            part.append(iv)
            axis_split_parts.append(list(reversed(part)))
        return axis_split_parts

    @staticmethod
    def _reorder_axis_parts(stage: tvm.te.Stage, *parts):
        ordered_axes = [iv for part in parts for iv in part]
        stage.reorder(*ordered_axes)
        return ordered_axes

    def _check_parameter_ready(self):
        return True

    def inline(self, op_id, op, sch, X):
        if op in self.hw_abs_dag_stage.operation_role:
            return
        else:
            if can_inline(op, self.target_dag):
                sch[X(op)].compute_inline()
                self.state.inlined_ops.add(op)

    def cache_read(self, op_id, op, sch, X):
        if op in self.hw_abs_dag_stage.operation_role:
            return
        # if op in self.state.inlined_ops:
        #     return
        if op not in self.target_dag.feed_graph:
            return
        do_cache_read_for_load = False
        # do_cache_read_for_last = False
        consumers = self.target_dag.feed_graph[op]
        if len(consumers) <= 0:
            return
        if consumers[0] in self.hw_abs_dag_stage.operation_role:
            if (self.hw_abs_dag_stage.operation_role[consumers[0]]
                    == OperationRole.main_op):
                if len(consumers) == 1:
                    do_cache_read_for_load = True
        # if consumers[0] == self.target_dag.op_lst[-1]:
        #     # the last op
        #     if len(consumers) == 1:
        #         do_cache_read_for_last = True

        # can't do both
        # assert not (do_cache_read_for_load and do_cache_read_for_last)

        if do_cache_read_for_load:
            S = sch.cache_read(X(op).output(0), "local",
                               [X(x) for x in consumers])
            axis = self.get_main_op_second_outermost_last_reduce_axis()
            # compute at to main op
            sch[S].compute_at(sch[X(self.main_op)], axis)
            warp_num = self.get_output_op_warp_numbers()
            vec_len = self.get_vectorize_length()
            fused = sch[S].fuse(*sch[S].op.axis)
            fused, vectorized = sch[S].split(fused, factor=vec_len)
            fused, thread_level = sch[S].split(fused, factor=self.warp_size)
            fused, warp_level = sch[S].split(fused, factor=warp_num)
            sch[S].bind(thread_level, self.tx)
            sch[S].bind(warp_level, self.ty)
            sch[S].vectorize(vectorized)

        # if do_cache_read_for_last:
        #     last_ops = [X(x) for x in consumers]
        #     S = sch.cache_read(X(op).output(0), "shared", last_ops)
        #     axis = self.get_last_op_innermost_last_axis()
        #     # compute at to last op
        #     sch[S].compute_at(sch[last_ops[0]], axis)
        #     warp_num = self.get_last_op_warp_numbers()
        #     fused = sch[S].fuse(*sch[S].op.axis)
        #     fused, thread_level = sch[S].split(fused, factor=self.warp_size)
        #     fused, warp_level = sch[S].split(fused, factor=warp_num)
        #     sch[S].bind(thread_level, self.otx)
        #     sch[S].bind(warp_level, self.oty)

    def set_scope(self, op_id, op, sch, X):
        if op in self.hw_abs_dag_stage.operation_role:
            # do not set scope for output op
            if self.hw_abs_dag_stage.operation_role[op] != OperationRole.main_op:
                # only handle register level
                sch[X(op)].set_scope("local")

    def tiling(self, op_id, op, sch, X):
        if op in self.state.inlined_ops:
            return
        op_stage: tvm.te.Stage = sch[X(op)]
        # only tiling for 3 ops: main, output, last
        if op_id == self.main_op_id:
            # compute at output op
            axis = self.get_output_op_second_innermost_last_axis()
            op_stage.compute_at(sch[X(self._last_op)], axis)

            # prepare spatial axis
            all_spatial_axes = op_stage.op.axis
            reserved_spatial_num = int(
                self.hw_abs_dag_stage.reserve_inner_axis_count[op])

            # split spatial axis
            spatial_axis_split_parts = [
                all_spatial_axes[:-reserved_spatial_num],
                all_spatial_axes[-reserved_spatial_num:]]

            # prepare reduce axis
            all_reduce_axis = op_stage.op.reduce_axis
            reserved_reduce_axes = []
            split_reduce_axes = []
            main_op_reserved_reduce_axes = set(
                int(x) for x in self.hw_abs_dag_stage.main_op_reserve_reduce_axis)
            for i, iv in enumerate(all_reduce_axis):
                if i in main_op_reserved_reduce_axes:
                    reserved_reduce_axes.append(iv)
                else:
                    split_reduce_axes.append(iv)
            reserved_reduce_num = len(reserved_reduce_axes)

            # split reduce axis
            reduce_axis_split_factors = self.get_main_op_reduce_axis_factors(
                len(split_reduce_axes))
            reduce_axis_split_parts = self._split_axes(
                op_stage, split_reduce_axes, reduce_axis_split_factors)
            reordered_reduce_parts = [
                list(x) for x in zip(*reduce_axis_split_parts)]
            reordered_reduce_parts.append(reserved_reduce_axes)
            assert len(
                reordered_reduce_parts) > 3, "No enough reduce axis split."

            # reorder
            ordered_axes = self._reorder_axis_parts(
                op_stage,
                *reordered_reduce_parts[:-2],
                *spatial_axis_split_parts[:1],
                *reordered_reduce_parts[-2:-1],
                *spatial_axis_split_parts[1:],
                *reordered_reduce_parts[-1:],
            )

            # save state info
            self.state.main_op_reduce_axis_parts = reordered_reduce_parts
            self.state.tensorized_axis[op] = ordered_axes[
                -(reserved_spatial_num + reserved_reduce_num)]
        elif op_id == self._last_op_id:
            # prepare spatial axis
            all_spatial_axes = op_stage.op.axis
            split_spatial_axes = all_spatial_axes

            # split spatial axis
            spatial_axis_split_factors = self.get_output_op_axis_factors(
                len(split_spatial_axes)
            )
            spatial_axis_split_parts = self._split_axes(
                op_stage, split_spatial_axes, spatial_axis_split_factors)
            reordered_spatial_parts = [
                list(x) for x in zip(*spatial_axis_split_parts)]

            # reorder
            self._reorder_axis_parts(op_stage, *reordered_spatial_parts)

            # fuse and bind
            assert len(
                reordered_spatial_parts) > 3, "No enough spatial axis split."
            fused_axes = [op_stage.fuse(*part)
                          for part in reordered_spatial_parts[:-2]]
            final_axis_parts = [[x] for x in fused_axes]
            op_stage.bind(fused_axes[0], self.bx)
            # the intermediate bind to vthread
            # for med_fused in fused_axis[1:-1]:
            #     op_stage.bind(med_fused, tvm.te.thread_axis("vthread"))
            op_stage.bind(fused_axes[-1], self.ty)
            # thread level intrinsic, still bind to thread x
            if self.hw_abs_dag_stage.instruction_scope == InstructionScope.thread:
                fused = op_stage.fuse(*reordered_spatial_parts[-2])
                outer, inner = op_stage.split(fused, nparts=self.warp_size)
                op_stage.bind(outer, self.tx)
                final_axis_parts.append([outer, inner])
            else:
                final_axis_parts.append(reordered_spatial_parts[-2])
            final_axis_parts.append(reordered_spatial_parts[-1])

            # save state info
            self.state.output_op_axis_parts = final_axis_parts
        else:
            # TODO(chenrenze): tiling for other axis
            pass

    def compute_at(self, op_id, op, sch, X):
        op_stage: tvm.te.Stage = sch[X(op)]
        if op not in self.hw_abs_dag_stage.operation_role:
            return
        if op_id < self.main_op_id:
            # compute at to main op
            axis = self.get_main_op_second_outermost_last_reduce_axis()
            op_stage.compute_at(sch[X(self.main_op)], axis)
            reserve_spatial_num = int(
                self.hw_abs_dag_stage.reserve_inner_axis_count[op])
            self.state.tensorized_axis[op] = op_stage.op.axis[
                -reserve_spatial_num]

    def unroll(self, op_id, op, sch, X):
        op_stage: tvm.te.Stage = sch[X(op)]
        if op_id == self._last_op_id:
            axis = self.get_output_op_outermost_last_axis()
            step = self.get_output_op_unroll_step()
            op_stage.pragma(axis, "auto_unroll_max_step", step)
            # op_stage.pragma(axis, "unroll_explicit", 0)
        elif op_id == self.main_op_id:
            axis = self.get_main_op_outermost_first_reduce_axis()
            # reuse output op unroll step
            step = self.get_output_op_unroll_step()
            op_stage.pragma(axis, "auto_unroll_max_step", step)
            # op_stage.pragma(axis, "unroll_explicit", 0)

    # def vectorize(self, op_id, op, sch, X):
    #     # TODO(chenrenze): vectorize
    #     return
    #     op_stage: tvm.te.Stage = sch[X(op)]
    #     if op_id == self.output_op_id:
    #         axis = self.get_output_op_innermost_last_axis()
    #         vec_len = self.get_vectorize_length()
    #         outer, inner = op_stage.split(axis, factor=vec_len)
    #         op_stage.vectorize(inner)
    #         self.state.output_op_axis_parts[-1].remove(axis)
    #         self.state.output_op_axis_parts[-1].extend([outer, inner])
    #     elif op_id == self._last_op_id:
    #         axis = self.get_last_op_innermost_last_axis()
    #         vec_len = self.get_vectorize_length()
    #         outer, inner = op_stage.split(axis, factor=vec_len)
    #         op_stage.vectorize(inner)
    #         self.state.last_op_axis_parts[-1].remove(axis)
    #         self.state.last_op_axis_parts[-1].extend([outer, inner])

    def tensorize(self, op_id, op, sch, X):
        if op not in self.hw_abs_dag_stage.operation_role:
            return
        intrin = self.hw_abs_dag.get_intrinsic(
            self.compute_key, self.shape_key,
            self.hw_abs_dag_stage.hw_abs_key[op])
        axis = self.get_tensorized_axis(op)
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
            # self.vectorize,
            self.tensorize
        ]

        # initialize parameters
        self._reset_parameters(params)
        # check if parameters are ready
        self._check_parameter_ready()
        # initialize state
        self._reset_state()

        dag = self.target_dag
        for op_id, op in reversed(list(enumerate(dag.op_lst))):
            if not isinstance(op, tvm.te.ComputeOp):
                continue
            else:
                for prim in primitives:
                    prim(op_id, op, sch, X)
        return sch
