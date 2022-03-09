import json
from ...utils import *
from ...target import *
from ...search import CDParamGenerator, Entry, SAEntryGenerator
from ...hw_abs_dag import OperationRole, HwAbsDAGStage, InstructionScope
from ..schedule_base import *
from ...backend import tenet


class CUDAStateTenet(object):
    def __init__(self, inlined, main_op_reduce_axis, output_op_axis, last_op_axis, tensorize_iter):
        self.inlined = inlined
        self.main_op_reduce_axis = main_op_reduce_axis
        self.output_op_axis = output_op_axis
        self.last_op_axis = last_op_axis
        self.tensorize_iter = tensorize_iter


def empty_cuda_state_tenet():
    return CUDAStateTenet(set(), [], [], [], {})


class CUDAParamsTenet(object):
    def __init__(
        self,
        inline,
        vectorize,
        spatial_factors,
        reduce_factors,
        last_factors,
        output_unroll_step,
        last_unroll_step,
    ):
        self.inline = inline
        self.vectorize = vectorize
        self.spatial_factors = spatial_factors
        self.reduce_factors = reduce_factors
        self.last_factors = last_factors
        self.output_unroll_step = output_unroll_step
        self.last_unroll_step = last_unroll_step

    def to_json(self):
        ret = {
            "inline": self.inline,
            "vectorize": self.vectorize,
            "spatial_factors": self.spatial_factors,
            "reduce_factors": self.reduce_factors,
            "last_factors": self.last_factors,
            "output_unroll_step": self.output_unroll_step,
            "last_unroll_step": self.last_unroll_step,
        }
        return ret

    def from_json(self, obj):
        self.inline = obj["inline"]
        self.vectorize = obj["vectorize"]
        self.spatial_factors = obj["spatial_factors"]
        self.reduce_factors = obj["reduce_factors"]
        self.last_factors = obj["last_factors"]
        self.output_unroll_step = obj["output_unroll_step"]
        self.last_unroll_step = obj["last_unroll_step"]

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


def empty_cuda_params_tenet():
    return CUDAParamsTenet(None, None, [], [], [], None, None)


#####################################################
# Target specific parameter generator
#####################################################
class CUDAKernelParamGeneratorTenet(CDParamGenerator):
    pass


class CUDAScheduleGeneratorTenet(AcceleratorScheduleGenerator):
    def __init__(
        self,
        intrin_match_result,
        transform_state,
        eps=0.9,
        reduce_tiling=3,
        spatial_tiling=4,
        last_tiling=3,
        arch=70,
        log_file="cuda_schedule_generator.log",
        steps=1,
        verbose_init=True,
    ):
        super(CUDAScheduleGeneratorTenet, self).__init__(
            eps, CUDAParamsTenet, steps=steps, log_file=log_file, verbose_init=verbose_init
        )
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
        self.last_op_tiling_parts = last_tiling
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
            self.shape_key,
        )
        # nodes: dict {hw_abs name : new tensor}
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

        hw_abs_names, read_graph, feed_graph = self.hw_abs_dag.serialize_dag(cond1=cond)

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
            spatial_axis, reduce_axis = self.hw_abs_dag.get_hw_abs_compute_reserve_axis(
                self.compute_key, self.shape_key, name
            )
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
                    main_op_reserve_reduce_axis.append(len(op.reduce_axis) - len(reduce_axis) + i)
                    main_op_reserve_reduce_axis_factor.append(int(red.dom.extent))
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
            self.hw_abs_dag.scope,
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
                gen = SplitFactorGenerator(int(iv.dom.extent), self.reduce_tiling_parts)
                self.reduce_splits.append(gen)
        # for output op spatial split
        self.spatial_splits = []
        reserve_axis_count = int(self.hw_abs_dag_stage.reserve_inner_axis_count[self.output_op])
        # skip the reserved spatial axis
        for iv in self.output_op.axis[:-reserve_axis_count]:
            gen = SplitFactorGenerator(int(iv.dom.extent), self.spatial_tiling_parts)
            self.spatial_splits.append(gen)
        # for last op spatial axis
        last_total_extent = 1
        for iv in self.target_dag.op_lst[-1].axis:
            last_total_extent *= int(iv.dom.extent)
        self.last_splits = [
            SplitFactorGenerator(
                (last_total_extent + self.warp_size - 1) // self.warp_size,
                self.last_op_tiling_parts,
            )
        ]
        self.inline = InlineGenerator()
        self.vectorize = VectorizeLengthGenerator(
            self.hw_abs_dag.target, self.main_op.input_tensors[0].dtype
        )
        self.unroll_output = UnrollStepGenerator([16, 64, 512, 1500])
        self.unroll_last = UnrollStepGenerator([16, 64, 512, 1500])
        self.generator_lst = [
            self.inline,
            self.vectorize,
            *self.spatial_splits,
            *self.reduce_splits,
            *self.last_splits,
            self.unroll_output,
            self.unroll_last,
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
            spatial_tiling=self.spatial_tiling_parts,
            reduce_tiling=self.reduce_tiling_parts,
            last_tiling=self.last_op_tiling_parts,
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
        warp_num = record.last_factors[0][0][-1]
        if warp_num > max_warps:
            return False
        block_num = record.last_factors[0][0][0]
        if block_num > max_blocks:
            return False
        return True

    def record_from_json(self, obj):
        return self.record_cls(
            obj["inline"],
            obj["vectorize"],
            obj["spatial_factors"],
            obj["reduce_factors"],
            obj["last_factors"],
            obj["output_unroll_step"],
            obj["last_unroll_step"],
        )

    def get_record(self, entry=None, policy="random"):
        if entry is None:
            record = self.record_cls(
                self.inline.get(policy=policy),
                self.vectorize.get(policy=policy),
                [gen.get(policy=policy) for gen in self.spatial_splits],
                [gen.get(policy=policy) for gen in self.reduce_splits],
                [gen.get(policy=policy) for gen in self.last_splits],
                self.unroll_output.get(policy=policy),
                self.unroll_last.get(policy=policy),
            )
        else:
            record = self.record_cls(
                self.inline.get(hint=entry.record.inline[0], policy="q"),
                self.vectorize.get(hint=entry.record.vectorize[0], policy="q"),
                [
                    gen.get(hint=x[0], policy="q")
                    for gen, x in zip(self.spatial_splits, entry.record.spatial_factors)
                ],
                [
                    gen.get(hint=x[0], policy="q")
                    for gen, x in zip(self.reduce_splits, entry.record.reduce_factors)
                ],
                [
                    gen.get(hint=x[0], policy="q")
                    for gen, x in zip(self.last_splits, entry.record.last_factors)
                ],
                self.unroll_output.get(hint=entry.record.output_unroll_step[0], policy="q"),
                self.unroll_last.get(hint=entry.record.last_unroll_step[0], policy="q"),
            )
        return record

    def get_records_mutate_one_generator(self, record, to_mutate, steps):
        inline = record.inline
        vec = record.vectorize
        spatial = record.spatial_factors
        reduce = record.reduce_factors
        last = record.last_factors
        unroll_output = record.output_unroll_step
        unroll_last = record.last_unroll_step

        next_inline = self.inline.get_next(inline[0], to_mutate)
        next_vec = self.vectorize.get_next(vec[0], to_mutate)
        next_spatial = [
            gen.get_next(x[0], to_mutate) for gen, x in zip(self.spatial_splits, spatial)
        ]
        next_reduce = [gen.get_next(x[0], to_mutate) for gen, x in zip(self.reduce_splits, reduce)]
        next_last = [gen.get_next(x[0], to_mutate) for gen, x in zip(self.last_splits, last)]
        next_unroll_output = self.unroll_output.get_next(unroll_output[0], to_mutate)
        next_unroll_last = self.unroll_last.get_next(unroll_last[0], to_mutate)

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
            inline = helper(next_inline, inline)
            vec = helper(next_vec, vec)
            spatial = [helper(_gen, org_val) for _gen, org_val in zip(next_spatial, spatial)]
            reduce = [helper(_gen, org_val) for _gen, org_val in zip(next_reduce, reduce)]
            last = [helper(_gen, org_val) for _gen, org_val in zip(next_last, last)]
            unroll_output = helper(next_unroll_output, unroll_output)
            unroll_last = helper(next_unroll_last, unroll_last)
            if has_mutate:
                yield self.record_cls(
                    inline, vec, spatial, reduce, last, unroll_output, unroll_last
                )
            has_mutate = False

    def feedback_value(self, entry, value):
        self.inline.feedback(*entry.record.inline, value)
        self.vectorize.feedback(*entry.record.vectorize, value)
        for gen, factors in zip(self.spatial_splits, entry.record.spatial_factors):
            gen.feedback(*factors, value)
        for gen, factors in zip(self.reduce_splits, entry.record.reduce_factors):
            gen.feedback(*factors, value)
        for gen, factors in zip(self.last_splits, entry.record.last_factors):
            gen.feedback(*factors, value)
        self.unroll_output.feedback(*entry.record.output_unroll_step, value)
        self.unroll_last.feedback(*entry.record.last_unroll_step, value)


class CUDAScheduleApplierTenet(object):
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
        self.state = empty_cuda_state_tenet()
        # the parameters during schedule
        self.params = empty_cuda_params_tenet()
        # some constants
        self.warp_size = CUDA(arch=arch).get_warp_size()
        self.bx = tvm.te.thread_axis("blockIdx.x")
        self.ty = tvm.te.thread_axis("threadIdx.y")
        self.tx = tvm.te.thread_axis("threadIdx.x")
        self.obx = tvm.te.thread_axis("blockIdx.x")
        self.oty = tvm.te.thread_axis("threadIdx.y")
        self.otx = tvm.te.thread_axis("threadIdx.x")
        self.reduce_tiling_parts = schedule_compute_info.kwargs["reduce_tiling"]
        self.spatial_tiling_parts = schedule_compute_info.kwargs["spatial_tiling"]
        self.last_op_tiling_parts = schedule_compute_info.kwargs["last_tiling"]

        self.tenet_ctx = None

    def initialize_state(self):
        # self.state = {
        #     "inlined": set(),
        #     "main_op_reduce_axis": [],
        #     "output_op_axis": [],
        #     "last_op_axis": [],
        #     "tensorize_iter": {}
        # }
        self.state = empty_cuda_state_tenet()

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

    def get_last_op_innermost_last_axis(self):
        assert len(self.state.last_op_axis) > 0
        assert isinstance(self.state.last_op_axis[-1], list)
        assert len(self.state.last_op_axis[-1]) > 0
        return self.state.last_op_axis[-1][-1]

    def get_last_op_second_innermost_last_axis(self):
        assert len(self.state.last_op_axis) > 0
        assert isinstance(self.state.last_op_axis[-1], list)
        assert len(self.state.last_op_axis[-1]) > 1
        return self.state.last_op_axis[-1][-2]

    def get_last_op_outermost_last_axis(self):
        assert len(self.state.last_op_axis) > 0
        assert isinstance(self.state.last_op_axis[0], list), self.state.last_op_axis[0]
        assert len(self.state.last_op_axis[0]) > 0
        return self.state.last_op_axis[0][0]

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
            len(self.params.spatial_factors),
            " vs. ",
            number,
        )
        return [x[0] for x in self.params.spatial_factors[:number]]

    def get_last_op_axis_factors(self, number):
        assert len(self.params.last_factors) >= number
        return [x[0] for x in self.params.last_factors[:number]]

    def get_last_op_warp_numbers(self):
        assert len(self.params.last_factors) > 0
        ret = 1
        for part in self.params.last_factors:
            assert len(part) > 0
            ret *= part[0][-1]
        return ret

    def get_vectorize_length(self):
        assert self.params.vectorize is not None
        return self.params.vectorize[0]

    def get_inline_choice(self):
        assert self.params.inline is not None
        return self.params.inline[0]

    def get_output_op_unroll_step(self):
        assert self.params.output_unroll_step is not None
        return self.params.output_unroll_step[0]

    def get_last_op_unroll_step(self):
        assert self.params.last_unroll_step is not None
        return self.params.last_unroll_step[0]

    def check_parameter_ready(self):
        return True

    def inline(self, op_id, op, sch, X):
        if op in self.hw_abs_dag_stage.operation_role:
            return
        else:
            if not op in self.target_dag.feed_graph:
                return
            else:
                consumers = self.target_dag.feed_graph[op]
                if len(consumers) <= 0:
                    return
                if (
                    len(consumers) == 1
                    and consumers[0] in self.hw_abs_dag_stage.operation_role
                    and self.hw_abs_dag_stage.operation_role[consumers[0]] == OperationRole.load_op
                ):
                    do_inline = self.get_inline_choice()
                    if not do_inline:
                        return
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
        do_cache_read_for_last = False
        consumers = self.target_dag.feed_graph[op]
        if len(consumers) <= 0:
            return
        if consumers[0] in self.hw_abs_dag_stage.operation_role:
            if self.hw_abs_dag_stage.operation_role[consumers[0]] == OperationRole.load_op:
                if len(consumers) == 1:
                    do_cache_read_for_load = True
        # if consumers[0] == self.target_dag.op_lst[-1]:
        #     # the last op
        #     if len(consumers) == 1:
        #         do_cache_read_for_last = True

        # can't do both
        # assert not (do_cache_read_for_load and do_cache_read_for_last)

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
            if self.hw_abs_dag_stage.operation_role[op] != OperationRole.output_op:
                # only handle register level
                sch[X(op)].set_scope("local")

    def tiling(self, op_id, op, sch, X):
        # only tiling for 3 ops: main, output, last
        if op == self.main_op:
            # prepare spatial axis
            axis = sch[X(op)].op.axis
            reserve_spatial_num = int(self.hw_abs_dag_stage.reserve_inner_axis_count[op])
            spatial_axis_split_parts = [axis[:-reserve_spatial_num], axis[-reserve_spatial_num:]]

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
            reduce_axis_split_factors = self.get_main_op_reduce_axis_factors(len(split_reduce_axis))
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
            ordered_axis = (
                reordered_reduce_axis[:-2]
                + [spatial_axis_split_parts[0]]
                + reordered_reduce_axis[-2:-1]
                + [spatial_axis_split_parts[1]]
                + reordered_reduce_axis[-1:]
            )
            ordered_axis = reduce(lambda x, y: x + y, ordered_axis, [])
            sch[X(op)].reorder(*ordered_axis)
            self.state.main_op_reduce_axis = reordered_reduce_axis
            self.state.tensorize_iter[op] = ordered_axis[
                -(reserve_spatial_num + reserve_reduce_num)
            ]
            #################################
            ## update tenet context
            #################################
            ## update time loops
            #################################
            ## this is outermost timeloop
            ## surrounding shared memory
            self.tenet_ctx.update_time_loop(
                0, reduce(lambda x, y: x + y, reordered_reduce_axis[:1], []), push_front=False
            )
            #################################
            ## this is intermediate timeloop
            ## surrounding register
            self.tenet_ctx.update_time_loop(
                1, reduce(lambda x, y: x + y, reordered_reduce_axis[1:-1], []), push_front=False
            )
            #################################
            ## no innermost timeloop
        elif op == self.output_op:
            axis = sch[X(op)].op.axis
            reserve_spatial_num = int(self.hw_abs_dag_stage.reserve_inner_axis_count[op])
            split_spatial_axis = axis[:-reserve_spatial_num]
            reserve_spatial_axis = axis[-reserve_spatial_num:]
            spatial_axis_split_factors = self.get_output_op_axis_factors(len(split_spatial_axis))
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
            #################################
            ## update tenet context
            #################################
            ## update space loops
            #################################
            ## this is outermost spaceloop
            ## surrounding shared memory
            self.tenet_ctx.set_space_loop(0, final_axis[0])
            #################################
            ## this is intermediate spaceloop
            ## surrounding register
            self.tenet_ctx.set_space_loop(1, final_axis[-3])
            #################################
            ## no innermost spaceloop
            #################################
            ## update time loops
            #################################
            ## this is intermediate timeloop
            self.tenet_ctx.update_time_loop(0, reduce(lambda x, y: x + y, final_axis[1:-3], []))
            #################################
            ## this is innermost timeloop
            self.tenet_ctx.update_time_loop(2, final_axis[-2])
            #################################
            ## no innermost timeloop
        elif op == self.target_dag.op_lst[-1]:
            # last op
            if op != self.output_op:
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
                self.state.last_op_axis = [split_parts + [thread_level]]
        else:
            # only tiling for op before tensorize load
            # when inline is not done
            do_tiling = False
            if not op in self.target_dag.feed_graph:
                return
            else:
                consumers = self.target_dag.feed_graph[op]
                if len(consumers) <= 0:
                    return
                if (
                    len(consumers) == 1
                    and consumers[0] in self.hw_abs_dag_stage.operation_role
                    and self.hw_abs_dag_stage.operation_role[consumers[0]] == OperationRole.load_op
                ):
                    do_inline = self.get_inline_choice()
                    if not do_inline:
                        do_tiling = True
            if do_tiling:
                bx = tvm.te.thread_axis("blockIdx.x")
                ty = tvm.te.thread_axis("threadIdx.y")
                tx = tvm.te.thread_axis("threadIdx.x")
                axis = sch[X(op)].op.axis
                fused = sch[X(op)].fuse(*axis)
                fused, thread_level = sch[X(op)].split(fused, factor=self.warp_size)
                # reuse the factors of output op
                split_factors = self.get_last_op_axis_factors(1)
                split_parts = []
                for f in reversed(split_factors[0][1:]):
                    fused, inner = sch[X(op)].split(fused, factor=f)
                    split_parts.append(inner)
                split_parts.append(fused)
                split_parts = list(reversed(split_parts))
                sch[X(op)].bind(split_parts[0], bx)
                sch[X(op)].bind(split_parts[-1], ty)
                sch[X(op)].bind(thread_level, tx)

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
        elif op == self.target_dag.op_lst[-1]:
            if op != self.output_op:
                axis = self.get_last_op_outermost_last_axis()
                step = self.get_last_op_unroll_step()
                sch[X(op)].pragma(axis, "auto_unroll_max_step", step)
                # sch[X(op)].pragma(axis, "unroll_explicit", 0)

    def tensorize(self, op_id, op, sch, X):
        if not op in self.hw_abs_dag_stage.operation_role:
            return
        intrin = self.hw_abs_dag.get_intrinsic(
            self.compute_key, self.shape_key, self.hw_abs_dag_stage.hw_abs_key[op]
        )
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
            self.tensorize,
        ]

        # prepare the tenet context
        self.tenet_ctx = tenet.TenetContext(3)
        self.tenet_ctx.set_memory_scope(0, "global")
        self.tenet_ctx.set_memory_scope(1, "shared")
        self.tenet_ctx.set_memory_scope(2, "local")

        # initialize parameters
        self.initialize_parameters(params)
        # check if parameters are ready
        self.check_parameter_ready()
        # initialize state
        self.initialize_state()

        dag = self.target_dag
        total_op = len(dag.op_lst)
        for op_id, op in enumerate(reversed(dag.op_lst)):
            if not isinstance(op, tvm.te.ComputeOp):
                continue
            else:
                for prim in primitives:
                    prim(total_op - op_id - 1, op, sch, X)
        return sch
