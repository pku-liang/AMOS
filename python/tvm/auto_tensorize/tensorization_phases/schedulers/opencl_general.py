import json
from typing import List, Dict, Tuple, Any, Set
from ...utils import *
from ...target import *
from ...search import CDParamGenerator, Entry, SAEntryGenerator
from ..compute_transform import TransformState
from ...hw_abstraction.hw_abs_base import ComputeDAG
from ...hw_abs_dag import OperationRole, HwAbsDAGStage, InstructionScope
from ..schedule_base import *
from tvm.tir import IterVar
from tvm.te import Stage, ComputeOp
from tvm import te

OpId = int


class MaliGeneralParams(object):
    def __init__(self, ops_sp_factors, ops_re_factors,
                 unroll_max_step):
        tmp_type = Dict[OpId, List[Tuple[List[int], Any]]]
        self.ops_sp_factors: tmp_type = ops_sp_factors
        self.ops_re_factors: tmp_type = ops_re_factors
        self.unroll_max_step: Tuple[int, Any] = unroll_max_step
        # self.vectorize_size: Tuple[int, Any] = vectorize_size
        self._normalize()

    def to_json(self):
        return {
            "ops_sp_factors": self.ops_sp_factors,
            "ops_re_factors": self.ops_re_factors,
            "unroll_max_step": self.unroll_max_step,
            # "vectorize_size": self.vectorize_size,
        }

    def from_json(self, obj):
        self.ops_sp_factors = obj["ops_sp_factors"]
        self.ops_re_factors = obj["ops_re_factors"]
        self.unroll_max_step = obj["unroll_max_step"]
        # self.vectorize_size = obj["vectorize_size"]
        self._normalize()

    def _normalize(self):
        self.ops_sp_factors = {int(i): fs for (i, fs)
                               in self.ops_sp_factors.items()}
        self.ops_re_factors = {int(i): fs for (i, fs)
                               in self.ops_re_factors.items()}

    def __str__(self):
        obj = self.to_json()

        def handle(v):
            if isinstance(v, list):
                return [handle(x) for x in v]
            if isinstance(v, dict):
                return {k: handle(x) for (k, x) in v.items()}
            if isinstance(v, tuple) and len(v) == 2:
                return handle(v[0])
            return v

        return json.dumps(handle(obj))


def empty_mali_general_params():
    return MaliGeneralParams(dict(), dict(), (1, 0))


class MaliGeneralScheduleGenerator(AcceleratorScheduleGenerator):
    def __init__(self, target_dag, eps=0.9, arch="g76",
                 n_re_levels=3, n_sp_levels=4,
                 log_file="mali_general_schedule_generator.log", steps=1):
        super(MaliGeneralScheduleGenerator, self).__init__(
            eps, MaliGeneralParams, steps=steps, log_file=log_file)

        self._target_dag = target_dag
        self._n_sp_levels = n_sp_levels
        self._n_re_levels = n_re_levels

        self._ops_sp_split_gens: Dict[OpId, List[CDParamGenerator]] = dict()
        self._ops_re_split_gens: Dict[OpId, List[CDParamGenerator]] = dict()
        # self._vectorize_gen: CDParamGenerator = CDParamGenerator()
        self._unroll_gen: CDParamGenerator = CDParamGenerator()

        self._generators = []

        self._arch_info = Mali(arch)
        self._warp_size = self._arch_info.get_warp_size()

        self.init_param_generator()

        self.score_table = []
        self.init_score_table()

    def init_param_generator(self):
        self._ops_sp_split_gens = dict()
        self._ops_re_split_gens = dict()

        for op_id, op in enumerate(self._target_dag.op_lst):
            if is_heavy_reduce_op(op):
                gens = [
                    SplitFactorGenerator(
                        int(iv.dom.extent), self._n_re_levels
                    ) for iv in op.reduce_axis]
                self._generators.extend(gens)
                self._ops_re_split_gens[op_id] = gens
            if not can_inline(op, self._target_dag):
                gens = [
                    SplitFactorGenerator(
                        int(iv.dom.extent), self._n_sp_levels
                    ) for iv in op.axis]
                self._generators.extend(gens)
                self._ops_sp_split_gens[op_id] = gens
        self._unroll_gen = UnrollStepGenerator([16, 64, 512, 1500])
        self._generators.append(self._unroll_gen)
        # self._vectorize_gen = VectorizeLengthGenerator(
        #     "opencl", self._target_dag.op_lst[0].input_tensors[0].dtype)
        # self._generators.append(self._vectorize_gen)

    def init_score_table(self):
        self.score_table = softmax([0.5 for _ in self._generators])

    def get_generators(self):
        return self._generators

    def get_schedule_compute_info(self):
        raise NotImplementedError(
            "Can get schedule compute info only for tensorizing")

    def valid(self, record):
        return True

    def record_from_json(self, obj):
        return self.record_cls(
            obj["ops_sp_factors"],
            obj["ops_re_factors"],
            obj["unroll_max_step"],
            # obj["vectorize_size"],
        )

    def get_record(self, entry=None, policy="random"):
        if entry is None:
            return self.record_cls(
                {i: [g.get(policy=policy) for g in gens]
                 for (i, gens) in self._ops_sp_split_gens.items()},
                {i: [g.get(policy=policy) for g in gens]
                 for (i, gens) in self._ops_re_split_gens.items()},
                self._unroll_gen.get(policy=policy),
                # self._vectorize_gen.get(policy=policy),
            )
        else:
            return self.record_cls(
                {i: [g.get(hint=fs, policy="q") for (g, (fs, _)) in
                     zip(gens, entry.record.ops_sp_factors[i])] for (i, gens)
                 in self._ops_sp_split_gens.items()},
                {i: [g.get(hint=fs, policy="q") for (g, (fs, _)) in
                     zip(gens, entry.record.ops_re_factors[i])] for (i, gens)
                 in self._ops_re_split_gens.items()},
                self._unroll_gen.get(
                    hint=entry.record.unroll_max_step[0], policy="q"),
                # self._vectorize_gen.get(
                #     hint=entry.record.vectorize_size[0], policy="q"),
            )

    def get_records_mutate_one_generator(self, record, to_mutate, steps):
        sp_fs = record.ops_sp_factors
        re_fs = record.ops_re_factors
        unroll = record.unroll_max_step
        # vectorize = record.vectorize_size

        next_sp = {i: [g.get_next(fs, to_mutate) for (g, (fs, _)) in
                       zip(gens, sp_fs[i])] for (i, gens) in
                   self._ops_sp_split_gens.items()}
        next_re = {i: [g.get_next(fs, to_mutate) for (g, (fs, _)) in
                       zip(gens, re_fs[i])] for (i, gens) in
                   self._ops_re_split_gens.items()}
        next_unroll = self._unroll_gen.get_next(unroll[0], to_mutate)
        # next_vectorize = self._vectorize_gen.get_next(vectorize[0], to_mutate)

        has_mutate = False

        def helper(gen, org_val):
            nonlocal has_mutate
            try:
                ret = next(gen)
                has_mutate = True
            except StopIteration:
                ret = org_val
            return ret

        for s in range(steps):
            sp_fs = {i: [helper(g, fs) for (g, fs) in zip(gens, sp_fs[i])]
                     for (i, gens) in next_sp.items()}
            re_fs = {i: [helper(g, fs) for (g, fs) in zip(gens, re_fs[i])]
                     for (i, gens) in next_re.items()}
            unroll = helper(next_unroll, unroll)
            # vectorize = helper(next_vectorize, vectorize)
            if has_mutate:
                yield self.record_cls(sp_fs, re_fs,
                                      unroll,
                                      # vectorize
                                      )
            has_mutate = False

    def feedback_value(self, entry, value):
        for i, gens in self._ops_sp_split_gens.items():
            for gen, entity in zip(gens, entry.record.ops_sp_factors):
                gen.feedback(*entity, value)
        for i, gens in self._ops_re_split_gens.items():
            for gen, entity in zip(gens, entry.record.ops_re_factors):
                gen.feedback(*entity, value)
        self._unroll_gen.feedback(*entry.record.unroll_max_step, value)
        # self._vectorize_gen.feedback(*entry.record.vectorize_size, value)


class MaliGeneralState(object):
    def __init__(self):
        self.ivs_extent: Dict[IterVar, int] = dict()


def empty_mali_general_state():
    return MaliGeneralState()


class MaliGeneralScheduleApplier(object):
    def __init__(self, target_dag, arch="g76"):
        self.target_dag = target_dag
        self._arch_info = Mali(arch)
        self._warp_size = self._arch_info.get_warp_size()

        self._params = empty_mali_general_params()
        self._state = empty_mali_general_state()

    def _reset_params(self, params):
        self._params = params

    def _reset_state(self):
        self._state = empty_mali_general_state()

    def _check_parameter_ready(self):
        return True

    def _split(self, stage: Stage, iv, factor=None, nparts=None):
        iv_ext = self._get_iv_extent(iv)
        ivo, ivi = stage.split(iv, factor=factor, nparts=nparts)
        if factor is None:
            factor = (iv_ext + nparts - 1) // nparts
        elif nparts is None:
            nparts = (iv_ext + factor - 1) // factor
        self._state.ivs_extent[ivo] = nparts
        self._state.ivs_extent[ivi] = factor
        return ivo, ivi

    def _fuse(self, stage: Stage, *ivs):
        fused_ext = reduce(lambda a, b: a * b,
                           (self._get_iv_extent(iv) for iv in ivs), 1)
        fused_iv = stage.fuse(*ivs)
        self._state.ivs_extent[fused_iv] = fused_ext
        return fused_iv

    def _split_axes(self, stage: Stage, axes, factors):
        split_parts = []
        for iv, factors in zip(axes, factors):
            part = []
            for f in reversed(factors[1:]):
                iv, inner = self._split(stage, iv, factor=f)
                part.append(inner)
            part.append(iv)
            split_parts.append(list(reversed(part)))
        return split_parts

    def _tile_axes(self, stage: Stage, axes, factors):
        split_parts = self._split_axes(stage, axes, factors)
        tiled_parts = [list(x) for x in zip(*split_parts)]
        stage.reorder(*(iv for part in tiled_parts for iv in part))
        return tiled_parts

    def _fuse_axes(self, stage: Stage, tiled_parts):
        fused_parts = []
        for part in tiled_parts:
            n_axes = len(part)
            tmp_part = []
            for sub_part in (part[:n_axes // 2], part[n_axes // 2:]):
                fuse_axis = self._fuse(stage, *sub_part)
                tmp_part.append(fuse_axis)
            fused_parts.append(tmp_part)
        return fused_parts

    def _tile_and_fuse_axes(self, stage: Stage, axes, factors):
        tiled_parts = self._tile_axes(stage, axes, factors)
        return self._fuse_axes(stage, tiled_parts)

    @staticmethod
    def _bind_axes(stage: Stage, tiled_parts):
        thread_parts = [[te.thread_axis(f"blockIdx.{x}") for x in "xyz"],
                        [te.thread_axis(f"threadIdx.{x}") for x in "xyz"]]
        kernel_scope = None
        n_thread_parts = 0
        for tiled_part, thread_part in zip(tiled_parts, thread_parts):
            assert len(tiled_part) <= len(thread_part)
            for tile_iv, thread_iv in zip(tiled_part, thread_part):
                stage.bind(tile_iv, thread_iv)
                kernel_scope = tile_iv
            n_thread_parts += 1
        return kernel_scope, n_thread_parts

    def _split_vectorize(self, stage: Stage, iv):
        iv_ext = self._get_iv_extent(iv)
        if iv_ext > 16:
            ivo, ivi = self._split(stage, iv, factor=16)
            stage.vectorize(ivi)
            return [ivo, ivi]
        largest_power2 = powerx_lst(2, 0, iv_ext + 1)[-1]
        if largest_power2 == iv_ext:
            stage.vectorize(iv)
            return [iv]
        else:
            ivo, ivi = self._split(stage, iv, factor=largest_power2)
            stage.vectorize(ivi)
            return [ivo, ivi]

    def _get_sp_factors(self, op_id):
        return [x[0] for x in self._params.ops_sp_factors[op_id]]

    def _get_re_factors(self, op_id):
        return [x[0] for x in self._params.ops_re_factors[op_id]]

    def _get_iv_extent(self, iv):
        if iv.dom is not None:
            return int(iv.dom.extent)
        return self._state.ivs_extent[iv]

    def _simple_schedule(self, op_id, op, sch, X):
        if can_inline(op, self.target_dag):
            sch[X(op)].compute_inline()
            return

        op_stg: Stage = sch[X(op)]

        def sch_spatial():
            sp_factors = self._get_sp_factors(op_id)
            sp_parts = self._tile_and_fuse_axes(
                op_stg, op_stg.op.axis, sp_factors)

            kernel_scope, n_thread_parts = self._bind_axes(op_stg, sp_parts)
            if len(sp_parts) > n_thread_parts:
                last_iv = sp_parts[-1].pop()
                sp_parts[-1].extend(self._split_vectorize(op_stg, last_iv))
                [op_stg.unroll(iv) for iv in sp_parts[-1][:-1]]

            op_stg.pragma(kernel_scope, "auto_unroll_max_step",
                          self._params.unroll_max_step[0])

            return kernel_scope

        if op_id in self._params.ops_re_factors:
            wc = sch.cache_write(op.output(0), "local")

            op_stg: Stage = sch[X(op)]
            kernel_scope = sch_spatial()

            wc_stg: Stage = sch[X(wc)]
            wc_stg.compute_at(op_stg, kernel_scope)
            re_factors = self._get_re_factors(op_id)
            re_parts = self._split_axes(
                wc_stg, wc_stg.op.reduce_axis, re_factors)
            re_parts = [list(x) for x in zip(*re_parts)]
            sp_parts = [[iv] for iv in wc_stg.op.axis]
            ordered_parts = (
                re_parts[:-1] +
                sp_parts[:1] +
                re_parts[-1:] +
                sp_parts[1:]
            )
            wc_stg.reorder(*(iv for part in ordered_parts for iv in part))
            [wc_stg.unroll(iv) for part in sp_parts for iv in part]
            [wc_stg.unroll(iv) for iv in re_parts[-1]]

        else:
            sch_spatial()

    def apply(self, sch, params, mapping_func=lambda x: x):
        self._reset_params(params)
        self._check_parameter_ready()
        self._reset_state()

        for op_id, op in reversed(list(enumerate(self.target_dag.op_lst))):
            if isinstance(op, ComputeOp):
                self._simple_schedule(op_id, op, sch, mapping_func)

        return sch
