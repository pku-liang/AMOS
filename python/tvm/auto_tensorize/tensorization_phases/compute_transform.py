import tvm._ffi
from tvm.runtime import Object
from ..capsule_base import ComputeDAG
from .intrin_match import IntrinMatchResult
from .. import _ffi_api


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
    axis_map:
    reverse_axis_map:
    time_loops:
    """

    def __init__(self, axis_map, reverse_axis_map, space_loops, time_loops):
        self.__init_handle_by_constructor__(
            _ffi_api.TransformRequest,
            axis_map,
            reverse_axis_map,
            space_loops,
            time_loops)


class TransformationStep(object):
    pass


class UnfoldStep(TransformationStep):
    def __init__(self):
        pass

    def transform(self, state):
        """
        state: TransformState
        """
        target_dag = state.target_dag
        intrin_dag = state.intrin_dag
        main_op_map = state.main_op_map
        assert len(main_op_map) > 0
        for k, v in main_op_map:
            intrin_main_op = intrin_dag.op_lst[k]
            target_main_op = target_dag.op_lst[v]
        assert len(intrin_main_op.input_tensors) == len(
            target_main_op.input_tensors)
        for i, intrin_inp in enumerate(intrin_main_op.input_tensors):
            target_inp = target_main_op.input_tensors[i]


class TransformationStepList(object):
    def __init__(self):
        self.steps = []
        self.cur_idx = 0

    def append(self, step):
        assert isinstance(TransformationStep)
        self.steps.append(step)

    def __getitem__(self, idx):
        return self.steps[idx]

    def __iter__(self):
        return iter(self.steps)


class ComputeTransformResult(object):
    def __init__(self, match_result, transform_steps):
        assert isinstance(match_result, IntrinMatchResult)
        assert isinstance(transform_steps, TransformationStepList)
        self.match_result = match_result
        self.transform_steps = transform_steps


class ComputeTransformer(object):
    def __init__(self):
        pass

    def transform(self, intrin_match_result):
        assert isinstance(intrin_match_result, IntrinMatchResult)

        # return list of ComputeTransformResult
        return []


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
