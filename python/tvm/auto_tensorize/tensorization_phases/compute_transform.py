from ..capsule_base import ComputeDAG
from .intrin_match import IntrinMatchResult


class TransformState(object):
    """
    Args:
    ---
    main_op_map: dict of {int:int}
    elem_op_map: dict of {int:int}
    axis_map: dict of {IterVar:IterVar}
    reverse_axis_map: dict of {IterVar:IterVar}
    target_dag: ComputeDAG
    intrin_dag: ComputeDAG
    """
    def __init__(self, main_op_map, elem_op_map,
                 axis_map, reverse_axis_map, target_dag, intrin_dag):
        self.main_op_map = main_op_map
        self.elem_op_map = elem_op_map
        self.axis_map = axis_map
        self.reverse_axis_map = reverse_axis_map
        self.target_dag = target_dag
        self.intrin_dag = intrin_dag


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
        assert len(intrin_main_op.input_tensors) == len(target_main_op.input_tensors)
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