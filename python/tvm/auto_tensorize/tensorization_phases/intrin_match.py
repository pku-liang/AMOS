from ..capsule_base import ComputeCapsule, ComputeDAG
from .target import *


class IntrinMatchResult(object):
    """
    Args:
    ---
    recipe: CompilationRecipe
        the matched recipe
    compute_key: str
    shape_key: str
    main_op: dict of {int:int}
    axis_map: dict of {IterVar: list of IterVar}
    target_dag: ComputeDAG
    intrin_dag: ComputeDAG
    """
    def __init__(self, recipe, compute_key, shape_key,
                 main_op_map, elem_op_map,
                 axis_map, target_dag, intrin_dag):
        self.recipe = recipe
        self.compute_key = compute_key
        self.shape_key = shape_key
        self.main_op_map = main_op_map
        self.elem_op_map = elem_op_map
        self.axis_map = axis_map
        self.target_dag = target_dag
        self.intrin_dag = intrin_dag

    def get_recipe(self):
        return self.recipe

    def get_compute_key(self):
        return self.compute_key

    def get_shape_key(self):
        return self.shape_key

    def get_main_op_map(self):
        return self.main_op_map

    def get_elem_op_map(self):
        return self.elem_op_map

    def get_target_dag(self):
        return self.target_dag

    def get_intrin_dag(self):
        return self.intrin_dag


class IntrinMatcher(object):
    def __init__(self):
        pass

    def match(self, target_dag, target_machine):
        assert target_machine in supported_target
        assert isinstance(target_dag, ComputeDAG)
        # do some match here for target dag and machine
        # get the result is a list of IntrinMatchResult
        return []
