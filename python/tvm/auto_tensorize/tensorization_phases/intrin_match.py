from ..capsule_base import ComputeCapsule, ComputeDAG, query_recipe
from ..target import *
import tvm
import tvm._ffi
import tvm.te as te
from .. import _ffi_api


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

    def __init__(
        self,
        recipe,
        compute_key,
        shape_key,
        main_op_map,
        elem_op_map,
        axis_map,
        target_dag,
        intrin_dag,
    ):
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


def intrinsic_match(target: te.Tensor, intrin: te.Tensor, main_capsule: te.TensorComputeOp) -> dict:
    flattened = _ffi_api.MatchIntrinsic(target, intrin, main_capsule)
    # Map<Operation, Array<Map<IterVar, IterVar>>>
    results = dict(zip(flattened[0], flattened[1]))
    results = {
        op: [dict(zip(m[0], m[1])) for m in itervarmaps] for op, itervarmaps in results.items()
    }
    return results


def get_match_result_with_recipe(target_dag, recipe, compute_key, shape_key):
    """
    target_dag: ComputeDAG
    recipe: CompilationRecipe
    compute_key: str
    shape_key: str
    """
    intrin_dag, main_tensors = recipe.get_effective_compute_dag(compute_key, shape_key)
    target_tensors = list(target_dag.tensors)
    intrin_tensors = list(intrin_dag.tensors)
    # TODO: (yicheng) remove such constraints, do a general DAG match
    assert len(target_tensors) == 1
    assert len(intrin_tensors) == 1
    assert len(main_tensors) == 1
    main_op = main_tensors[0].op

    raw_match = intrinsic_match(
        target_tensors[0],
        intrin_tensors[0],
        main_op)

    match_results = []
    for top, match_points in raw_match.items():
        main_op_map = {
            main_op: top
        }
        # TODO: (size): elem mapping seems not necessary
        elem_op_map = {}
        intrin_axis = main_op.axis
        intrin_reduce_axis = main_op.reduce_axis
        axis_map = {
            iiv : [] for iiv in list(intrin_axis) + list(intrin_reduce_axis)
        }
        for point in match_points:
            for tiv, iiv in point.items():
                axis_map[iiv].append(tiv)
        match_result = IntrinMatchResult(
            recipe, compute_key, shape_key,
            main_op_map, elem_op_map,
            axis_map, target_dag, intrin_dag
        )
        match_results.append(match_result)
    return match_results


def get_match_results(target_dag, target):
    """
    target_dag: ComputeDAG
    target: str
    """
    ret = []
    for recipe_cls in query_recipe(target):
        recipe = recipe_cls()
        for compute_key in recipe.get_all_compute_keys():
            for shape_key in recipe.get_all_shape_keys():
                ret.extend(
                    get_match_result_with_recipe(
                        target_dag, recipe, compute_key, shape_key))
    return ret
