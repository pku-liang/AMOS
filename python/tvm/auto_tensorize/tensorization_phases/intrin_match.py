from ..hw_abstraction import ComputeAbstraction, ComputeDAG
from ..hw_abs_dag import query_hw_abs_dag
from ..target import *
import tvm
import tvm._ffi
import tvm.te as te
from .. import _ffi_api


class IntrinMatchResult(object):
    """
    Args:
    ---
    hw_abs_dag: HardwareAbstractionDAG
        the matched hw_abs_dag
    compute_key: str
    shape_key: str
    main_op: dict of {int:int}
    axis_map: dict of {IterVar: list of IterVar}
    target_dag: ComputeDAG
    intrin_dag: ComputeDAG
    """

    def __init__(
        self,
        hw_abs_dag,
        compute_key,
        shape_key,
        main_op_map,
        elem_op_map,
        axis_map,
        target_dag,
        intrin_dag,
    ):
        self.hw_abs_dag = hw_abs_dag
        self.compute_key = compute_key
        self.shape_key = shape_key
        self.main_op_map = main_op_map
        self.elem_op_map = elem_op_map
        self.axis_map = axis_map
        self.target_dag = target_dag
        self.intrin_dag = intrin_dag

    def get_hw_abs_dag(self):
        return self.hw_abs_dag

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

    def __str__(self):
        ret = "MatchResult(hw_abs_dag:%s, compute:%s, shape:%s)" % (
            self.hw_abs_dag.get_name(), self.compute_key, self.shape_key)
        return ret


class IntrinMatcher(object):
    def __init__(self):
        pass

    def match(self, target_dag, target_machine):
        assert target_machine in supported_target
        assert isinstance(target_dag, ComputeDAG)
        # do some match here for target dag and machine
        # get the result is a list of IntrinMatchResult
        return []


def intrinsic_match(target: te.Tensor, intrin: te.Tensor, main_hw_abs: te.TensorComputeOp) -> dict:
    flattened = _ffi_api.MatchIntrinsic(target, intrin, main_hw_abs)
    # Map<Operation, Array<Map<IterVar, IterVar>>>
    results = dict(zip(flattened[0], flattened[1]))

    def key_helper(x):
        s = set()
        for axis in x[0]:
            s.add(axis.var.name)
        return str(list(sorted(tuple(s))))

    new_results = {}
    for op, itervarmaps in results.items():
        sorted_maps = list(sorted(itervarmaps, key=lambda x: key_helper(x)))
        new_results[op] = sorted_maps

    results = {
        op: [dict(zip(m[0], m[1])) for m in itervarmaps] for op, itervarmaps in new_results.items()
    }
    return results


def intrinsic_multi_match(target_dag, intrin_dag, main_op):
    intrin_tensors = list(intrin_dag.tensors)
    # TODO: (yicheng) remove such constraints, do a general DAG match
    assert len(intrin_tensors) == 1
    results = {}
    intrin_tensor = intrin_tensors[0]
    for op in target_dag.op_lst:
        tmp = intrinsic_match(op.output(0), intrin_tensor, main_op)
        results.update(tmp)
    return results


def get_match_result_with_hw_abs_dag(target_dag, hw_abs_dag, compute_key, shape_key):
    """
    target_dag: ComputeDAG
    hw_abs_dag: HardwareAbstractionDAG
    compute_key: str
    shape_key: str
    """
    intrin_dag, main_tensors = hw_abs_dag.get_effective_compute_dag(compute_key, shape_key)
    # target_tensors = list(target_dag.tensors)
    # intrin_tensors = list(intrin_dag.tensors)
    # TODO: (yicheng) remove such constraints, do a general DAG match
    # assert len(target_tensors) == 1
    # assert len(intrin_tensors) == 1
    assert len(main_tensors) == 1
    main_op = main_tensors[0].op

    # raw_match = intrinsic_match(
    #     target_tensors[0],
    #     intrin_tensors[0],
    #     main_op)

    raw_match = intrinsic_multi_match(
        target_dag,
        intrin_dag,
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
        unsorted_tuples = []
        for point in match_points:
            for tiv, iiv in point.items():
                axis_map[iiv].append(tiv)
        match_result = IntrinMatchResult(
            hw_abs_dag, compute_key, shape_key,
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
    for hw_abs_dag_cls in query_hw_abs_dag(target):
        hw_abs_dag = hw_abs_dag_cls()
        for compute_key in hw_abs_dag.get_all_compute_keys():
            for shape_key in hw_abs_dag.get_all_shape_keys():
                ret.extend(
                    get_match_result_with_hw_abs_dag(
                        target_dag, hw_abs_dag, compute_key, shape_key))
    return ret
