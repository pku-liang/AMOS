import math
import numpy as np
import tvm
from .recipes import construct_dag
from itertools import permutations, product
from functools import reduce
from . import _ffi_api


####################################################
# schedule parameter functions
####################################################
def get_factor_lst(value):
    assert isinstance(value, int)
    ret = []
    end = math.sqrt(value)
    for i in range(1, math.ceil(end)):
        if value % i == 0:
            ret.append(i)
            ret.append(value // i)
    if end - int(end) < 1e-10 and value % int(end) == 0:
        ret.append(int(end))

    return ret


def powerx_lst(x, left, right):
    ret = []
    beg = 1
    while beg < left:
        beg *= x
    while beg < right:
        ret.append(beg)
        beg = beg * x
    return ret


def any_factor_split(value, number, allow_non_divisible="off"):
    assert allow_non_divisible in ["off", "power2", "continuous"]
    ret = []
    assert isinstance(number, int)
    recursive_factor_split(value, [], number, ret, allow_non_divisible)
    return ret


def recursive_factor_split(left, cur, number, ret, policy):
    if number == 1:
        ret.append(cur + [left])
        return
    if policy == "power2":
        f_lst = get_factor_lst(left)
        f_lst.extend(powerx_lst(2, 1, left))
        f_lst = list(set(f_lst))
    elif policy == "continuous":
        f_lst = list(range(1, left + 1))
    else:
        f_lst = get_factor_lst(left)
        f_lst = sorted(f_lst)
    for f in f_lst:
        recursive_factor_split(left // f, cur + [f], number - 1, ret, policy)


def remap_factors(factor_lst):
    assert isinstance(factor_lst, (list, tuple))
    assert len(factor_lst) > 0
    sample = factor_lst[0]
    assert isinstance(sample, (list, tuple))
    assert len(sample) > 0
    dim = len(sample) - 1
    number_count = {i: set() for i in range(dim + 1)}
    # check the factor list
    for v in factor_lst:
        assert isinstance(v, (list, tuple))
        assert len(v) == dim + 1, dim
        for i, ele in enumerate(v):
            number_count[i].add(ele)
    num_factors = len(number_count[0])
    for k, v in number_count.items():
        assert len(v) == num_factors
    # remap the factor list
    sorted_factors = sorted(number_count[0])
    factor_map = {x: i for i, x in enumerate(sorted_factors)}
    reverse_map = {i: x for i, x in enumerate(sorted_factors)}
    ret = list(map(lambda factors: [factor_map[x] for x in factors], factor_lst))
    return ret, reverse_map, dim, num_factors - 1


def get_directions(dim):
    return list(product([-1, 0, 1], repeat=dim))


def get_partial_directions(dim):
    def worker(v):
        def set_value(i):
            d = [0 for _ in range(dim)]
            d[i] = v
            return d

        return set_value

    return list(map(worker(1), range(dim))) + list(map(worker(-1), range(dim)))


def bi_product(repeat):
    return list(product([0, 1], repeat=repeat))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return (e_x / (e_x.sum() + 1e-5)).tolist()


####################################################
# schedule helper tools
####################################################
def substitute_inputs(org_dag, op_map):
    """Infer ranges for expressions

    Parameters
    ----------
    org_dag: ComputeDAG
    op_map: dict of {Operation: Operation}

    Returns
    -------
    ComputeDAG
    """
    n = _ffi_api.SubstituteInputs(org_dag, op_map)
    return n


def reconstruct_dag_as_intrin(target_dag, main_op, recipe, compute_key, shape_key):
    inputs = list(main_op.input_tensors)
    outputs = [main_op.output(0)]
    # TODO: consider elem op in dag construction
    input_names, output_names, nodes, read_graph, feed_graph = construct_dag(
        recipe, compute_key, shape_key, inputs, outputs, [], outputs
    )
    output_tensors = reduce(lambda x, y: x + y, [nodes[x] for x in output_names], [])
    output = output_tensors[0]
    replace_map = {main_op: output.op}
    result_dag = substitute_inputs(target_dag, replace_map)
    return (result_dag, (input_names, output_names, nodes, read_graph, feed_graph))


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
    return True
