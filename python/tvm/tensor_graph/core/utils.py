import tvm
import tvm._ffi
import sys
import math
import logging
import itertools
from collections import deque
from .tensor import GraphTensor, GraphOp


logger = logging.getLogger("tensor_graph")
formatter = logging.Formatter("[%(levelname)s] %(message)s (%(asctime)s)")

sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(formatter)

logger.setLevel(logging.DEBUG)
logger.addHandler(sh)


def REFUSE(msg):
  raise RuntimeError(msg)


def ERROR(*msg):
  print(*msg)
  sys.exit(1)


def ASSERT(cond, *msg):
  if (not cond):
    ERROR(*msg)


@tvm._ffi.register_func("tg.runtime.call_unpack")
def call_unpack(f, args):
    """ Call PackedFunc with packed arguments.
    """
    f(*args)


def set_call_unpack(call_unpack):
    tvm._ffi.register_func("tg.runtime.call_unpack", call_unpack, True)


def can_to_int(expr):
    try:
        res = int(expr)
        return True
    except Exception as e:
        return False


def to_int_or_None(expr):
    try:
        res = int(expr)
    except Exception as e:
        res = None
    return res


def to_int(expr):
    try:
        res = int(expr)
    except Exception as e:
        raise RuntimeError("fail to convert to int: %s" % str(e))
    return res


def to_tuple(expr_tuple):
    return tuple([to_int(x) for x in expr_tuple])


def power_of_x_near(x, near):
    return x**math.floor(math.log(near, x))


def flatten_forward_graph(ops):
  bfs_order = []
  down_graph = {}
  visited = set()
  q = deque()
  for op in ops:
    q.append(op)
    visited.add(op)
  while q:
    cur = q.popleft()
    if isinstance(cur, GraphOp):
        bfs_order.append(cur)
        for t in cur.inputs:
            if t not in visited:
                visited.add(t)
                q.append(t)
            if t not in down_graph:
                down_graph[t] = []
            down_graph[t].append(cur)
  return list(reversed(bfs_order)), down_graph


def flatten_tir_graph(ops, output_first=False):
    bfs_order = []
    down_graph = {}
    visited = set()
    q = deque()
    for op in ops:
        q.append(op)
        visited.add(op)
    while q:
        cur = q.popleft()
        if isinstance(cur, tvm.te.tensor.ComputeOp):
            bfs_order.append(cur)
        for t in cur.input_tensors:
            if t.op not in visited:
                visited.add(t.op)
                q.append(t.op)
            if t not in down_graph:
                down_graph[t] = []
            down_graph[t].append(cur)
    if output_first:
        return bfs_order, down_graph
    return list(reversed(bfs_order)), down_graph


def op_feature(op):
    if op.tag == "":
        print(op.name, isinstance(op, tvm.te.tensor.ComputeOp))
    assert isinstance(op, tvm.te.tensor.ComputeOp)
    shape = tuple([to_int(x.dom.extent) for x in op.axis])
    if hasattr(op, "reduce_axis"):
        reduction = tuple([to_int(x.dom.extent) for x in op.reduce_axis])
    else:
        reduction = tuple([])
    feature = str(shape) + str(reduction)
    for inp in op.input_tensors:
        feature += str(to_tuple(inp.shape))
    feature += op.tag

    return feature


def is_power_of_x(x, val):
    assert isinstance(val, int) and val > 0
    return math.fabs(math.pow(x, int(math.log(val, x))) - val) < 1e-20


def powerx_lst(x, left, right):
    ret = []
    beg = 1
    while beg < left:
        beg *= x
    while beg < right:
        ret.append(beg)
        beg = beg * x
    return ret


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


def any_factor_split(value, number, allow_non_divisible='off'):
    assert allow_non_divisible in ['off', 'power2', 'continuous']
    ret = []
    assert (isinstance(number, int))
    recursive_factor_split(value, [], number, ret, allow_non_divisible)
    return ret


def recursive_factor_split(left, cur, number, ret, policy):
    if number == 1:
        ret.append(cur + [left])
        return
    if policy == 'power2':
        f_lst = get_factor_lst(left)
        f_lst.extend(powerx_lst(2, 1, left))
        f_lst = list(set(f_lst))
    elif policy == 'continuous':
        f_lst = list(range(1, left + 1))
    else:
        f_lst = get_factor_lst(left)
        f_lst = sorted(f_lst)
    for f in f_lst:
        recursive_factor_split(left // f, cur + [f], number - 1, ret, policy)


def choose_any_from_any(total, want):
    if total < want:
        ret = itertools.combinations_with_replacement(range(total), want)
        # raise RuntimeError("want to select %d elements from %d elements" % (want, total))
        # _choose_any_from_any([], set(), 0, True, 0, total, want, ret)
    else:
        ret = itertools.combinations(range(total), want)
        # _choose_any_from_any([], set(), 0, False, 0, total, want, ret)
    return list(ret)


def _choose_any_from_any(cur, chose, val, repeat, num, total, want, ret):
    if num == want:
        ret.append(cur)
        return
    if repeat:
        for i in range(0, total):
            _choose_any_from_any(cur + [i], chose, i+1, repeat, num+1, total, want, ret)
    else:
        for i in range(val, total):
            if i not in chose:
                chose.add(i)
                _choose_any_from_any(cur + [i], chose, i+1, repeat, num+1, total, want, ret)
                chose.remove(i)


def product_of_factor_lists(*factor_lists):
    ret = itertools.product(*factor_lists)
    return ret


class SplitFactorCache(object):
    def __init__(self):
        self.cache = {}
        for parts in [2, 3, 4]:
            for i in range(12):
                value = 2**i
                f_list = any_factor_split(value, parts, "power2")
                self.cache[(value, parts)] = f_list
    
    def query(self, value, parts):
        if (value, parts) in self.cache:
            return self.cache[(value, parts)]
        self.cache[(value, parts)] = any_factor_split(value, parts, "power2")
        return self.cache[(value, parts)]


class ChooseCache(object):
    def __init__(self):
        self.cache = {}

    def query(self, total, want):
        if (total, want) in self.cache:
            return self.cache[(total, want)]
        self.cache[(total, want)] = choose_any_from_any(total, want)
        return self.cache[(total, want)]


class UtilCache(object):
    def __init__(self):
        self.split_cache = SplitFactorCache()
        self.decompose_cache = {}
        self.choose_cache = ChooseCache()

    def query_split(self, value, parts):
        return self.split_cache.query(value, parts)

    def query_choose(self, total, want):
        return self.choose_cache.query(total, want)

    def query_decompose(self, extent_list, parts):
        if (extent_list, parts) in self.decompose_cache:
            return self.decompose_cache[(extent_list, parts)]
        
        factors_list = []
        for extent in extent_list:
            factors = self.query_split(extent, parts)
            factors_list.append(factors)
        ret = list(product_of_factor_lists(*factors_list))
        self.decompose_cache[(extent_list, parts)] = ret
        return ret



# this is global cache
util_cache = UtilCache()