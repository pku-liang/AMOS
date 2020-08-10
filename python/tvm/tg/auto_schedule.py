"""
Author: size zheng
"""

"""Longtail related functions."""
import itertools
import tvm
import tvm._ffi
from . import _ffi_api
from tvm import target as _target


# @tvm._ffi.register_func("tg.utils.permutation")
def permutation(num_total):
  return list(itertools.permutations(list(range(num_total))))


class UtilBox(object):
  @staticmethod
  def any_part_split(extent, nparts, policy="normal"):
    """Split extent to nparts according to policy

    Parameters
    ----------
    extent : int
        The value to be split.

    nparts : int
        How many parts to split.

    policy : str
        The split policy: normal or power2.

    Returns
    -------
    factors: list of list of int
        The split factors.
    """
    return _ffi_api.any_part_split(extent, nparts, policy)

  @staticmethod
  def permutation(num_total):
    """Permute the int number in [0, num_total)

    Parameters
    ----------
    num_total: int
        The upper bound of permute range.

    Returns
    -------
    permutations: list of list of int
        The permutations.
    """
    # return _ffi_api.permutation(num_total)
    return permutation(num_total)

  @staticmethod
  def choose_from(total, want):
    """Choose want numbers from int range [0, total)

    Parameters
    ----------
    total: int
        The upper bound of choose range.
    
    want: int
        The number of wanted numbers.

    Returns
    -------
    choices: list of list of int
        The choices.
    """
    return _ffi_api.choose_from(total, want)


def get_schedule_skeletons(graph, target):
  """Get the schedule skeletons from a given graph
  This is function is used to test
  the internal of TensorGraph AutoScheduler

  Parameters
  ----------
  graph: TIRGraph
  
  target: tvm.target.Target

  Returns
  -------
  skeletons: list of list of ScheduleSkeleton
  """
  target = _target.create(target)
  return _ffi_api.get_schedule_skeletons(graph, target)


def get_schedule_entities(graph, target, number):
  """Get the schedule entities from a given graph
  This is function is used to test
  the internal of TensorGraph AutoScheduler

  Parameters
  ----------
  graph: TIRGraph
  
  target: tvm.target.Target

  number: int

  Returns
  -------
  entities: list of list of MultiScheduleEntity
  """
  target = _target.create(target)
  return _ffi_api.get_schedule_entities(graph, target, number)


def schedule_entity_to_string(entity):
  """Get the string representation of ScheduleEntity
  This is function is used to test
  the internal of TensorGraph AutoScheduler

  Parameters
  ----------
  entity: ScheduleEntity

  Returns
  -------
  str
  """
  return _ffi_api.schedule_entity_to_string(entity)


def string_to_schedule_entity(string):
  """Get the ScheduleEntity from string representation
  This is function is used to test
  the internal of TensorGraph AutoScheduler

  Parameters
  ----------
  string: str

  Returns
  -------
  ScheduleEntity
  """
  return _ffi_api.schedule_entity_from_string(string)


#   Feature get_feature(
#     te::Schedule sch, const Array<te::Tensor>& tensors, Target target)
#   Array<Array<Array<PrimExpr>>> get_feature_structured(
#     te::Schedule sch, const Array<te::Tensor>& tensors, Target target)
#   """
#   if flatten:
#     features = _ffi_api.get_feature(schedule, tensors, target)
#     features = [v.value for v in features.features]
#   else:
#     def pythonify_features(f):
#       if isinstance(f, tvm.ir.container.Array):
#         return [pythonify_features(ff) for ff in f]
#       else:
#         return f.name if isinstance(f, tvm.tir.expr.Var) else f.value

#     def feature_row_to_dict(row):
#       from collections import defaultdict
#       LOOP_ANNOTS = [
#         'kBlockX', 'kBlockY', 'kBlockZ', 'kThreadX', 'kThreadY', 'kThreadZ',
#         'kUnrolled', 'kVectorized', 'kParallel', 'kSerial', 'kVirtualThread',
#       ]
#       TOUCH_PATTERN_DEF = ('stride', 'mod', 'bytes', 'reuse', 'thread_count', 'thread_reuse', 'loop_reuse')
#       FEATURE_DEF = defaultdict(lambda: TOUCH_PATTERN_DEF)
#       FEATURE_DEF.update({
#         '_itervar_': ('name',),
#         '_access_type_': ('write', 'read',),
#         '_attr_': ('length', 'nest_level', 'topdown', 'bottomup', *LOOP_ANNOTS, 'serial_reuse',),
#         '_arith_': ('add_ct', 'mul_ct', 'div_ct',),
#       })
#       return {
#         entry[0]: dict(zip(FEATURE_DEF[entry[0]], entry[1:]))
#         for entry in row
#       }

#     def untake_log(row):
#       from collections import defaultdict

#       def weak_round(x, eps=1e-6):
#         return round(x) if abs(x - round(x)) < eps else x

#       def unlog(x):
#         y1 = weak_round(pow(2, x) - 1)
#         y2 = weak_round(1 - pow(2, -x))
#         return y1 if y1 >= 0 else y2

#       SHOULD_UNLOG = defaultdict(lambda:('stride', 'mod', 'bytes', 'reuse', 'thread_count', 'thread_reuse'))
#       SHOULD_UNLOG.update({
#         '_itervar_': (),
#         '_access_type_': (),
#         '_attr_': ('length', 'topdown', 'bottomup',),
#         '_arith_': ('add_ct', 'mul_ct', 'div_ct',),
#       })

#       return {
#         k: {kk: unlog(vv) if kk in SHOULD_UNLOG[k] else vv for kk, vv in v.items()}
#         for k, v in row.items()
#       }

#     features = _ffi_api.get_feature_structured(schedule, tensors, target)
#     features = pythonify_features(features)
#     features = [feature_row_to_dict(row) for row in features]
#     features = [untake_log(row) for row in features]
  
#   return features


def get_feature(schedule, tensors, target, flatten=True):
  if flatten:
    features = _ffi_api.get_feature(schedule, tensors, target)
    features = [[v.value for v in fea.features] for fea in features]
  else:
    BUFFER_ACCESS_PATTERN_DEF = (
      'access_type', 'bytes', 'unique_bytes', 'lines', 'unique_lines',
      'reuse_type', 'reuse_distance', 'reuse_counter', 'stride', 'topdown',
    )
    INTRIN_KEYS = (
      "exp", "exp2", "exp10", "erf", "tanh", "sigmoid", "log", "log2", "log10",
      "tan", "cos", "cosh", "sin", "sinh", "atan", "sqrt", "rsqrt",
    )
    ARITH_FEATURE_KEYS = (
      'add', 'sub', 'mul', 'div', 'mod', 'cmp', *INTRIN_KEYS
    )
    ANNOT_FEATURE_KEYS = ('len_imost', 'len_prod', 'loop_num', 'loop_pos')
    THREAD_BIND_KEYS = (
      'kBlockX', 'kBlockY', 'kBlockZ', 'kThreadX', 'kThreadY',
      'kThreadZ', 'kVirtualThread'
    )

    def pythonify_features(f):
      if isinstance(f, tvm.ir.container.Array):
        return [pythonify_features(ff) for ff in f]
      else:
        return f.name if isinstance(f, tvm.tir.expr.Var) else f.value

    def feature_row_to_dict(row):
      from collections import defaultdict
      
      FEATURE_DEF = defaultdict(lambda: BUFFER_ACCESS_PATTERN_DEF)
      FEATURE_DEF.update({
        '_stmt_': ('name',),
        'int_arith_features': ARITH_FEATURE_KEYS,
        'flt_arith_features': ARITH_FEATURE_KEYS,
        'vectorization_features': ANNOT_FEATURE_KEYS,
        'unrolling_features': ANNOT_FEATURE_KEYS,
        'parallel_features': ANNOT_FEATURE_KEYS,
        'thread_binding_features': THREAD_BIND_KEYS,
        'allocation_features': ('num_alloc', *[f'out_size{i}' for i in range(10)]),
        'other_features': ('num_outer_loops', 'prod_outer_loops', 'auto_unroll_max_step'),
      })
      return {
        entry[0]: dict(zip(FEATURE_DEF[entry[0]], entry[1:]))
        for entry in row
      }

    def untake_log(row):
      from collections import defaultdict

      def weak_round(x, eps=1e-4):
        return round(x) if abs(x - round(x)) < eps else x

      def unlog(x):
        y1 = weak_round(pow(2, x) - 1)
        y2 = weak_round(1 - pow(2, -x))
        return y1 if y1 >= 0 else y2

      SHOULD_UNLOG = defaultdict(lambda: (
        'bytes', 'unique_bytes', 'lines', 'unique_lines',
        'reuse_distance', 'reuse_counter', 'stride', 'topdown',
      ))
      SHOULD_UNLOG.update({
        '_stmt_': (),
        'int_arith_features': ARITH_FEATURE_KEYS,
        'flt_arith_features': ARITH_FEATURE_KEYS,
        'vectorization_features': ('len_imost', 'len_prod', 'loop_num'),
        'unrolling_features': ('len_imost', 'len_prod', 'loop_num'),
        'parallel_features': ('len_imost', 'len_prod', 'loop_num'),
        'thread_binding_features': THREAD_BIND_KEYS,
        'allocation_features': ('num_alloc', *[f'out_size{i}' for i in range(10)]),
        'other_features': ('num_outer_loops', 'prod_outer_loops', 'auto_unroll_max_step'),
      })

      return {
        k: {kk: unlog(vv) if kk in SHOULD_UNLOG[k] else vv for kk, vv in v.items()}
        for k, v in row.items()
      }

    def get_enum_names(row):
      from collections import defaultdict

      def convert(k, v):
        if k == 'access_type':
          return ['kNone', 'kRead', 'kWrite', 'kReadWrite'][v]
        elif k == 'reuse_type':
          return ['kNoReuse', 'kLoopMultipleRead', 'kSerialMultipleRead', 'kBothReuse'][v]
        elif k == 'loop_pos':
          return ['kNonePosition', 'kInnerSpatial', 'kMiddleSpatial', 'kOuterSpatial',
            'kInnerReduce', 'kMiddleReduce', 'kOuterReduce', 'kMixedPosition'][v]
        else:
          raise ValueError(f"Unrecognized enum: {k}")

      SHOULD_CONVERT = defaultdict(lambda: (
        'access_type', 'reuse_type',
      ))
      SHOULD_CONVERT.update({
        '_stmt_': (),
        'int_arith_features': (),
        'flt_arith_features': (),
        'vectorization_features': ('loop_pos'),
        'unrolling_features': ('loop_pos'),
        'parallel_features': ('loop_pos'),
        'thread_binding_features': (),
        'allocation_features': (),
        'other_features': (),
      })

      return {
        k: {kk: convert(kk, vv) if kk in SHOULD_CONVERT[k] else vv for kk, vv in v.items()}
        for k, v in row.items()
      }

    features = _ffi_api.get_structured_feature(schedule, tensors, target)
    features = pythonify_features(features.features)
    features = [feature_row_to_dict(row) for row in features]
    features = [untake_log(row) for row in features]
    features = [get_enum_names(row) for row in features]

  return features
