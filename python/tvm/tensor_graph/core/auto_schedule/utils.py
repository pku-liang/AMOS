import tvm
from functools import reduce
from ..utils import to_int, to_int_or_None


def get_need_tile(need_tile):
  return [True if x.value == 1 else False for x in need_tile]


def get_factors(split_factor_entities):
  return [[x.value for x in factors.factors] for factors in split_factor_entities]


def tile_axis(stage, axis, factors, inner_to_outer=False):
  ret = []
  if inner_to_outer:
    factors = list(reversed(factors))
    for f in factors[:-1]:
      axis, inner = stage.split(axis, f)
      ret.append(inner)
    ret.append(axis)
    ret = list(reversed(ret))
  else:
    for f in factors[:-1]:
      outer, axis = stage.split(axis, nparts=f)
      ret.append(outer)
    ret.append(axis)
  return ret


def tile_axes(sch, op, axes, need_tile, split_factors, inner_to_outer=False):
  """Tile axes according to need_tile and split_factors
  """
  axis_map = {}
  count_axis = 0
  split_axis_list = []
  split_factor_list = []
  for axis, need_tile, factors in zip(axes, need_tile, split_factors):
    if need_tile:
      split_axis = tile_axis(sch[op], axis, factors, inner_to_outer=inner_to_outer)
      split_axis_list.append(split_axis)
      split_factor_list.append(factors)
      axis_map[count_axis] = split_axis
    else:
      axis_map[count_axis] = axis
    count_axis += 1
  
  return axis_map, split_axis_list, split_factor_list


def get_bind_spec(binding_entity):
  ret = []
  for b in binding_entity:
    tmp = []
    for bb in b:
      tmp.append([bb[0].value, bb[1].value])
    ret.append(tmp)
  return ret


def bind_axes(sch, op, axis_map, bind, to_bind, already_bind=None, factors=None, extents=None):
  """The bind function will fuse some axes,
  which is dangerous because this is not updated
  to the schedule state. For now it shouldn't be
  a problem because the fusion should only happen
  on blockIdx.z
  """
  ret = []
  for part in bind:
    to_fuse = []
    to_fuse_extent = 1
    for ele in part:
      if ele[1] < 0:
        axis = axis_map[ele[0]]
        if already_bind is not None:
          to_fuse_extent *= extents[ele[0]]
      else:
        axis = axis_map[ele[0]][ele[1]]
        if already_bind is not None:
          to_fuse_extent *= factors[ele[0]][ele[1]]
      to_fuse.append(axis)
      
    if len(to_fuse) > 1:
      sch[op].reorder(*to_fuse)
      fused_axis = sch[op].fuse(*to_fuse)
    else:
      fused_axis = to_fuse[0]
    ret.append(fused_axis)
    sch[op].bind(fused_axis, to_bind)
    if already_bind is not None:
      already_bind["extent"] = to_fuse_extent
  return ret


def get_move_to_inner(move):
  return [x.value for x in move]


def reorder_spatial_and_reduce_axes(sch, op, axis_map, split_axis_list, reduce_split_axis_list, extents_info=None):
  """Reorder spatial and reduce axes
  """
  pre = []
  ones = []
  for k, v in axis_map.items():
    if not isinstance(v, (list, tuple)):
      if v.dom is None:
        ext = None
      else:
        ext = to_int_or_None(v.dom.extent)
      if ext is None:
        if v in extents_info:
          ext = extents_info[v]
        else:
          ERROR("Can't decide extent for %s" % (str(v)))
      
      if ext > 1:
        pre.append(v)
      else:
        ones.append(v)
  # perform local reorder
  num_axis_parts = len(split_axis_list[0]) if len(split_axis_list) > 0 else 0
  num_reduce_axis_parts = len(reduce_split_axis_list[0]) if len(reduce_split_axis_list) > 0 else 0

  leveled_axes = []
  reduce_leveled_axes = []
  local_order = []

  def _inner(axis_list, leveled, nparts):
    for i in range(nparts):
      leveled.append([])
    for part in axis_list:
      for i, axis in enumerate(part):
        leveled[i].append(axis)
  
  _inner(split_axis_list, leveled_axes, num_axis_parts)
  _inner(reduce_split_axis_list, reduce_leveled_axes, num_reduce_axis_parts)

  if len(leveled_axes) >= 1:
    # GPU specific reorder choice
    # put the inner part as inner-most axes
    local_order = list(reduce(lambda x, y: x + y, leveled_axes[:-1], []))
    local_order += list(reduce(lambda x, y: x + y, reduce_leveled_axes, []))
    local_order += leveled_axes[-1]
  else:
    local_order += list(reduce(lambda x, y: x + y, reduce_leveled_axes, []))

  if len(local_order) > 0:
    sch[op].reorder(*ones, *pre, *local_order)
  
  return leveled_axes, reduce_leveled_axes
