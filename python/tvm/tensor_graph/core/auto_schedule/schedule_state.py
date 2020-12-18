import sys
import tvm
from ..utils import ERROR


class RealScheduleState(object):
  def __init__(self, target):
    if target == "cuda":
      # bx = tvm.te.thread_axis("blockIdx.x")
      # by = tvm.te.thread_axis("blockIdx.y")
      # bz = tvm.te.thread_axis("blockIdx.z")
      # vx = tvm.te.thread_axis("vthread")
      # vy = tvm.te.thread_axis("vthread")
      # vz = tvm.te.thread_axis("vthread")
      # tx = tvm.te.thread_axis("threadIdx.x")
      # ty = tvm.te.thread_axis("threadIdx.y")
      # tz = tvm.te.thread_axis("threadIdx.z")

      self.levels = ["block", "vthread", "thread", "inner"]
      ################################
      # the merge
      self.compute_inline = False
      self.compute_at = False
      self.compute_at_op = None  # this is reserved
      self.compute_at_pos = None  # this is reserved
      self.compute_at_level = -1  # -1 root, 0 block, 1 vthread, 2 thread, 3 inner
      #################################
      # the buffer output
      self.buffer_output = False
      self.buffer_output_tensor = None
      ################################
      # the buffer input
      self.shared_cache_list = []
      self.local_cache_list = []
      #################################
      # the allreduce
      self.allreduce = False
      self.rf = None
      #################################
      # the leaf axes of spatial
      self.leaf_axes = {}
      self.leaf_axes_belong_to_op = {}  # each level of axes belong to
      for level in self.levels:
        self.leaf_axes[level] = []
        self.leaf_axes_belong_to_op[level] = None
      #################################
      # the leaf axes of reduce
      self.leaf_reduce_axes = []
      self.leaf_reduce_axes_op = None
      #################################
      # the binding
      self.binding = {}
      for level in self.levels:
        self.binding[level] = {}
        for dim in ["x", "y", "z"]:
          self.binding[level][dim] = {}
          self.binding[level][dim]["extent"] = -1
      # unroll
      self.kernel_scope = None
      self.kernel_scope_op = None
    else:
      ERROR("Target %s is not supported now." % target)
  
  def copy_binding_for_extents(self):
    ret = {}
    for level in self.levels:
      ret[level] = {}
      for dim in ["x", "y", "z"]:
        ret[level][dim] = {}
        ret[level][dim]["extent"] = self.binding[level][dim]["extent"]
    return ret