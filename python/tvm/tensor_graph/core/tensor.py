import tvm
from .context import Context


class LayoutContext(Context):
  stack = []
  def __init__(self, reorder):
    self.reorder = reorder

  def __enter__(self):
    self.stack.append(self)

  def __exit__(self, type, value, trace):
    self.stack.pop()

  @classmethod
  def empty(cls):
    return len(cls.stack) == 0


##################################################
# tensors
class NamedDimTensor(object):
  def __init__(self, tvm_tensor, op=None, dim_order=None):
    assert isinstance(tvm_tensor, tvm.te.tensor.Tensor)
    if dim_order is not None:
      assert isinstance(dim_order, (list, tuple))
    self.tvm_tensor = tvm_tensor
    self.op = op
    self.dim_order = dim_order
    self.shape = tvm_tensor.shape

  def __getitem__(self, indices):
    if not isinstance(indices, (list, tuple)):
      indices = [indices]
    if self.dim_order is not None:
      num_index = len(indices)
      assert num_index == len(self.dim_order)
      real_indices = [indices[self.dim_order[i]] for i in range(num_index)]
    else:
      real_indices = indices
    if not isinstance(real_indices, (list, tuple)):
      real_indices = [real_indices]
    return self.tvm_tensor(*real_indices)

  def __call__(self, *indices):
    return self.__getitem__(indices)

  def __repr__(self):
    return "NamedDimTensor("+str(self.tvm_tensor)+", dim_order="+str(self.dim_order)+")"

  def __str__(self):
    return "NamedDimTensor("+str(self.tvm_tensor)+", dim_order="+str(self.dim_order)+")"


##################################################
# basic functions
def compute(shape, fcompute, name="compute", tag="", attrs=None, requires_grad=True):
  if tag == "":
    tag = "TG_AUTOGEN"
  if not LayoutContext.empty():
    reorder = LayoutContext.stack[-1].reorder
  else:
    reorder = None
  tvm_tensor = tvm.te.compute(
      shape, fcompute, name=name, tag=tag, attrs=attrs, requires_grad=requires_grad, reorder=reorder)
  assert isinstance(tvm_tensor, tvm.te.tensor.Tensor), "only support output one tensor now."
  return NamedDimTensor(tvm_tensor, dim_order=reorder)


##################################################3
# Graphs
class GraphNode(object):
  def __init__(self):
    self.children = []


class GraphTensor(GraphNode):
  def __init__(self, shape, dtype="float32", name="tensor", requires_grad=True):
    super(GraphTensor, self).__init__()
    self.shape = shape
    self.dtype = dtype
    self.name = name
    self.requires_grad = requires_grad

    self.layout_transform = None
    self.possible_layouts = []

  def assign_exclusive_attributes(self, another):
    assert isinstance(another, GraphNode), \
      "cant assign attributes to type " + str(type(another))
    another.layout_transform = self.layout_transform
    another.possible_layouts = self.possible_layouts

  def __call__(self, params):
    if self in params:
      return params[self], params
    
    if self.layout_transform is not None:
      assert isinstance(self.layout_transform, (list, tuple))
      assert len(self.layout_transform) == len(self.shape)
      ipl_shape = [self.shape[i] for i in self.layout_transform]
      tensor = tvm.te.placeholder(ipl_shape, self.dtype, self.name, requires_grad=self.requires_grad)
      named_tensor = NamedDimTensor(tensor, op=self, dim_order=self.layout_transform)
    else:
      tensor = tvm.te.placeholder(self.shape, self.dtype, self.name, requires_grad=self.requires_grad)
      named_tensor = NamedDimTensor(tensor, op=self)
    return named_tensor, {self: named_tensor, **params}

  def __repr__(self):
    return "GraphTensor("+str(self.shape)+", dtype="+self.dtype+", name=" \
      +self.name+", grad="+str(self.requires_grad)+")"

  def __str__(self):
    return self.__repr__()
    # return ("GraphTensor("+str(self.shape)+", dtype="+self.dtype+", name="
    #         +self.name+", trans="+str(self.layout_transform)+")")


class GraphOp(GraphNode):
  def __init__(self, shape, reduces, inputs, func, name="operator", requires_grad=True):
    super(GraphOp, self).__init__()
    assert isinstance(shape, (list, tuple))
    assert isinstance(reduces, (list, tuple))
    assert isinstance(inputs, (list, tuple))
    assert callable(func)

    self.shape = shape
    self.reduces = reduces
    self.inputs = inputs
    self.func = func
    self.name = name
    self.requires_grad = requires_grad

    self.layout_transform = None
    self.possible_layouts = []
    dtype = None
    for inp in self.inputs:
      if dtype is None:
        dtype = inp.dtype
      else:
        assert dtype == inp.dtype, "Find different dtype in inputs"
      inp.children.append(self)
    self.dtype = dtype
  
  def assign_exclusive_attributes(self, another):
    assert isinstance(another, GraphNode), \
      "cant assign attributes to type " + str(type(another))
    another.layout_transform = self.layout_transform
    another.possible_layouts = self.possible_layouts

  def __call__(self, params):
    if self in params:
      return params[self], params
    args = []
    updated_params = {**params}
    for inp in self.inputs:
      tmp_tensor, tmp_params = inp(updated_params)
      args.append(tmp_tensor)
      updated_params.update(tmp_params)
    
    if self.layout_transform is not None:
      with LayoutContext(self.layout_transform):
        # ipl_shape = [self.shape[i] for i in self.layout_transform]
        # tensor = self.func(*ipl_shape, *self.reduces, *args)
        try:
          tensor = self.func(*self.shape, *self.reduces, *args, name=self.name)
        except TypeError:
          tensor = self.func(*self.shape, *self.reduces, *args)
        tensor.op = self
        tensor.dim_order = self.layout_transform
        is_direct_source = True
        for nt in args:
          t = nt.tvm_tensor
          if t not in tensor.tvm_tensor.op.input_tensors:
            is_direct_source = False
            break
        assert is_direct_source, "For GraphOp, the given function should \
                                  make use of all its arguments \
                                  and only contains one compute operation"
        tensor.op = self
        named_tensor = tensor
    else:
      try:
        tensor = self.func(*self.shape, *self.reduces, *args, requires_grad=self.requires_grad, name=self.name)
      except TypeError:
        tensor = self.func(*self.shape, *self.reduces, *args, requires_grad=self.requires_grad)
      tensor.op = self
      named_tensor = tensor
    
    updated_params.update({self: named_tensor})
    return named_tensor, updated_params
  
  def __repr__(self):
    return "GraphOp("+str(self.shape)+", reduces="+str(self.reduces) \
      +", name="+self.name+", grad="+str(self.requires_grad)+")"

  def __str__(self):
    return self.__repr__()
    # return ("GraphOp("+str(self.shape)+", reduces="+
    #         str(self.reduces)+", inputs="+str(self.inputs)+
    #         ", func="+str(self.func)+", name="+self.name+")")