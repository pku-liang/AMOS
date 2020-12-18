import tvm 

from ..tensor import GraphTensor, GraphOp, compute
from ..abs_graph import GraphVisitor, GraphMutator
from .utils import cut_adjacent_subgraph, insert



class LayoutChangeFinder(GraphVisitor):
  def __init__(self):
    super(LayoutChangeFinder, self).__init__("up")

    self.batch_like_dim_dict = {}

  def visit_tensor(self, graph, graph_tensor):
    return

  def visit_op(self, graph, graph_op):
    sub_inputs, sub_dst = cut_adjacent_subgraph(graph_op)
    out, params = sub_dst({})
    batch_dim = tvm.tg.get_batch_like_dim(out.tvm_tensor)
    batch_dim = [x.value for x in batch_dim]

    # record the found batch dim
    if graph_op not in self.batch_like_dim_dict:
      self.batch_like_dim_dict[graph_op] = batch_dim
    else:
      self.batch_like_dim_dict[graph_op] = insert(self.batch_like_dim_dict[graph_op], batch_dim)
    batch_axis = [out.tvm_tensor.op.axis[x] for x in batch_dim]
    # get the layout transform for inputs
    for (inp1, inp2) in sub_inputs.items():
      inp_tensor = params[inp2].tvm_tensor
      inp_batch_dim = tvm.tg.find_axis_in(batch_axis, inp_tensor, out.tvm_tensor)
      # record the batch dim for input
      if inp1 not in self.batch_like_dim_dict:
        self.batch_like_dim_dict[inp1] = [x.value for x in inp_batch_dim]
      else:
        self.batch_like_dim_dict[inp1] = insert(self.batch_like_dim_dict[inp1], [x.value for x in inp_batch_dim])
    for inp in graph_op.inputs:
      self.visit(graph, inp)

  
class LayoutChangeApplier(GraphMutator):
  """LayoutChangeApplier
  Apply layout change to graph according to found batch dim and level
  -----------------------
  Args:
  -----------------------
  batch_dim_dict: dict of GraphNode: list of int
    record the batch like dim of every graph node
  level         : int
    hyperparameter of applier
    0: do not apply
    k: the biggest k dims are altered as inner-most dim
  order         : str
    asc: smallest batch inner-most
    des: biggest batch inner-most
  """
  def __init__(self, batch_dim_dict, level=0, order="des"):
    super(LayoutChangeApplier, self).__init__("up")
    self.batch_dim_dict = batch_dim_dict
    for (k, v) in batch_dim_dict.items():
      assert len(set(v)) == len(v)
    self.level = level
    self.order = order
    assert order in ["asc", "des"]

  def mutate_tensor(self, graph, graph_tensor):
    transform = graph_tensor.layout_transform \
      if graph_tensor.layout_transform is not None \
        else list(range(len(graph_tensor.shape)))
    if graph_tensor in self.batch_dim_dict:
      batch_dim = self.batch_dim_dict[graph_tensor]
      # remember to apply existing layout transform
      batch_size = [graph_tensor.shape[transform[x]] for x in batch_dim]
      if self.order == "des":
        # descending order
        # get the first k dim number
        tmp = list(sorted(zip(batch_dim, batch_size),
                    key=lambda x: x[1], reverse=True))[:self.level]
      else:
        # descending order
        # get the first k dim number
        tmp = list(sorted(zip(batch_dim, batch_size),
                    key=lambda x: x[1], reverse=False))[:self.level]
      tmp = list(list(zip(*tmp))[0]) if len(tmp) > 0 else []
      tmp_transform = []
      for i in range(len(graph_tensor.shape)):
        if i not in tmp:
          tmp_transform.append(i)
      # the biggest put at the inner-most
      tmp_transform.extend(list(reversed(tmp)))
      # remember to apply the existing transform
      new_transform = [transform[x] for x in tmp_transform]
    else:
      tmp_transform = list(range(len(graph_tensor.shape)))
      new_transform = transform

    print(graph_tensor, new_transform, flush=True)

    # no_change = new_transform == transform
    
    if (graph_tensor not in graph.inputs and graph_tensor not in graph.outputs):
      # not in inputs or outputs, we can change the layout directly
      ret = GraphTensor(graph_tensor.shape, graph_tensor.dtype,
        graph_tensor.name, graph_tensor.requires_grad)
      graph_tensor.assign_exclusive_attributes(ret)
      # update the new transform, which puts batch dim as inner-most
      ret.layout_transform = new_transform
      record = ret
    else:
      # in the inputs or outputs, we should add a transform op
      src = GraphTensor(graph_tensor.shape, graph_tensor.dtype,
        graph_tensor.name, graph_tensor.requires_grad)
      graph_tensor.assign_exclusive_attributes(src)
      
      def _identity(*args):
        assert len(args) > 1
        A = args[-1]
        shape = args[:-1]
        return compute(
          shape,
          lambda *indices: A(*indices),
          # tag="layout_trans_"+str(new_transform),
          name="alter_layout")
      
      alter = GraphOp(graph_tensor.shape, [], [src], _identity,
          name="alter_layout", requires_grad=graph_tensor.requires_grad)
      alter.layout_transform = new_transform
      # ret is different from record
      # ret will be in graph inputs/outputs
      # record is cached mutated op
      ret = src
      record = alter

    # record the mutation
    self.visited[graph_tensor] = record
    return ret

  def mutate_op(self, graph, graph_op):
    transform = graph_op.layout_transform \
      if graph_op.layout_transform is not None \
        else list(range(len(graph_op.shape)))
    
    if graph_op in self.batch_dim_dict:
      batch_dim = self.batch_dim_dict[graph_op]
      # remember to apply existing layout transform
      batch_size = [graph_op.shape[transform[x]] for x in batch_dim]
      if self.order == "des":
        # descending order
        # get the first k dim number
        tmp = list(sorted(zip(batch_dim, batch_size),
                    key=lambda x: x[1], reverse=True))[:self.level]
      else:
        # descending order
        # get the first k dim number
        tmp = list(sorted(zip(batch_dim, batch_size),
                    key=lambda x: x[1], reverse=False))[:self.level]
      tmp = list(list(zip(*tmp))[0]) if len(tmp) > 0 else []
      tmp_transform = []
      for i in range(len(graph_op.shape)):
        if i not in tmp:
          tmp_transform.append(i)
      # the biggest/smallest put at the inner-most
      tmp_transform.extend(list(reversed(tmp)))
      # remember to apply the existing transform
      new_transform = [transform[x] for x in tmp_transform]
    else:
      new_transform = transform
    
    print(graph_op, new_transform, flush=True)

    new_inputs = []
    for inp in graph_op.inputs:
      new_inputs.append(self.mutate(graph, inp))
    
    ret = GraphOp(graph_op.shape, graph_op.reduces,
                  new_inputs, graph_op.func, graph_op.name, graph_op.requires_grad)
    graph_op.assign_exclusive_attributes(ret)

    if graph_op in graph.outputs:
      # do not change the layout
      pass
    else:
      # update the new transform, which puts batch dim as inner-most
      ret.layout_transform = new_transform

    # record the mutation
    self.visited[graph_op] = ret
    return ret

  # need to re-write mutate, because the graph structure may change
  def mutate(self, graph, graph_op):
    if graph_op in self.visited:
      return self.visited[graph_op]
    
    if isinstance(graph_op, GraphTensor):
      ret = self.mutate_tensor(graph, graph_op)
    elif isinstance(graph_op, GraphOp):
      ret = self.mutate_op(graph, graph_op)

    if graph_op in graph.inputs_set:
        self.new_inputs[graph_op] = ret
    elif graph_op in graph.outputs_set:
      self.new_outputs[graph_op] = ret
    elif graph_op in graph.weights_set:
      self.new_weights[graph_op] = ret

    return self.visited[graph_op]


def apply_layout_change(fwd_graph, level=2):
  lcf = LayoutChangeFinder()
  lcf(fwd_graph)
  batch_like_dim_dict = {}
  print("check finding batch like dim:", flush=True)
  for (k, v) in lcf.batch_like_dim_dict.items():
    print("op=", k, flush=True)
    print("batch_like_dim:", list(sorted(list(set(v)))), flush=True)
    print()
    batch_like_dim_dict[k] = list(sorted(list(set(v))))
  lca = LayoutChangeApplier(batch_like_dim_dict, level=level, order="des")
  fwd_graph = lca(fwd_graph)
  return fwd_graph


# This is not used currently
class LayoutPermuter(GraphMutator):
  def __init__(self, trans_dict):
    super(LayoutPermuter, self).__init__("up")

    self.trans_dict = trans_dict

  def mutate_tensor(self, graph, graph_tensor):
    new_tensor = GraphTensor(graph_tensor.shape, graph_tensor.dtype,
      graph_tensor.name, graph_tensor.requires_grad)
    new_tensor.possible_layouts = graph_tensor.possible_layouts
    num_layouts = len(new_tensor.possible_layouts)
    if num_layouts > 0 and graph_tensor in self.trans_dict:
      new_tensor.layout_transform = new_tensor.possible_layouts[self.trans_dict[graph_tensor] % num_layouts]
    ret = new_tensor

    return ret

  def mutate_op(self, graph, graph_op):
    new_inputs = [self.mutate(graph, inp) for inp in graph_op.inputs]
    new_op = GraphOp(graph_op.shape, graph_op.reduces, new_inputs,
      graph_op.func, graph_op.name, graph_op.requires_grad)
    new_op.possible_layouts = graph_op.possible_layouts
    num_layouts = len(new_op.possible_layouts)
    if num_layouts > 0:
      new_op.layout_transform = new_op.possible_layouts[self.trans_dict[graph_op] % num_layouts]
    ret = new_op

    return ret


