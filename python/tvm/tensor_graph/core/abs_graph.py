import tvm

from .tensor import GraphTensor, GraphOp, NamedDimTensor, compute
from .utils import flatten_forward_graph


class GraphVisitor(object):
  def __init__(self, mode="up"):
    assert mode in ["up"]
    self.mode = mode
    self.visited = set()

  def visit_tensor(self, graph, graph_tensor):
    if self.mode == "up":
      return
    else:
      raise ValueError("Unsupported graph visitor mode: %s" % self.mode)

  def visit_op(self, graph, graph_op):
    if self.mode == "up":
      for inp in graph_op.inputs:
        self.visit(graph, inp)
    else:
      raise ValueError("Unsupported graph visitor mode: %s" % self.mode)

  def visit(self, graph, graph_op):
    if graph_op in self.visited:
      return

    if isinstance(graph_op, GraphTensor):
      self.visit_tensor(graph, graph_op)
    elif isinstance(graph_op, GraphOp):
      self.visit_op(graph, graph_op)
    else:
      raise ValueError("Unknown type:", type(graph_op))

    self.visited.add(graph_op)
    return

  def __call__(self, graph):
    if self.mode == "up":
      for out in graph.outputs:
        self.visit(graph, out)
    else:
      raise ValueError("Unsupported graph visitor mode: %s" % self.mode)


class GraphMutator(object):
  def __init__(self, mode="up"):
    assert mode in ["up"]
    self.mode = mode

    self.new_inputs = {}
    self.new_outputs = {}
    self.new_weights = {}

    self.visited = {}

  def mutate_tensor(self, graph, graph_tensor):
    if self.mode == "up":
      ret = GraphTensor(graph_tensor.shape, graph_tensor.dtype,
        graph_tensor.name, graph_tensor.requires_grad)
    # elif self.mode == "down":
    #   new_children = [self.mutate(graph, cld) for cld in graph_tensor.children]
    #   new_op = GraphTensor(graph_tensor.shape, graph_tensor.dtype, graph_tensor.name)
    #   new_op.children = new_children
    #   ret = new_op
    # elif self.mode == "bi":
    #   new_children = [self.passing(cld) for cld in graph_tensor.children]
    #   new_op = GraphTensor(graph_tensor.shape, graph_tensor.dtype, graph_tensor.name)
    #   new_op.children = new_children
    #   return new_op
    else:
      raise ValueError("Unsupported graph mutator mode: %s" % self.mode)

    return ret

  def mutate_op(self, graph, graph_op):
    if self.mode == "up":
      new_inputs = [self.mutate(graph, inp) for inp in graph_op.inputs]
      ret = GraphOp(graph_op.shape, graph_op.reduces, new_inputs,
        graph_op.func, graph_op.name, graph_op.requires_grad)
    # elif self.mode == "down":
    #   new_children = [self.mutate(graph, cld) for cld in graph_op.children]
    #   new_op = GraphOp(graph_op.shape, graph_op.reduces, graph_op.inputs, graph_op.func, graph_op.name)
    #   new_op.children = new_children
    #   ret = new_op
    # elif self.mode == "bi":
    #   new_inputs = [self.passing(inp) for inp in graph_op.inputs]
    #   new_children = [self.passing(inp) for inp in graph_op.inputs]
    #   new_op = GraphOp(graph_op.shape, graph_op.reduces, new_inputs, graph_op.func, graph_op.name)
    #   new_op.children = new_children
    #   return new_op
    else:
      raise ValueError("Unsupported graph mutator mode: %s" % self.mode)

    return ret

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

    self.visited[graph_op] = ret
    return ret

  # def passing(self, graph_op):
  #   return graph_op

  def clear(self):
    for state in [self.new_inputs, self.new_outputs, self.new_weights, self.visited]:
      state.clear()

  def __call__(self, graph):
    self.clear()
    if self.mode == "up":
      for out in graph.outputs:
        self.mutate(graph, out)
    # elif self.mode == "down":
    #   for inp in graph.inputs:
    #     self.mutate(graph, inp)
    else:
      raise ValueError("Unsupported graph mutator mode: %s" % self.mode)

    new_inputs = [self.new_inputs[x] for x in graph.inputs]
    new_outputs = [self.new_outputs[x] for x in graph.outputs]
    new_weights = [self.new_weights[x] for x in graph.weights]
    return graph.make_new(new_inputs, new_outputs, new_weights)


class Graph(object):
  def __init__(self, inputs, outputs, weights):
    self.inputs = inputs
    self.outputs = outputs
    self.weights = weights
    # this is for subgraph
    self.wire = {}

    self.inputs_set = set(inputs)
    self.outputs_set = set(outputs)
    self.weights_set = set(weights)

  @classmethod
  def make_new(cls, inputs, outputs, weights):
    return cls(inputs, outputs, weights)


class BackwardGraph(Graph):
  def __init__(self, inputs, labels, outputs, weights, loss, gradients, lr, updates):
    super(BackwardGraph, self).__init__(inputs, outputs, weights)
    self.labels = labels
    self.loss = loss
    self.gradients = gradients
    self.lr = lr
    self.updates = updates

  # def create_schedule(self, need_outputs=False, need_loss=False, need_gradients=False):
  #   tensors = self.inputs + self.labels + self.weights + [self.lr]
  #   if need_outputs:
  #     tensors += self.outputs
  #   if need_loss:
  #     tensors.append(self.loss)
  #   if need_gradients:
  #     tensors += self.gradients
  #   tensors += self.updates
  #   ops = [x.tvm_tensor.op for x in tensors]
  #   return tvm.te.create_schedule(ops), [x.tvm_tensor for x in tensors]

  # def build(self, sch, bufs, target):
  #   return tvm.build(sch, bufs, target)


class ForwardGraph(Graph):
  def __init__(self, inputs, outputs, weights):
    super(ForwardGraph, self).__init__(inputs, outputs, weights)
    op_list, down_graph = flatten_forward_graph(outputs)
    self.op_list = op_list
    self.down_graph = down_graph

  def __call__(self):
    params = {}
    output_tensors = []
    for out in self.outputs:
      out_tensor, params = out(params)
      output_tensors.append(out_tensor)
    
    # get inputs
    input_tensors = [params[x] for x in self.inputs]

    # get weights
    weight_tensors = []
    # relax for redundant weight definition
    for w in self.weights:
      if w in params:
        weight_tensors.append(params[w])

    return input_tensors, output_tensors, weight_tensors

  def make_backward(self, loss_engine, optimize_engine):
    params = {}
    output_tensors = []
    for out in self.outputs:
      out_tensor, params = out(params)
      output_tensors.append(out_tensor)

    # get loss
    loss_tensor, label_tensors = loss_engine(output_tensors)
    if isinstance(loss_tensor, NamedDimTensor):
      loss_tvm_tensor = loss_tensor.tvm_tensor
    # do not allow tvm tensor, to provide a unique interface
    # elif isinstance(loss_tensor, tvm.te.tensor.Tensor):
    #   loss_tvm_tensor = loss_tensor
    else:
      raise RuntimeError(
        "Expect loss engine returns loss of type NamedDimTensor \
        but get type %s" % (str(type(loss_tensor))))

    # get inputs
    input_tensors = [params[x] for x in self.inputs]
    
    # get weights
    # relax for redundant weight definition
    weight_tensors = []
    for w in self.weights:
      if w in params:
        weight_tensors.append(params[w])

    weight_tvm_tensors = [x.tvm_tensor for x in weight_tensors]
    # get gradients
    gradient_tvm_tensors = tvm.tg.gradient(loss_tvm_tensor, weight_tvm_tensors)
    # wrap as named dim tensor
    # this is just to force a unique interface
    weight_tensors = [NamedDimTensor(x.tvm_tensor, x.op, None) for x in weight_tensors]
    gradient_tensors = [NamedDimTensor(x, None, None) for x in gradient_tvm_tensors]

    if not isinstance(optimize_engine.lr_tensor, NamedDimTensor):
      lr_tensor = NamedDimTensor(optimize_engine.lr_tensor, None, None)
    else:
      lr_tensor = optimize_engine.lr_tensor

    update_tensors = optimize_engine(weight_tensors, gradient_tensors)

    return BackwardGraph(input_tensors, label_tensors, output_tensors, weight_tensors,
      loss_tensor, gradient_tensors, lr_tensor, update_tensors)


def make_fwd_graph(model, inputs):
  outputs = model(*inputs)
  if not isinstance(outputs, (list, tuple)):
    outputs = [outputs]
  # get the weights tensors
  weights_tensors = []
  for w in model.weights():
    weights_tensors.append(w)
  
  return ForwardGraph(inputs, outputs, weights_tensors)