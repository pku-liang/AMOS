import tvm
from tvm import topi
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from tvm.tensor_graph.nn.layers import Layer
from tvm.tensor_graph.nn.functional import dense
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, GraphNode

def elu(inputs):
    '''
    ELU(x)=max(0,x)+min(0,(exp(x)−1))
    '''
    def _inner_elu(dim0, dim1, inputs, requires_grad=True):
      return compute([dim0, dim1],
              lambda i, j: tvm.te.if_then_else(inputs[i, j] > 0, inputs[i, j], tvm.tir.exp(inputs[i, j]) - 1),
              name="elu",
              tag="elu",
              requires_grad=requires_grad)
    return GraphOp(inputs.shape, [], inputs, _inner_elu, name="elu")

def internel_lltm(input, weight_for_gate, bias_for_gate, old_h, old_c):
    '''
    input: [batch_size, 28*28]
    old_h  & old_c: [batch_size, state_size]
    >>>>> cat -> X: [batch_size, state_size+28*28]
    weight_for_gate: [3*state_size, state_size+28*28]
    bias_for_gate:[3*state_size]
    '''
    batch_size, input_size = input.shape
    batch_size_, state_size = old_h.shape
    three_stsz, cat_size = weight_for_gate.shape
    two_stsz = state_size * 2
    assert batch_size == batch_size_
    cat_size_ = input_size + state_size
    assert cat_size == cat_size_
    # X = topi.concatenate([old_h, input], axis=1)
    def _inner_cat(batch_size, cat_size, old_h, input, requires_grad=True):
      return compute([batch_size, cat_size],
              lambda i, j: tvm.te.if_then_else(j < input_size, input[i, j], old_h[i, j-input_size]),
              name="cat",
              tag="cat",
              requires_grad=requires_grad)
    X = GraphOp([batch_size, cat_size], [], [old_h, input], _inner_cat, name="cat")
    gates = dense(X, weight_for_gate, bias_for_gate)
    # gate_weights = topi.nn.dense(X, weight_for_gate, bias_for_gate)
    # gates = topi.split(gate_weights, 3, axis=1)
    # gates[0]: [batch_size, 0 to state_size] -> input_gate
    # gates[1]: [batch_size, state_size to 2*state_size] -> output_gate
    # gates[2]: [batch_size, 2*state_size to 3*state_size] -> candidate_cell
    # input_gate = topi.sigmoid(gates[0])
    # output_gate = topi.sigmoid(gates[1])
    # candidate_cell = elu(gates[2])
    def _inner_sse(batch_size, three_stsz, gates, requires_grad=True):
      return compute([batch_size, three_stsz],
              lambda i, j: tvm.te.if_then_else(
                            j < two_stsz, 
                            tvm.te.sigmoid(gates[i, j]), 
                            tvm.te.if_then_else(
                              gates[i, j] > 0, 
                              gates[i, j], 
                              tvm.tir.exp(gates[i, j]) - 1
                              )
                            ),
              name="sse",
              tag="sse",
              requires_grad=requires_grad)
    gates_act = GraphOp([batch_size, three_stsz], [], [gates], _inner_sse, name="sse")
    
    # new_c = topi.add(old_c, topi.multiply(candidate_cell, input_gate))
    def _inner_new_c(batch_size, state_size, gates_act, old_c, requires_grad=True):
      return compute([batch_size, state_size],
              lambda i, j: old_c[i, j] + gates_act[i, j + two_stsz] * gates_act[i, j],
              name="new_c_lltm",
              tag="new_c_lltm",
              requires_grad=requires_grad)
    new_c = GraphOp([batch_size, state_size], [], [gates_act, old_c], _inner_new_c, name="new_c_lltm")

    # new_h = topi.multiply(topi.tanh(new_c), output_gate)
    def _inner_new_h(batch_size, state_size, new_c, gates_act, requires_grad=True):
      return compute([batch_size, state_size],
              lambda i, j: tvm.te.tanh(new_c[i, j]) * gates_act[i, j + state_size],
              name="new_h_lltm",
              tag="new_h_lltm",
              requires_grad=requires_grad)
    new_h = GraphOp([batch_size, state_size], [], [new_c, gates_act], _inner_new_h, name="new_h_lltm")
    
    return new_h, new_c

class LLTM(Layer):
    def __init__(self, state_size=128, input_size=28*28, dtype="float32"):
        super(LLTM, self).__init__()
        self.weight_for_gate = GraphTensor([3*state_size, state_size+input_size], dtype=dtype, name="weight_for_gate", requires_grad=True)
        self.bias_for_gate = GraphTensor([3*state_size], name="bias_for_gate", dtype=dtype, requires_grad=True)
        
        self.weight_for_classify = GraphTensor([10, state_size], dtype=dtype, name="weight_for_classify", requires_grad=True)
        self.bias_for_classify = GraphTensor([10], dtype=dtype, name="bias_for_classify", requires_grad=True)
    
    def forward(self, x, old_h, old_c):
        new_h, new_c = internel_lltm(x, self.weight_for_gate, self.bias_for_gate, old_h, old_c)
        result = dense(new_h, self.weight_for_classify, self.bias_for_classify)
        return result, new_h, new_c

def get_model():
    model = LLTM()
    return model