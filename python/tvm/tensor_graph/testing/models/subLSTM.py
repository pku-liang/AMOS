import tvm
from tvm import topi
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from tensor_graph.nn.layers import Layer
from tensor_graph.nn.functional import dense, gemm
from tensor_graph.core import compute, GraphTensor, GraphOp, GraphNode

def internel_sublstm(input, weight_ih, bias_ih, weight_hh, bias_hh, old_h, old_c):
    '''
    input: [batch_size, input_size]
    weight_ih: [4*state_size, input_size]
    weight_hh: [4*state_size, state_size]
    bias_ih:[4*state_size]
    bias_hh:[4*state_size]
    old_h  & old_c: [batch_size, state_size]
    ------
    cell_gate: [batch_size, state_size]
    '''
    batch_size, input_size = input.shape
    batch_size_, state_size = old_h.shape
    four_stsz, input_size_ = weight_ih.shape
    four_stsz_, state_size_ = weight_hh.shape
    assert batch_size_ == batch_size and input_size == input_size_
    assert four_stsz == four_stsz_ and state_size == state_size_
    assert state_size * 4 == four_stsz

    # input[i, k1] * weight_ih[j, k1] 
    input_m_weight = gemm(input, weight_ih, transposeA=False, transposeB=True)
    # old_h[i, k2] * weight_hh[j, k2] 
    oldh_m_weight = gemm(old_h, weight_hh, transposeA=False, transposeB=True)

    def _inner_gate(batch_size, four_stsz, input_m_weight, bias_ih, oldh_m_weight, bias_hh, requires_grad=True):
        return compute([batch_size, four_stsz],
                lambda i, j: tvm.te.sigmoid
                                (
                                input_m_weight[i, j] 
                                + bias_ih[j]
                                + oldh_m_weight[i, j] 
                                + bias_hh[j]
                                ),
                name="gate_sublstm",
                tag="gate_sublstm",
                requires_grad=requires_grad)
    gates = GraphOp([batch_size, four_stsz], [], [input_m_weight, bias_ih, oldh_m_weight, bias_hh],
                _inner_gate, name="gate_sublstm")
    
    # in_gate, forget_gate, cell_gate, out_gate = topi.split(gates, 4, axis=1)
    # in_gate: gates[i, 0 to state_size]
    # forget_gate: gates[i, state_size to 2*state_size]
    # cell_gate: gates[i, 2*state_size to 3*state_size]
    # out_gate: gates[i, 3*state_size to 4*state_size]
    
    # new_c = forget_gate * old_c + cell_gate - in_gate
    # new_c = gates[i, state_size + j] * old_c[i,j] + gates[i, j + 2*state_size] - gates[i, j]
    def _inner_new_c(batch_size, state_size, gates, old_c, requires_grad=True):
        return compute([batch_size, state_size],
                lambda i, j: gates[i, state_size + j] * old_c[i,j] + gates[i, j + 2*state_size] - gates[i, j],
                name="new_c_sublstm",
                tag="new_c_sublstm",
                requires_grad=requires_grad)
    new_c = GraphOp([batch_size, state_size], [], [gates, old_c], _inner_new_c, name="new_c_sublstm")

    # new_h = topi.sigmoid(new_c) - out_gate
    # new_h = tvm.te.sigmoid(new_c[i, j]) - gates[i, j + 3*state_size]
    def _inner_new_h(batch_size, state_size, new_c, gates, requires_grad=True):
        return compute([batch_size, state_size],
                lambda i, j: tvm.te.sigmoid(new_c[i, j]) - gates[i, j + 3*state_size],
                name="new_h_sublstm",
                tag="new_h_sublstm",
                requires_grad=requires_grad)
    new_h = GraphOp([batch_size, state_size], [], [new_c, gates], _inner_new_h, name="new_h_sublstm")
    return new_h, new_c

class subLSTM(Layer):
    def __init__(self, state_size=128, input_size=28*28):
        super(subLSTM, self).__init__()
        self.weight_ih = GraphTensor([4*state_size, input_size], name="weight_ih", requires_grad=True)
        self.bias_ih = GraphTensor([4*state_size], name="bias_ih", requires_grad=True)
        
        self.weight_hh = GraphTensor([4*state_size, state_size], name="weight_hh", requires_grad=True)
        self.bias_hh = GraphTensor([4*state_size], name="bias_hh", requires_grad=True)
        
        self.weight_for_classify = GraphTensor([10, state_size], name="weight_for_classify", requires_grad=True)
        self.bias_for_classify = GraphTensor([10], name="bias_for_classify", requires_grad=True)
    
    def forward(self, x, old_h, old_c):
        new_h, new_c = internel_sublstm(x, self.weight_ih, self.bias_ih, self.weight_hh, self.bias_hh, old_h, old_c)
        result = dense(new_h, self.weight_for_classify, self.bias_for_classify)
        return result, new_h, new_c

def get_model():
    model = subLSTM()
    return model
