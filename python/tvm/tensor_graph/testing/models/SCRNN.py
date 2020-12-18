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

def internel_SCRNN(inputs, B, U, V, fc_weight, state_h, state_c, alpha = 0.5):
    '''
    # state_h : [batch, 128=num_units]
    # state_c : [batch, 64=context_units]
    # inputs: [batch, 28*28=input_size]
    # B : [28*28, 64]
    # (inputs @ B) : [batch, 64]
    # context_state : [batch, 64] -> next state_c
    
    # concated: [batch, 64+28*28+128]
    # FC layer 64+28*28+128 -> 128
    # hidden_state : [batch, 128]
    # U: [128, 128]
    # V: [64, 128]
    # new_h: [batch, 128] -> next state_h
    '''
    batch, input_size = inputs.shape
    _, num_units = state_h.shape
    __, context_units = state_c.shape
    # context_state = (1 - alpha) * (inputs @ self.B) + alpha * state_c
    input_at_B = gemm(inputs, B, transposeA=False, transposeB=False)
    def _inner_state_c(batch, context_units, input_at_B, state_c, requires_grad=True):
        return compute([batch, context_units],
                lambda i, j: (1-alpha) * input_at_B[i, j] + alpha * state_c[i, j],
                name="state_c",
                requires_grad=requires_grad)
    context_state = GraphOp([batch, context_units], [], [input_at_B, state_c],
                        _inner_state_c, name="context_state")
    
    # concated = torch.cat([context_state, inputs, state_h], dim=1)
    def _inner_concated(batch, cat_dim, context_state, inputs, state_h, requires_grad=True):
        return compute([batch, cat_dim],
                lambda i, j: tvm.te.if_then_else(j < context_units, 
                                    context_state[i, j],
                                tvm.te.if_then_else(j < context_units+input_size, 
                                        inputs[i, j-context_units],
                                            state_h[i, j-context_units-input_size])),
                name="concated",
                requires_grad=requires_grad)
    concated = GraphOp([batch, context_units+input_size+num_units], [], [context_state, inputs, state_h],
                _inner_concated, name="concated")
    # concated: [batch, 64+28*28+128], FC layer 64+28*28+128 -> 128
    # hidden_state = torch.sigmoid(self.fc(concated))
    fc_ed = gemm(concated, fc_weight, transposeA=False, transposeB=False)
    def _inner_hidden_state(batch, num_units, fc_ed, requires_grad=True):
        return compute([batch, num_units], 
                lambda i, j: tvm.te.sigmoid(fc_ed[i, j]),
                name="sigmoid",
                requires_grad=requires_grad)
    hidden_state = GraphOp([batch, num_units], [], [fc_ed], _inner_hidden_state, name="sigmoid")
    # new_h = hidden_state @ self.U + context_state @ self.V
    h_at_U = gemm(hidden_state, U, transposeA=False, transposeB=False)
    c_at_V = gemm(context_state, V, transposeA=False, transposeB=False)
    def _inner_new_h(batch, num_units, h_at_U, c_at_V, requires_grad=True):
        return compute([batch, num_units],
                lambda i, j: h_at_U[i, j] + c_at_V[i, j],
                name="new_h",
                requires_grad=requires_grad)
    new_h = GraphOp([batch, num_units], [], [h_at_U, c_at_V], _inner_new_h, name="new_h")

    return new_h, context_state


class SCRNN(Layer):
    def __init__(self, num_units=128,context_units=64, input_size=28*28):
        super(SCRNN, self).__init__()
        # B : [28*28, 64]
        # U: [128, 128]
        # V: [64, 128]
        # FC layer 64+28*28+128 -> 128
        self.B = GraphTensor([input_size, context_units], name="B", requires_grad=True)
        self.U = GraphTensor([num_units, num_units], name="U", requires_grad=True)
        self.V = GraphTensor([context_units, num_units], name="V", requires_grad=True)
        self.fc_weight = GraphTensor([num_units+context_units+input_size, num_units], name="fc_weight", requires_grad=True)

        self.weight_for_classify = GraphTensor([10, num_units], name="weight_for_classify", requires_grad=True)
    
    def forward(self, x, old_h, old_c):
        # state_h : [batch, 128=num_units]
        # state_c : [batch, 64=context_units]
        new_h, new_c = internel_SCRNN(x, self.B, self.U, self.V, self.fc_weight, old_h, old_c)
        result = dense(new_h, self.weight_for_classify, bias=None)
        return result, new_h, new_c

def get_model():
    model = SCRNN()
    return model
