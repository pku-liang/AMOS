import tvm
from tvm import topi
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from tvm.tensor_graph.nn.layers import Layer, Linear
from tvm.tensor_graph.nn.functional import dense, gemm
from tvm.tensor_graph.core import compute, GraphTensor, GraphOp, GraphNode

def internal_gate(alpha, beta1, beta2, xi, hh, activation):
    # alpha [hidden_size]
    # beta1, beta2 [hidden_size]
    # xi [batch, hidden_size]
    # hh [batch, hidden_size]
    # activation: sigmoid, tanh
    # Returns: [batch, hidden_size]
    # activation(alpha * xi * hh + (beta1 * xi) + (beta2 *hh))
    batch, hidden_size = xi.shape
    b_, h_ = hh.shape
    assert b_ == batch and h_ == hidden_size
    if activation == "sigmoid":
        def _inner_compute_gate(batch, hidden_size, alpha, beta1, beta2, xi, hh, requires_grad=True):
            return compute([batch, hidden_size],
                    lambda i, j: tvm.te.sigmoid(
                                    alpha[j] * xi[i, j] * hh[i, j] + 
                                    (beta1[j] * xi[i, j]) + 
                                    (beta2[j] * hh[i, j])
                                ),
                    name="gate_sigmoid",
                    requires_grad=requires_grad)
        return GraphOp([batch, hidden_size], [], [alpha, beta1, beta2, xi, hh], _inner_compute_gate, name="gate_sigmoid")
    elif activation == "tanh":
        def _inner_compute_gate2(batch, hidden_size, alpha, beta1, beta2, xi, hh, requires_grad=True):
            return compute([batch, hidden_size],
                    lambda i, j: tvm.te.tanh(
                                    alpha[j] * xi[i, j] * hh[i, j] + 
                                    (beta1[j] * xi[i, j]) + 
                                    (beta2[j] * hh[i, j])
                                ),
                    name="gate_tanh",
                    requires_grad=requires_grad)
        return GraphOp([batch, hidden_size], [], [alpha, beta1, beta2, xi, hh], _inner_compute_gate2, name="gate_tanh")
    else:
        print("Unsupported Activation!")
        assert True == False
    
def compute_cx(f_g, old_c, i_g, z_t):
    # cx = f_g * old_c + i_g * z_t
    batch, hidden_size = old_c.shape
    def _inner_compute_cx(batch, hidden_size, f_g, old_c, i_g, z_t, requires_grad=True):
        return compute([batch, hidden_size],
                lambda i, j: f_g[i, j] * old_c[i, j] + i_g[i, j] * z_t[i, j],
                name="compute_cx",
                requires_grad=requires_grad)
    return GraphOp([batch, hidden_size], [], [f_g, old_c, i_g, z_t], _inner_compute_cx, name="compute_cx")

def compute_hx(o_g, cx):
    # hx = o_g * tanh(cx)
    batch, hidden_size = o_g.shape
    b_, h_ = cx.shape
    assert b_ == batch and hidden_size == h_
    def _inner_compute_hx(batch, hidden_size, o_g, cx, requires_grad=True):
        return compute([batch, hidden_size],
                lambda i, j: o_g[i, j] * tvm.te.tanh(cx[i, j]),
                name="compute_hx",
                requires_grad=requires_grad)
    return GraphOp([batch, hidden_size], [], [o_g, cx], _inner_compute_hx, name="compute_hx")

class MI_LSTM(Layer):
    def __init__(self, input_size=28*28, hidden_size=1024, n_class=10, dtype="float32", out_dtype="float32"):
        super(MI_LSTM, self).__init__()
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.weight_fh = Linear(hidden_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype)
        self.weight_ih = Linear(hidden_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype)
        self.weight_zh = Linear(hidden_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype)
        self.weight_oh = Linear(hidden_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype)
        self.weight_fx = Linear(input_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype)
        self.weight_ix = Linear(input_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype)
        self.weight_zx = Linear(input_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype)
        self.weight_ox = Linear(input_size, hidden_size, bias=True, dtype=dtype, out_dtype=out_dtype)
        # alphas and betas
        self.alpha_f = GraphTensor([hidden_size], dtype=dtype, name="alpha_f", requires_grad=True)
        self.beta_f1 = GraphTensor([hidden_size], dtype=dtype, name="beta_f1", requires_grad=True)
        self.beta_f2 = GraphTensor([hidden_size], dtype=dtype, name="beta_f2", requires_grad=True)
        
        self.alpha_i = GraphTensor([hidden_size], dtype=dtype, name="alpha_i", requires_grad=True)
        self.beta_i1 = GraphTensor([hidden_size], dtype=dtype, name="beta_i1", requires_grad=True)
        self.beta_i2 = GraphTensor([hidden_size], dtype=dtype, name="beta_i2", requires_grad=True)
        
        self.alpha_o = GraphTensor([hidden_size], dtype=dtype, name="alpha_o", requires_grad=True)
        self.beta_o1 = GraphTensor([hidden_size], dtype=dtype, name="beta_o1", requires_grad=True)
        self.beta_o2 = GraphTensor([hidden_size], dtype=dtype, name="beta_o2", requires_grad=True)
        
        self.alpha_z = GraphTensor([hidden_size], dtype=dtype, name="alpha_z", requires_grad=True)
        self.beta_z1 = GraphTensor([hidden_size], dtype=dtype, name="beta_z1", requires_grad=True)
        self.beta_z2 = GraphTensor([hidden_size], dtype=dtype, name="beta_z2", requires_grad=True)

        self.weight_for_classify = GraphTensor([10, hidden_size], dtype=out_dtype, name="weight_for_classify", requires_grad=True)
    
    def forward(self, inp, old_h, old_c):
        fxi = self.weight_fx(inp)
        fhh = self.weight_fh(old_h)
        f_g = internal_gate(self.alpha_f, self.beta_f1, self.beta_f2, fxi, fhh, "sigmoid")

        ixi = self.weight_ix(inp)
        ihh = self.weight_ih(old_h)
        i_g = internal_gate(self.alpha_i, self.beta_i1, self.beta_i2, ixi, ihh, "sigmoid")

        oxi = self.weight_ox(inp)
        ohh = self.weight_oh(old_h)
        o_g = internal_gate(self.alpha_o, self.beta_o1, self.beta_o2, oxi, ohh, "sigmoid")
        
        zxi = self.weight_zx(inp)
        zhh =  self.weight_zh(old_h)
        z_t = internal_gate(self.alpha_z, self.beta_z1, self.beta_z2, zxi, zhh, "tanh")

        cx = compute_cx(f_g, old_c, i_g, z_t)
        hx = compute_hx(o_g, cx)

        result = dense(hx, self.weight_for_classify, bias=None, out_dtype=self.out_dtype)
        return result, hx, cx

def get_model():
    model = MI_LSTM()
    return model
