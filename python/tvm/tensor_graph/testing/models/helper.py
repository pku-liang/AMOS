from tensor_graph.core import compute, GraphTensor, GraphOp, GraphNode
import tvm
from tvm import topi

def norm(inputs, verify_num_caps):
    # [20, 32 * 6 * 6, 8] -> [20, 32 * 6 * 6]
    batch_size, vector_dim, num_capsules = inputs.shape
    # assert batch_size == 20 
    assert vector_dim == 32*6*6 and num_capsules == verify_num_caps
    def _inner_norm(batch_size, vector_dim, num_capsules, inputs, requires_grad=True):
        r = tvm.te.reduce_axis([0, num_capsules])
        return compute([batch_size, vector_dim],
                lambda i, j: tvm.te.sum(tvm.tir.power(inputs[i, j, r], 2.0), axis=[r]),
                name="inner_norm",
                tag="inner_norm",
                requires_grad=requires_grad)
    inner_norm = GraphOp([batch_size, vector_dim], [num_capsules], [inputs], _inner_norm,
                name="inner_norm")
    
    return inner_norm
        
# Deprecated : integrated into weight_squash
# def scaling(inputs):
#     # [20, 32*6*6]
#     batch_size, vector_dim = inputs.shape
#     assert batch_size == 20 and vector_dim == 32 * 6 * 6
#     def _inner_scaling(batch_size, vector_dim, inputs, requires_grad=True):
#         return compute([batch_size, vector_dim],
#                 lambda i, j: inputs[i, j] / (inputs[i, j] + 1),
#                 name="inner_scaling",
#                 tag="inner_scaling",
#                 requires_grad=requires_grad)
#     scale_op = GraphOp([batch_size, vector_dim], [], [inputs], _inner_scaling, name="inner_scaling")
#     return scale_op


def weight_squash(input_tensor, squared_norm, verify_num_caps):
    batch_size, vector_dim, num_capsules = input_tensor.shape
    batch_size_, vector_dim_ = squared_norm.shape
    assert batch_size == batch_size_  and vector_dim == vector_dim_
    assert num_capsules == verify_num_caps and vector_dim == 32*6*6
    def _inner_weight_squash(batch_size, vector_dim, num_capsules, input_tensor, squared_norm, requires_grad=True):
        return compute([batch_size, vector_dim, num_capsules],
                lambda i, j, k: 
                    input_tensor[i, j, k] * squared_norm[i, j] / (squared_norm[i, j] + 1)
                        / tvm.te.sqrt(squared_norm[i, j]),
                name="inner_weight_squash",
                tag="inner_weight_squash",
                requires_grad=requires_grad)
    return GraphOp([batch_size, vector_dim, num_capsules], [], [input_tensor, squared_norm], 
            _inner_weight_squash, name="inner_weight_squash")

def cu_multiply(c_ij, u_hat):
    # c_ij torch.Size([10, 20, 1152, 16])
    # u_hat torch.Size([10, 20, 1152, 16])
    # return  [10, 20, 16]
    ten, twenty, n1152, n16 = c_ij.shape
    assert u_hat.shape == c_ij.shape
    assert ten == 10 and n1152 == 1152  and n16 == 16 # and twenty == 20 
    def _inner_cu_multiply(ten, twenty, n16, n1152, c_ij, u_hat, requires_grad=True):
        r = tvm.te.reduce_axis([0, n1152])
        return compute([ten, twenty, n16],
                lambda i, j, n: tvm.te.sum(c_ij[i,j,r,n] * u_hat[i,j,r,n], axis=[r]),
                name="cu_multiply",
                tag="cu_multiply",
                requires_grad=requires_grad)
    return GraphOp([ten, twenty, n16], [n1152], [c_ij, u_hat], _inner_cu_multiply, name="cu_multiply")



def norm2(inputs):
    # inputs: [10, 20, 16]
    # output: [10, 20]
    dim1, dim2, dim3  = inputs.shape
    assert dim1 == 10 and dim3 == 16 # and dim2 == 20
    def _inner_norm(dim1, dim2, dim3, inputs, requires_grad=True):
        r = tvm.te.reduce_axis([0, dim3])
        return compute([dim1, dim2],
                lambda i, j: tvm.te.sum(tvm.tir.power(inputs[i, j, r], 2.0), axis=[r]),
                name="inner_norm2",
                tag="inner_norm2",
                requires_grad=requires_grad)
    return GraphOp([dim1, dim2], [dim3], [inputs], _inner_norm, name="inner_norm2")
        
        
def scaling2(inputs):
    # inputs: [10, 20]
    # output: [10, 20]
    dim1, dim2 = inputs.shape
    assert dim1 == 10 # and dim2 == 20
    def _inner_scaling(dim1, dim2, inputs, requires_grad=True):
        return compute([dim1, dim2],
                lambda i, j: inputs[i, j] / (inputs[i, j] + 1),
                name="inner_scaling2",
                tag="inner_scaling2",
                requires_grad=requires_grad)
    scale_op = GraphOp([dim1, dim2], [], [inputs], _inner_scaling, name="inner_scaling2")
    return scale_op


def weight_squash2(scale, input_tensor, squared_norm):
    # scale: [10, 20]
    # input_tensor & return: [10, 20, 16]
    # squared_norm: [10, 20]
    dim1, dim2, dim3 = input_tensor.shape
    assert dim1 == 10 and dim3 == 16 # and dim2 == 20 
    assert scale.shape == squared_norm.shape
    def _inner_weight_squash(dim1, dim2, dim3, scale, input_tensor, squared_norm, requires_grad=True):
        return compute([dim1, dim2, dim3],
                lambda i, j, k: scale[i,j] * input_tensor[i,j,k] / tvm.te.sqrt(squared_norm[i,j]),
                name="inner_weight_squash",
                tag="inner_weight_squash",
                requires_grad=requires_grad)
    return GraphOp([dim1, dim2, dim3], [], [scale, input_tensor, squared_norm], 
            _inner_weight_squash, name="inner_weight_squash")


def squash2(s_j):
    # inputs & output: [10, 20, 16]
    ten, twenty, n16 = s_j.shape
    # assert ten == 10 and twenty == 20 and n16 == 16
    squared_norm = norm2(s_j)
    # squared_norm: [10, 20]
    scale = scaling2(squared_norm)
    # scale: [10, 20]
    out_squash = weight_squash2(scale, s_j, squared_norm)
    # out_squash: [10, 20, 16]
    # ten_, twenty_, n16_ = out_squash.shape
    # assert ten_ == 10 and twenty_ == 20 and n16_ == 16
    return out_squash

def uv_dot(u_hat, v_j):
    # u_hat torch.Size([10, 20, 1152, 16])
    # v_j torch.Size([10, 20, 16])
    # a_ij = (u_hat * v_j).sum(dim=-1, keepdim=False!)
    # -> a_ij torch.Size([10, 20, 1152])
    dim1, dim2, dim3, dim4 = u_hat.shape
    assert dim1 == 10 and dim3 == 1152  and dim4 == 16 # and dim2 == 20 
    dim1_, dim2_, dim4_ = v_j.shape
    assert dim1 == dim1_ and  dim2 == dim2_ and dim4 == dim4_
    def _inner_uv_dot(dim1, dim2, dim3, dim4, u_hat, v_j, requires_grad=True):
        r = tvm.te.reduce_axis([0, dim4])
        return compute([dim1, dim2, dim3],
                lambda i, j, k: tvm.te.sum(u_hat[i,j,k,r]*v_j[i,j,r],axis=[r]),
                name="uv_dot",
                tag="uv_dot",
                requires_grad=requires_grad)
    return GraphOp([dim1, dim2, dim3], [dim4], [u_hat, v_j], _inner_uv_dot, name="uv_dot")

def update_by_aij(b_ij, a_ij, s):
    # a_ij torch.Size([10, 20, 1152])
    # b_ij, [10, 20, 1152, 16]
    # b_ij = b_ij + a_ij
    dim1, dim2, dim3, dim4 = b_ij.shape
    assert dim1 == 10 and dim3 == 1152 and dim4 == 16 # and dim2 == 20 
    dim1_, dim2_, dim3_ = a_ij.shape
    assert dim1 == dim1_ and dim2 == dim2_ and dim3 == dim3_
    def _inner_update_by_aij(dim1, dim2, dim3, dim4, b_ij, a_ij, requires_grad=True):
        return compute([dim1, dim2, dim3, dim4],
                lambda i, j, k, n: b_ij[i,j,k,n]+a_ij[i,j,k],
                name="update_by_aij" + s,
                tag="update_by_aij",
                requires_grad=requires_grad)
    return GraphOp(b_ij.shape, [], [b_ij, a_ij], _inner_update_by_aij, name="update_by_aij" + s)

def uW_multiply(u, W, verify_num_caps):
    # u torch.Size([20j, 1152k, 8r])
    # W torch.Size([10i, 1152k, 8r, 16n])
    # -> [10i, 20j, 1152k, 16n]
    u1, u2, u3 = u.shape
    W1, W2, W3, W4 = W.shape
    assert u2 == 1152 and u3 == verify_num_caps # and u1 == 20 
    assert W1 == 10 and W2 == 1152 and W3 == verify_num_caps and W4 == 16
    def _inner_uW_multiply(dim1, dim2, dim3, dim4, redim, u, W, requires_grad=True):
        r = tvm.te.reduce_axis([0, redim])
        return compute([dim1, dim2, dim3, dim4],
                lambda i, j, k, n: tvm.te.sum(W[i,k,r,n]*u[j,k,r], axis=[r]),
                name="uW_multiply",
                tag="uW_multiply",
                requires_grad=requires_grad)
    return GraphOp([W1, u1, u2, W4], [u3], [u, W], _inner_uW_multiply, name="uW_multiply")

def softmax_2(b_ij):
    # c_ij torch.Size([10, 20, 1152, 16])
    # F.softmax(b_ij, dim=2) | 1152-dim
    dim0, dim1, dim2, dim3 = b_ij.shape
    assert dim0 == 10 and dim2 == 1152 and dim3 == 16 #and dim1 == 20
    def _inner_maxtrick(dim0, dim1, dim3, dim2, b_ij, requires_grad=True):
        r = tvm.te.reduce_axis([0, dim2])
        return compute([dim0, dim1, dim3],
                lambda i, j, m: tvm.te.max(b_ij[i,j,r,m], axis=[r]),
                name="maxtrick",
                tag="maxtrick",
                requires_grad=requires_grad)
    # maxval [10, 20, 16]
    maxval = GraphOp([dim0, dim1, dim3], [dim2], [b_ij], _inner_maxtrick, name="maxtrick")
    def _inner_exp(dim0, dim1, dim2, dim3, b_ij, maxval, requires_grad=True):
        return compute([dim0, dim1, dim2, dim3],
                lambda i, j, k, m: tvm.tir.exp(b_ij[i,j,k,m] - maxval[i,j,m]),
                name="softmax_exp",
                tag="softmax_exp",
                requires_grad=requires_grad)
    # exp [10, 20, 1152, 16]
    exp = GraphOp([dim0, dim1, dim2, dim3], [], [b_ij, maxval], _inner_exp, name="softmax_exp")

    def _inner_sum(dim0, dim1, dim3, dim2, exp, requires_grad=True):
        r = tvm.te.reduce_axis([0, dim2])
        return compute([dim0, dim1, dim3],
                lambda i, j, m: tvm.te.sum(exp[i,j,r,m], axis=[r]),
                name="softmax_sum",
                tag="softmax_sum",
                requires_grad=requires_grad)
    # sum_exp [10, 20, 16]
    sum_exp = GraphOp([dim0, dim1, dim3], [dim2], [exp], _inner_sum, name="softmax_sum")

    def _inner_divide(dim0, dim1, dim2, dim3, exp, sum_exp, requires_grad=True):
        return compute([dim0, dim1, dim2, dim3],
                lambda i, j, k, m: exp[i,j,k,m] / sum_exp[i,j,m],
                name="softmax_divide",
                tag="softmax_divide",
                requires_grad=requires_grad)
    # return [10, 20, 1152, 16]
    return GraphOp([dim0, dim1, dim2, dim3], [], [exp, sum_exp], _inner_divide, name="softmax_divide")


def two_flatten(inputs, verify_num_caps):
    # inputs [20, 32, 6, 6, 8]
    # output [20, 32 * 6 * 6, 8]
    batch, dim1, dim2, dim3, num_caps = inputs.shape
    assert dim1 == 32 and dim2 == 6 and dim3 == 6 and num_caps == verify_num_caps
    dim2_dim3 = dim2 * dim3
    dim1_dim2_dim3 = dim1 * dim2_dim3

    def _inner_two_flatten(batch, dim1_dim2_dim3, num_caps, inputs, requires_grad=True):
        return compute([batch, dim1_dim2_dim3, num_caps],
                lambda i, j, k: inputs[i, j // dim2_dim3, 
                    (j % dim2_dim3) // dim3, (j % dim2_dim3) % dim3, k],
                    name="two_flatten",
                    tag="two_flatten",
                    requires_grad=requires_grad)
    return GraphOp([batch, dim1_dim2_dim3, num_caps], [], [inputs], _inner_two_flatten, name="flatten2")

# This is deprecated
# def concat_eight_vector_lastdim(cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7):
#     batch, features, one = cap1.shape
#     assert one == 1
#     stacked_one = 8
#     def _inner_cat(batch, features, stacked_one, cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7, requires_grad=True):
#     # def _inner_cat(batch, features, stacked_one, cap0, requires_grad=True):
#         return compute([batch, features, stacked_one],
#                 lambda i, j, k:
#                     tvm.te.if_then_else(k == 0, cap0[i, j, k],
#                         tvm.te.if_then_else(k == 1, cap1[i, j, k-1],
#                             tvm.te.if_then_else(k == 2, cap2[i, j, k-2],
#                                 tvm.te.if_then_else(k == 3, cap3[i, j, k-3],
#                                     tvm.te.if_then_else(k == 4, cap4[i, j, k-4],
#                                         tvm.te.if_then_else(k == 5, cap5[i, j, k-5], 
#                                             tvm.te.if_then_else(k == 6, cap6[i, j, k-6],
#                                                 cap7[i, j, k-7]))))))),
#                     #lambda i, j, k: cap0[i,j,k],
#                 name="concat",
#                 tag="concat",
#                 requires_grad=requires_grad)
#     return GraphOp([batch, features, stacked_one], [], 
#         [cap0, cap1, cap2, cap3, cap4, cap5, cap6, cap7],
#            # [cap0],
#             _inner_cat, name="concat")

# this is deprecated
# def squeeze_transpose(caps_output):
#     # caps_output [10, 20, 1, 1, 16]
#     # caps_output2:[20, 10, 16]
#     dim0, dim1, dim2, dim3, dim4 = caps_output.shape
#     assert dim2 == 1 and dim3 == 1
#     def _inner_squeeze_transpose(dim1, dim0, dim4, caps_output, requires_grad=True):
#         return compute([dim1, dim0, dim4],
#                 lambda i, j, k: i+j+k,#caps_output[j, i, 0, 0, k],
#                 name="sq_tr",
#                 tag="sq_tr",
#                 requires_grad=requires_grad)
#     return GraphOp([dim1, dim0, dim4], [], [caps_output], _inner_squeeze_transpose, name="sq_tr")