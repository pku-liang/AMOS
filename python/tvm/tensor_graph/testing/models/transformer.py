from tvm.tensor_graph.nn.functional import elementwise_add
from tvm.tensor_graph.nn.layers import GELU, Layer

from tvm.tensor_graph.core import ForwardGraph, BackwardGraph, compute, \
                              GraphTensor, GraphOp, PyTIRGraph, make_fwd_graph, \
                              make_tir_graph

import tvm.te
import tvm.tir

# import tensorflow.compat.v1 as tf

# def mask(inputs, key_masks=None, type=None):
#     """Masks paddings on keys or queries to inputs
#     inputs: 3d tensor. (h*N, T_q, T_k)
#     key_masks: 3d tensor. (N, 1, T_k)
#     type: string. "key" | "future"
#     e.g.,
#     >> inputs = tf.zeros([2, 2, 3], dtype=tf.float32)
#     >> key_masks = tf.constant([[0., 0., 1.],
#                                 [0., 1., 1.]])
#     >> mask(inputs, key_masks=key_masks, type="key")
#     array([[[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
#         [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
#        [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
#         [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]],
#        [[ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09],
#         [ 0.0000000e+00,  0.0000000e+00, -4.2949673e+09]],
#        [[ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09],
#         [ 0.0000000e+00, -4.2949673e+09, -4.2949673e+09]]], dtype=float32)
#     """
#     padding_num = -2 ** 32 + 1
#     assert type in ("k", "key", "keys")
#     key_masks = tf.tile(key_masks, [tf.shape(inputs)[0] // tf.shape(key_masks)[0], 1]) # (h*N, seqlen)
#     key_masks = tf.expand_dims(key_masks, 1)  # (h*N, 1, seqlen)
#     outputs = inputs + key_masks * padding_num

#     return outputs



def dense(inputs, weight, bias=None, out_dtype="float32"):
    batch, N, in_feature = inputs.shape
    out_feature, in_feature2 = weight.shape
    assert in_feature == in_feature2, "%d vs. %d" % (in_feature, in_feature2)

    def _inner_dense(batch, N, out_feature, in_feature, inputs, weight, requires_grad=True):
        k = tvm.te.reduce_axis((0, in_feature))
        return compute(
            [batch, N, out_feature],
            lambda i, n, j: tvm.te.sum(
                (inputs[i, n, k] * weight[j, k]).astype(out_dtype), axis=[k]),
            name="dense",
            requires_grad=requires_grad)

    withoutBias = GraphOp([batch, N, out_feature], [in_feature], [
                          inputs, weight], _inner_dense, name="linear")

    if bias is not None:
        assert(bias.shape[0] == weight.shape[0])

        def _inner_bias_add(batch, N, out_feature, withoutBias, bias, requires_grad=True):
            return compute(
                [batch, N, out_feature],
                lambda i, n, j: withoutBias[i, n, j] + bias[j],
                name="dense_bias",
                # tag=tag_gen("dense_bias"),
                requires_grad=requires_grad)
        return GraphOp([batch, N, out_feature], [], [withoutBias, bias], _inner_bias_add, name="bias_add")

    return withoutBias

class Linear_(Layer):
  def __init__(self, in_features, out_features, bias=False,
    dtype="float32", out_dtype="float32"):
    super(Linear_, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = GraphTensor(
      [out_features, in_features], dtype=dtype, name="linear_weight")
    if bias:
      self.bias = GraphTensor(
        [out_features], dtype=out_dtype, name="linear_bias", requires_grad=True)
    else:
      self.bias = None

    self.dtype = dtype
    self.out_dtype = out_dtype
    
  def forward(self, x):
    return dense(x, self.weight, self.bias, out_dtype=self.out_dtype)

def scaled_dot_product_attention(N, Q, K, V,
                                 causality=False, out_dtype="float32"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    # key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    '''
    d_k = int(Q.shape[-1])
    d_k_const = tvm.tir.const(Q.shape[-1], dtype=out_dtype)
    T_q = int(Q.shape[-2])
    T_k = int(K.shape[-2])
    d_v = int(V.shape[-1])

    # dot product and scale
    def _inner_dense_t(N, T_q, T_k, d_k, inputs, weight, requires_grad=True):
        k = tvm.te.reduce_axis((0, d_k))
        return compute(
            [N, T_q, T_k],
            lambda n, tq, tk: tvm.te.sum(
                (inputs[n, tq, k] * weight[n, tk, k] / tvm.te.sqrt(d_k_const)).astype(out_dtype), axis=[k]),
            name="dense",
            requires_grad=requires_grad)

    outputs = GraphOp([N, T_q, T_k], [d_k], [Q, K], _inner_dense_t, name="linear") # (N, T_q, T_k)

    # # key masking
    # outputs = mask(outputs, key_masks=key_masks, type="key")

    # # causality or future blinding masking
    # if causality:
    #     outputs = mask(outputs, type="future")

    # softmax
    def _inner_sum_val(N, T_q, T_k, data, requires_grad=True):
        k1 = tvm.te.reduce_axis([0, T_k], name="k1")
        return compute(
            [N, T_q],
            lambda n, t: tvm.te.sum(tvm.tir.exp(data[n, t, k1]), axis=[k1]),
            requires_grad=requires_grad)
    sum_val = GraphOp([N, T_q], [T_k], [outputs], _inner_sum_val)

    def _inner_soft(N, T_q, T_k, data, sum_val, requires_grad=True):
        return compute(
            [N, T_q, T_k],
            lambda n, tq, tk: tvm.tir.exp(data[n, tq, tk]) / sum_val[n, tq],
            requires_grad=requires_grad)
    outputs = GraphOp([N, T_q, T_k], [], [outputs, sum_val], _inner_soft)

    # weighted sum (context vectors)
    def _inner_dense(B, M, N, K, inputs, weight, requires_grad=True):
        k = tvm.te.reduce_axis((0, K))
        return compute(
            [B, M, N],
            lambda b, m, n: tvm.te.sum(
                (inputs[b, m, k] * weight[b, k, n]).astype(out_dtype), axis=[k]),
            name="dense",
            requires_grad=requires_grad)
    outputs = GraphOp([N, T_q, d_v], [T_k], [outputs, V], _inner_dense, name="linear") # (N, T_q, d_v)

    return outputs

def multihead_attention(ma_l1, ma_l2, ma_l3, ma_l4, queries, keys, values,
                        num_heads=8,
                        causality=False, out_dtype="float32"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    causality: Boolean. If true, units that reference the future are masked.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    N, T, d_model = queries.shape
    # Linear projections
    Q = ma_l1(queries) # (N, T_q, d_model)
    K = ma_l2(keys) # (N, T_k, d_model)
    V = ma_l3(values) # (N, T_k, d_model)
    
    # Split and concat
    def _split_concat(h_N, T, d_model_h, data, requires_grad=False):
        h = tvm.tir.const(num_heads, "int32")
        return compute([h_N, T, d_model_h],
                        lambda n_h, t, d_h: data[n_h // h, t, d_h * h],
                        requires_grad=requires_grad)
    Q_ = GraphOp([num_heads * N, T, d_model // num_heads], [], [Q], _split_concat, requires_grad=False) # (h*N, T_q, d_model/h)
    K_ = GraphOp([num_heads * N, T, d_model // num_heads], [], [K], _split_concat, requires_grad=False) # (h*N, T_k, d_model/h)
    V_ = GraphOp([num_heads * N, T, d_model // num_heads], [], [V], _split_concat, requires_grad=False) # (h*N, T_k, d_model/h)

    # Attention
    outputs = scaled_dot_product_attention(N, Q_, K_, V_, causality, out_dtype=out_dtype)

    # Restore shape
    def _split_concat_restore(N, T, d_model, data, requires_grad=False):
        h = tvm.tir.const(num_heads, "int32")
        return compute([N, T, d_model],
                        lambda n, t, d: data[n * h, t, d // h],
                        requires_grad=requires_grad)
    outputs = GraphOp([N, T, d_model], [], [outputs], _split_concat_restore, requires_grad=False) # (N, T_q, d_model)

    outputs = ma_l4(outputs)  # (N, T_q, d_model)

    # Residual connection
    outputs = elementwise_add(outputs, queries)
            
    # Normalize
    outputs = ln(outputs)
 
    return outputs

def ln(inputs, epsilon=1e-8):
    '''
    inputs: A 3d tensor with shape of [N, T, d_model].
    '''
    N, T, d_model = inputs.shape
    prefix = inputs.name
    epsilon = tvm.tir.const(epsilon, inputs.dtype)

    def _inner_mean(d_model, N, T, inputs, requires_grad=False):
        rn1 = tvm.te.reduce_axis([0, N], name=prefix + "_rn1")
        rt1 = tvm.te.reduce_axis([0, T], name=prefix + "_rt1")
        return compute([d_model],
                        lambda d: tvm.te.sum(
                            inputs[rn1, rt1, d] / (N*T), axis=[rn1, rt1]),
                        name=prefix + "_mean",
                        requires_grad=requires_grad)
    mean = GraphOp([d_model], [N, T], [inputs], _inner_mean,
                    name=prefix + "_mean", requires_grad=False)

    def _inner_square(d_model, N, T, inputs, requires_grad=False):
        rn2 = tvm.te.reduce_axis([0, N], name=prefix + "_rn1")
        rt2 = tvm.te.reduce_axis([0, T], name=prefix + "_rt1")
        return compute([d_model],
                        lambda d: tvm.te.sum(
                            (inputs[rn2, rt2, d] * inputs[rn2, rt2, d]) / (N*T), axis=[rn2, rt2]),
                        name=prefix + "_square",
                        requires_grad=requires_grad)
    square = GraphOp([d_model], [N, T], [inputs], _inner_square,
                        name=prefix + "_square", requires_grad=False)

    def _inner_var(d_model, square, mean, requires_grad=False):
        return compute([d_model],
                        lambda d: square[d] - mean[d] * mean[d],
                        name=prefix + "_var",
                        requires_grad=requires_grad)
    var = GraphOp([d_model], [], [square, mean], _inner_var,
                    name=prefix + "_var", requires_grad=False)

    def _inner_ln(N, T, d_model, inputs, mean, var, requires_grad=True):
        return compute([N, T, d_model],
                        lambda n, t, d: (
                            inputs[n, t, d] - mean[d]) / tvm.te.sqrt(var[d] + epsilon),
                        name=prefix + "_bn2d",
                        requires_grad=requires_grad)
    return GraphOp([N, T, d_model], [], [inputs, mean, var], _inner_ln, name=prefix+"layer_norm")

def ff(ff_l1, ff_l2, inputs, num_units):
    '''position-wise feed forward net. See 3.3
    
    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    # Inner layer
    outputs = ff_l1(inputs)
    outputs = GELU()(outputs)

    # Outer layer
    outputs = ff_l2(outputs)

    # Residual connection
    outputs = elementwise_add(outputs, inputs)
    
    # Normalize
    outputs = ln(outputs)
    
    return outputs

class Transformer(Layer):
    '''
    xs: tuple of
        x: tensor. (N, T1)
    '''
    def __init__(self, num_blocks, num_heads, d_ff, d_model, dtype="float32", out_dtype="float32"):
        super(Transformer, self).__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.d_model = d_model
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.ma_l1 = []
        self.ma_l2 = []
        self.ma_l3 = []
        self.ma_l4 = []
        self.ff_l1 = []
        self.ff_l2 = []
        for i in range(num_blocks):
            self.ma_l1.append(Linear_(d_model, d_model, bias=True, dtype=dtype, out_dtype=out_dtype))
            self.ma_l2.append(Linear_(d_model, d_model, bias=True, dtype=dtype, out_dtype=out_dtype))
            self.ma_l3.append(Linear_(d_model, d_model, bias=True, dtype=dtype, out_dtype=out_dtype))
            self.ma_l4.append(Linear_(d_model, d_model, bias=True, dtype=dtype, out_dtype=out_dtype))
            self.ff_l1.append(Linear_(d_model, self.d_ff, dtype=dtype, out_dtype=out_dtype))
            self.ff_l2.append(Linear_(self.d_ff, self.d_model, dtype=dtype, out_dtype=out_dtype))

    def forward(self, x):
        '''
        Returns
        memory: encoder outputs. (N, T1, d_model)
        '''

        enc = x

        ## Blocks
        for i in range(self.num_blocks):
            # self-attention
            enc = multihead_attention(self.ma_l1[i], self.ma_l2[i], self.ma_l3[i], self.ma_l4[i],
                                        queries=enc,
                                        keys=enc,
                                        values=enc,
                                        num_heads=self.num_heads,
                                        causality=False, out_dtype=self.out_dtype)
            # feed forward
            enc = ff(self.ff_l1[i], self.ff_l2[i], enc, num_units=[self.d_ff, self.d_model])
        return enc

if __name__ == "__main__":
    # https://huggingface.co/bert-base-uncased/blob/main/config.json
    bert_base_config = {
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }

    N = 1  # Batch Size
    T = bert_base_config['max_position_embeddings']
    d_model = bert_base_config['hidden_size']
    d_ff = bert_base_config['intermediate_size']
    num_blocks = bert_base_config['num_hidden_layers']
    num_heads = bert_base_config['num_attention_heads']
    dtype = "float16"
    out_dtype = "float16"

    net = Transformer(num_blocks, num_heads, d_ff, d_model, dtype=dtype, out_dtype=out_dtype)

    x = GraphTensor([N, T, d_model], dtype, name="data")
    outputs = net.forward(x)

    print(outputs)
    print(list(net.weights()))
