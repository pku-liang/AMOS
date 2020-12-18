import tvm
import numpy as np
from tvm import tg
from tensor_graph.core import compute, GraphOp, GraphTensor, ForwardGraph, \
  make_fwd_graph, make_tir_graph
from tensor_graph.testing.models import resnet
from tensor_graph.nn import CELoss, SGD


def zero_pad2d(inputs, padding=0):
  """
  Zero padding for 2d tensor

  Args:
  -----------------------------
  inputs : GraphNode
      shape [batch, channel, height, width]
  padding: (optional:0) int or tuple
      expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
  -----------------------------
  
  Returns:
  -----------------------------
  GraphOp
      shape [batch, channel, padded_height, padded_width]
  -----------------------------
  """
  padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
  assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
  if len(padding) == 2:
    padding = (padding[0], padding[0], padding[1], padding[1])
  assert (len(padding) == 4)
  batch_size, in_channel, height, width = inputs.shape
  padded_shape = (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3])
  def _inner_zero_pad2d(batch_size, in_channel, h, w, inputs, requires_grad=True):
    # Warning, we use "float32" as type of 0
    padding_zero = tvm.tir.expr.const(0, "float32")
    return compute(padded_shape,
      lambda b, c, h, w: tvm.te.if_then_else(
            tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
            inputs[b, c, h - padding[0], w - padding[2]],
            padding_zero
            ),
      name="zero_pad2d",
      tag="TG_AUTOGEN",
      requires_grad=requires_grad)
  return GraphOp(padded_shape, [] , [inputs], _inner_zero_pad2d, name="zero_pad2d")


def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
  """Convolution 2d NCHW layout

  Args:
  -----------------------------
  inputs  : GraphNode
      shape [batch, channel, height, width]
  weight  : GraphNode
      shape [out_channel, channel // groups, kernel_height, kernel_width]
  bias    : (optional:None) GraphNode
      shape [out_channel]
  stride  : (optional:1) int or tuple

  padding : (optional:0) int or tuple

  dilation: (optional:1) int

  groups  : (optional:1) int
  -----------------------------

  Returns:
  -----------------------------
  GraphOp
      shape [batch, out_channel, output_height, output_width]
  -----------------------------
  """
  batch_size, in_channel, in_h, in_w = inputs.shape
  out_channel, channel_per_group, k_h, k_w = weight.shape
  assert channel_per_group * groups == in_channel
  out_channel_per_group = out_channel // groups
  assert out_channel_per_group * groups == out_channel

  stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
  padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
  dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
  assert isinstance(stride, tuple) and len(stride) == 2
  assert isinstance(padding, tuple) and len(padding) == 2
  assert isinstance(dilation, tuple) and len(dilation) == 2

  out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
  out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1

  padded = zero_pad2d(inputs, padding=padding)
  conv_out_shape = (batch_size, out_channel, out_h, out_w)
  def _inner_conv2d_nchw(batch_size, out_channel, out_h, out_w, channel_per_group,
                            k_w, k_h, padded, weight, requires_grad=True):
    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    return compute(conv_out_shape,
        lambda b, c, h, w: tvm.te.sum(
          (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
          * weight[c, rc, rh, rw]),
          axis=[rc, rw, rh]),
          name="conv2d_nchw",
          tag="TG_AUTOGEN",
          requires_grad=requires_grad
        )
  conv_out = GraphOp(
    conv_out_shape,
    [channel_per_group, k_w, k_h],
    [padded, weight],
    _inner_conv2d_nchw,
    name="conv2d_nchw")
  def _inner_bias(batch_size, out_channel, out_h, out_w, conv_out, bias, requires_grad=True):
    return compute(
      (batch_size, out_channel, out_h, out_w),
      lambda b, c, h, w: conv_out[b, c, h, w] + bias[c],
      name="conv2d_nchw_bias",
      tag="TG_AUTOGEN",
      requires_grad=requires_grad
      )
  if bias is not None:
    return GraphOp(conv_out_shape, [], [conv_out, bias], _inner_bias, name="conv2d_bias")
  return conv_out


def test1():
  print("test 1 ##############################")
  A = GraphTensor([1, 1024, 14, 14])
  W = GraphTensor([2048, 1024, 3, 3])
  B = conv2d_nchw(A, W, bias=None, stride=1, padding=1)

  B_tensor, params = B({})
  tvm_tensor = B_tensor.tvm_tensor
  print(tvm_tensor.op.tag)
  print(tvm_tensor.op.body)
  print(tvm_tensor.op.input_tensors[0].op.tag)

  print("Success!")

def test2():
  print("test 2 ##############################")
  batch = 64
  img_shape = [batch, 3, 224, 224]
  num_classes = 1000
  label_shape = [batch, num_classes]
  dtype = "float32"
  model = resnet.resnet50(num_classes=1000)
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_np = np.random.uniform(-1, 1, [batch, num_classes]).astype(dtype)

  ce_loss = CELoss(label_tensor)
  sgd = SGD(0.002)

  fwd_graph = make_fwd_graph(model, [img_tensor])
  tir_graph = make_tir_graph(fwd_graph, inference=True)

  op_map = {}
  for op in tir_graph.operation_list:
    if op.tag not in op_map:
      op_map[op.tag] = []
    op_map[op.tag].append(op)

  print("Totally", len(op_map), "different tags.")
  for k, v in op_map.items():
    print("Tag:", k, "has", len(v), "operators")


  print("Success")


def test3():
  print("test 3 ##############################")
  batch = 64
  img_shape = [batch, 3, 224, 224]
  num_classes = 1000
  label_shape = [batch, num_classes]
  dtype = "float32"
  model = resnet.resnet50(num_classes=1000)
  img_tensor = GraphTensor(img_shape, dtype=dtype, name="image")
  label_tensor = GraphTensor(label_shape, dtype=dtype, name="label")

  # this is data
  img_np = np.random.uniform(-1, 1, img_shape).astype(dtype)
  label_np = np.random.uniform(-1, 1, [batch, num_classes]).astype(dtype)

  ce_loss = CELoss(label_tensor)
  sgd = SGD(0.002)

  fwd_graph = make_fwd_graph(model, [img_tensor])
  tir_graph = make_tir_graph(fwd_graph, loss=ce_loss, optimizer=sgd, inference=False)

  op_map = {}
  for op in tir_graph.operation_list:
    if op.tag not in op_map:
      op_map[op.tag] = []
    op_map[op.tag].append(op)

  
  with open("trace.log", "w") as fout:
    print("Totally", len(op_map), "different tags.", file=fout)
    for k, v in op_map.items():
      print("Tag:", k, "has", len(v), "operators", file=fout)


  print("Success")


if __name__ == "__main__":
  test1()
  test2()
  test3()