"""
author: zhengsz@pku.edu.cn
"""

import tvm 
import numpy as np
import math
import torch
import time

torch.backends.cudnn.enabled = False


def zero_pad2d(inputs, padding=0):
    """Zero padding for 2d tensor

    Args:
    -----------------------------
    inputs : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    padding: (optional:0) int or tuple
        expected: (h_pad_up, h_pad_down, w_pad_up, w_pad_down)
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, padded_height, padded_width]
    -----------------------------
    """
    padding = (padding, padding, padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    assert isinstance(padding, tuple), "type(padding)={}".format(type(padding))
    if len(padding) == 2:
        padding = (padding[0], padding[0], padding[1], padding[1])
    assert (len(padding) == 4)

    padding_zero = tvm.tir.expr.const(0, inputs.dtype)

    batch_size, in_channel, height, width = inputs.shape
    return tvm.te.compute(
        (batch_size, in_channel, height + padding[0] + padding[1], width + padding[2] + padding[3]),
        lambda b, c, h, w: tvm.te.if_then_else(
                            tvm.te.all(h >= padding[0], h < height + padding[0], w >= padding[2], w < width + padding[2]),
                            inputs[b, c, h - padding[0], w - padding[2]],
                            padding_zero
                            ),
        name='Padding'
        )


def conv2d_nchw(inputs, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Convolution 2d NCHW layout

    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    weight  : tvm.te.tensor.Tensor
        shape [out_channel, channel // groups, kernel_height, kernel_width]
    bias    : (optional:None) tvm.te.tensor.Tensor
        shape [out_channel]
    stride  : (optional:1) int or tuple

    padding : (optional:0) int or tuple

    dilation: (optional:1) int

    groups  : (optional:1) int
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, out_channel, output_height, output_width]
    -----------------------------
    """
    batch_size, in_channel, in_h, in_w = inputs.shape
    out_channel, channel_per_group, k_h, k_w = weight.shape
    assert (channel_per_group * groups).value == in_channel.value
    out_channel_per_group = out_channel // groups
    assert (out_channel_per_group * groups).value == out_channel.value

    stride = (stride, stride) if isinstance(stride, (int, tvm.tir.IntImm)) else stride
    padding = (padding, padding) if isinstance(padding, (int, tvm.tir.IntImm)) else padding
    dilation = (dilation, dilation) if isinstance(dilation, (int, tvm.tir.IntImm)) else dilation
    assert isinstance(stride, tuple) and len(stride) == 2
    assert isinstance(padding, tuple) and len(padding) == 2
    assert isinstance(dilation, tuple) and len(dilation) == 2

    out_h = (in_h + 2 * padding[0] - dilation[0] * (k_h - 1) - 1) // stride[0] + 1
    out_w = (in_w + 2 * padding[1] - dilation[1] * (k_w - 1) - 1) // stride[1] + 1
    rc = tvm.te.reduce_axis((0, channel_per_group), name="rc")
    rh = tvm.te.reduce_axis((0, k_h), name="rh")
    rw = tvm.te.reduce_axis((0, k_w), name="rw")

    padded = zero_pad2d(inputs, padding=padding)
    output = tvm.te.compute(
        (batch_size, out_channel, out_h, out_w),
        lambda b, c, h, w: tvm.te.sum(
            (padded[b, c // out_channel_per_group * channel_per_group + rc, 
                    h * stride[0] + rh * dilation[0], w * stride[1] + rw * dilation[1]]
            * weight[c, rc, rh, rw]),
            axis=[rc, rw, rh]
        )
    )
    if bias is not None:
        output = tvm.te.compute(
            (batch_size, out_channel, out_h, out_w),
            lambda b, c, h, w: output[b, c, h, w] + bias[c]
        )
    return output


def batch_norm(inputs, alpha, beta, epsilon=1e-5):
  """Batch Normalization for NCHW inputs
  
    Args:
    -----------------------------
    inputs  : tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    alpha   : tvm.te.tensor.Tensor
        shape [channel]
    beta    : tvm.te.tensor.Tensor
        shape [channel]
    epsilon : float
        optional
    -----------------------------

    Returns:
    -----------------------------
    tvm.te.tensor.Tensor
        shape [batch, channel, height, width]
    -----------------------------
  """
  assert isinstance(inputs, tvm.te.tensor.Tensor)
  assert isinstance(alpha, tvm.te.tensor.Tensor)
  assert isinstance(beta, tvm.te.tensor.Tensor)
  N, C, H, W = inputs.shape
  prefix = inputs.name
  epsilon = tvm.tir.const(epsilon, inputs.dtype)
  rn1 = tvm.te.reduce_axis([0, N], name=prefix + "_rn1")
  rh1 = tvm.te.reduce_axis([0, H], name=prefix + "_rh1")
  rw1 = tvm.te.reduce_axis([0, W], name=prefix + "_rw1")
  rn2 = tvm.te.reduce_axis([0, N], name=prefix + "_rn2")
  rh2 = tvm.te.reduce_axis([0, H], name=prefix + "_rh2")
  rw2 = tvm.te.reduce_axis([0, W], name=prefix + "_rw2")
  assert (len(alpha.shape) == 1) and (alpha.shape[0].value == C.value)
  assert (len(beta.shape) == 1) and (beta.shape[0].value == C.value)
  mean = tvm.te.compute([C],
    lambda c: tvm.te.sum(inputs[rn1, c, rh1, rw1] / (N*H*W), axis=[rn1, rh1, rw1]),
    name=prefix + "_mean")
  var = tvm.te.compute([C],
    lambda c: tvm.te.sum(tvm.te.power(inputs[rn2, c, rh2, rw2] - mean[c], 2) / (N*H*W), axis=[rn2, rh2, rw2]),
    name=prefix + "_var")
  inputs_p = tvm.te.placeholder(inputs.shape, dtype=inputs.dtype, name=prefix + "_inputs_p")
  mean_p = tvm.te.placeholder(mean.shape, dtype=mean.dtype, name=prefix + "_mean_p")
  var_p = tvm.te.placeholder(var.shape, dtype=var.dtype, name=prefix + "_var_p")
  bn = tvm.te.compute([N, C, H, W],
    lambda n, c, i, j: (inputs_p[n, c, i, j] - mean_p[c]) / tvm.te.sqrt(var_p[c] + epsilon) * alpha[c] + beta[c],
    name=prefix + "_bn")
  return (mean, var), (inputs_p, mean_p, var_p), bn


def relu(inputs):
  return tvm.te.compute(
          inputs.shape,
          lambda *i: tvm.te.if_then_else(inputs(*i) > 0, inputs(*i), tvm.tir.const(0, inputs.dtype)))


def add(a, b):
  assert len(a.shape) == len(b.shape)
  for i, dim in enumerate(a.shape):
    assert dim.value == b.shape[i].value
  return tvm.te.compute(a.shape, lambda *args: a(*args) + b(*args), name=a.name + "_add_" + b.name)


def pytorch_mismatch_bottle_neck(N, C, H, W, K, L, dtype="float32", target="llvm", training=True):
  conv1 = torch.nn.Conv2d(C, K, 1, stride=2, padding=0, dilation=1, groups=1, bias=False)
  conv2 = torch.nn.Conv2d(K, K, 3, stride=1, padding=1, dilation=1, groups=1, bias=False)
  conv3 = torch.nn.Conv2d(K, L, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
  conv4 = torch.nn.Conv2d(C, L, 1, stride=2, padding=0, dilation=1, groups=1, bias=False)
  bn1 = torch.nn.BatchNorm2d(K, track_running_stats=training)
  bn2 = torch.nn.BatchNorm2d(K, track_running_stats=training)
  bn3 = torch.nn.BatchNorm2d(L, track_running_stats=training)
  bn4 = torch.nn.BatchNorm2d(L, track_running_stats=training)
  relu = torch.relu 

  def _func(img_np, weights_np, alphas_np, betas_np, output_np, repeat=10, min_repeat_ms=200):
    img_torch = torch.tensor(img_np)
    weight1_np, weight2_np, weight3_np, weight4_np = weights_np
    alpha1_np, alpha2_np, alpha3_np, alpha4_np = alphas_np
    beta1_np, beta2_np, beta3_np, beta4_np = betas_np
    conv1.weight = torch.nn.Parameter(torch.tensor(weight1_np, requires_grad=training))
    conv2.weight = torch.nn.Parameter(torch.tensor(weight2_np, requires_grad=training))
    conv3.weight = torch.nn.Parameter(torch.tensor(weight3_np, requires_grad=training))
    conv4.weight = torch.nn.Parameter(torch.tensor(weight4_np, requires_grad=training))
    bn1.weight = torch.nn.Parameter(torch.tensor(alpha1_np, requires_grad=training))
    bn1.bias = torch.nn.Parameter(torch.tensor(beta1_np, requires_grad=training))
    bn2.weight = torch.nn.Parameter(torch.tensor(alpha2_np, requires_grad=training))
    bn2.bias = torch.nn.Parameter(torch.tensor(beta2_np, requires_grad=training))
    bn3.weight = torch.nn.Parameter(torch.tensor(alpha3_np, requires_grad=training))
    bn3.bias = torch.nn.Parameter(torch.tensor(beta3_np, requires_grad=training))
    bn4.weight = torch.nn.Parameter(torch.tensor(alpha4_np, requires_grad=training))
    bn4.bias = torch.nn.Parameter(torch.tensor(beta4_np, requires_grad=training))

    layers = [conv1, conv2, conv3, conv4, bn1, bn2, bn3, bn4]

    def _run_net():
      conv_res1 = conv1(img_torch)
      bn_res1 = bn1(conv_res1)
      relu_res1 = relu(bn_res1)
      conv_res2 = conv2(relu_res1)
      bn_res2 = bn2(conv_res2)
      relu_res2 = relu(bn_res2)
      conv_res3 = conv3(relu_res2)
      bn_res3 = bn3(conv_res3)
      conv_res4 = conv4(img_torch)
      bn_res4 = bn4(conv_res4)
      add_res = bn_res3 + bn_res4

      return relu(add_res)

    time_lst = []
    if target == "llvm":
      # warmup
      output = _run_net()
      for i in range(repeat):
        begin = time.time()
        count = 0
        while (time.time() - begin < min_repeat_ms/1e3):
          output = _run_net()
          count += 1
        interval = time.time() - begin
        if count > 0:
          time_lst.append(interval/count)
        else:
          time_lst.append(float("inf"))
    elif target == "cuda":
      img_torch = img_torch.cuda(0)
      for layer in layers:
        layer.cuda(0)
      # warmup
      output = _run_net()
      for i in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = _run_net()
        end.record()
        torch.cuda.synchronize()
        interval = start.elapsed_time(end)
        time_lst.append(interval/1e3)

    time_lst = np.array(time_lst) * 1e3
    mean_time = time_lst.mean()
    dev = np.sqrt(time_lst.var())

    return output.detach().cpu().numpy(), (mean_time, dev)

  return _func


def tvm_conv_bn_relu_conv(N, C, H, W, K, L, dtype="float32", target="llvm", training=True):
  img = tvm.te.placeholder([N, C, H, W], dtype=dtype, name="img")
  weight1 = tvm.te.placeholder([K, C, 1, 1], dtype=dtype, name="weight1", requires_grad=training)
  weight2 = tvm.te.placeholder([K, K, 3, 3], dtype=dtype, name="weight2", requires_grad=training)
  weight3 = tvm.te.placeholder([L, K, 1, 1], dtype=dtype, name="weight3", requires_grad=training)
  weight4 = tvm.te.placeholder([L, C, 1, 1], dtype=dtype, name="weight4", requires_grad=training)
  alpha1 = tvm.te.placeholder([K], dtype=dtype, name="alpha1", requires_grad=training)
  beta1 = tvm.te.placeholder([K], dtype=dtype, name="beta1", requires_grad=training)
  alpha2 = tvm.te.placeholder([K], dtype=dtype, name="alpha2", requires_grad=training)
  beta2 = tvm.te.placeholder([K], dtype=dtype, name="beta2", requires_grad=training)
  alpha3 = tvm.te.placeholder([L], dtype=dtype, name="alpha3", requires_grad=training)
  beta3 = tvm.te.placeholder([L], dtype=dtype, name="beta3", requires_grad=training)
  alpha4 = tvm.te.placeholder([L], dtype=dtype, name="alpha4", requires_grad=training)
  beta4 = tvm.te.placeholder([L], dtype=dtype, name="beta4", requires_grad=training)
  conv_res1 = conv2d_nchw(img, weight1, stride=2, padding=0, dilation=1, groups=1)
  (mean_res1, var_res1), (inputs_p1, mean_p1, var_p1), bn_res1 = batch_norm(conv_res1, alpha1, beta1)
  relu_res1 = relu(bn_res1)
  conv_res2 = conv2d_nchw(relu_res1, weight2, stride=1, padding=1, dilation=1, groups=1)
  (mean_res2, var_res2), (inputs_p2, mean_p2, var_p2), bn_res2 = batch_norm(conv_res2, alpha2, beta2)
  relu_res2 = relu(bn_res2)
  conv_res3 = conv2d_nchw(relu_res2, weight3, stride=1, padding=0, dilation=1, groups=1)
  (mean_res3, var_res3), (inputs_p3, mean_p3, var_p3), bn_res3 = batch_norm(conv_res3, alpha3, beta3)
  conv_res4 = conv2d_nchw(img, weight4, stride=2, padding=0, dilation=1, groups=1)
  (mean_res4, var_res4), (inputs_p4, mean_p4, var_p4), bn_res4 = batch_norm(conv_res4, alpha4, beta4)
  add_res = add(bn_res3, bn_res4)
  output = relu(add_res)

  def _func(img_np, weights_np, alphas_np, betas_np, output_np, repeat=10, min_repeat_ms=200):
    # naive schedule
    s1 = tvm.te.create_schedule([var_res1.op])
    s2 = tvm.te.create_schedule([var_res2.op])
    s3 = tvm.te.create_schedule([var_res3.op])
    s4 = tvm.te.create_schedule([var_res4.op])
    s5 = tvm.te.create_schedule([output.op])
    if target == "llvm":
      pass
    elif target == "cuda":
      ############################################
      ############################################
      def _schedule_s(sch, conv_res, var_res, *args):
        pad1 = sch[conv_res].op.input_tensors[0]
        sch[pad1].compute_inline()
        _, mean_t = sch[var_res].op.input_tensors
        
        ###########################
        # schedule for conv_res
        # create cache
        write_cache = sch.cache_write(conv_res, "local")
        weight = sch[write_cache].op.input_tensors[1]
        read_share_weight = sch.cache_read(weight, "shared", [write_cache])
        read_local_weight = sch.cache_read(read_share_weight, "local", [write_cache])
        read_share_inputs = sch.cache_read(pad1, "shared", [write_cache])
        read_local_inputs = sch.cache_read(read_share_inputs, "local", [write_cache])

        # tunable parameters
        b_factors = args[0]
        k_factors = args[1]
        p_factors = args[2]
        q_factors = args[3]
        rc_factors = args[4]
        ry_factors = args[5]
        rx_factors = args[6]

        # prepare thread_axis
        bx = tvm.te.thread_axis("blockIdx.x")
        by = tvm.te.thread_axis("blockIdx.y")
        bz = tvm.te.thread_axis("blockIdx.z")
        vx = tvm.te.thread_axis("vthread")
        vy = tvm.te.thread_axis("vthread")
        vz = tvm.te.thread_axis("vthread")
        tx = tvm.te.thread_axis("threadIdx.x")
        ty = tvm.te.thread_axis("threadIdx.y")
        tz = tvm.te.thread_axis("threadIdx.z")

        # split the spatial axes
        b, k, p, q = sch[conv_res].op.axis

        kernel_scope, b = sch[conv_res].split(b, nparts=1)

        bo, bi = sch[conv_res].split(b, nparts=b_factors[0])
        ko, ki = sch[conv_res].split(k, nparts=k_factors[0])
        po, pi = sch[conv_res].split(p, nparts=p_factors[0])
        qo, qi = sch[conv_res].split(q, nparts=q_factors[0])

        vbo, bi = sch[conv_res].split(bi, nparts=b_factors[1])
        vko, ki = sch[conv_res].split(ki, nparts=k_factors[1])
        vpo, pi = sch[conv_res].split(pi, nparts=p_factors[1])
        vqo, qi = sch[conv_res].split(qi, nparts=q_factors[1])

        tbo, bi = sch[conv_res].split(bi, nparts=b_factors[2])
        tko, ki = sch[conv_res].split(ki, nparts=k_factors[2])
        tpo, pi = sch[conv_res].split(pi, nparts=p_factors[2])
        tqo, qi = sch[conv_res].split(qi, nparts=q_factors[2])

        # reorder
        sch[conv_res].reorder(bo, ko, po, qo, vbo, vko, vpo, vqo, tbo, tko, tpo, tqo, bi, ki, pi, qi)

        # fuse
        bko = sch[conv_res].fuse(bo, ko)
        vbko = sch[conv_res].fuse(vbo, vko)
        tbko = sch[conv_res].fuse(tbo, tko)
        bki = sch[conv_res].fuse(bi, ki)

        # bind
        sch[conv_res].bind(bko, bz)
        sch[conv_res].bind(po, by)
        sch[conv_res].bind(qo, bx)
        sch[conv_res].bind(vbko, vz)
        sch[conv_res].bind(vpo, vy)
        sch[conv_res].bind(vqo, vx)
        sch[conv_res].bind(tbko, tz)
        sch[conv_res].bind(tpo, ty)
        sch[conv_res].bind(tqo, tx)

        # compute at write cache
        sch[write_cache].compute_at(sch[conv_res], tqo)

        rc, ry, rx = sch[write_cache].op.reduce_axis
        rco, rci = sch[write_cache].split(rc, nparts=rc_factors[0])
        rcm, rci = sch[write_cache].split(rci, nparts=rc_factors[1])
        ryo, ryi = sch[write_cache].split(ry, nparts=ry_factors[0])
        rym, ryi = sch[write_cache].split(ryi, nparts=ry_factors[1])
        rxo, rxi = sch[write_cache].split(rx, nparts=rx_factors[0])
        rxm, rxi = sch[write_cache].split(rxi, nparts=rx_factors[1])
        a, b, c, d = sch[write_cache].op.axis
        sch[write_cache].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, a, b, c, d)

        # compute at read cache
        sch[read_share_weight].compute_at(sch[write_cache], rxm)
        sch[read_local_weight].compute_at(sch[write_cache], rxi)
        sch[read_share_inputs].compute_at(sch[write_cache], rxm)
        sch[read_local_inputs].compute_at(sch[write_cache], rxi)

        # cooperative fetching
        for cache in [read_share_inputs, read_share_weight]:
            cb, ck, ch, cw = sch[cache].op.axis
            fused = sch[cache].fuse(cb, ck, ch, cw)
            fused, bindx = sch[cache].split(fused, factor=q_factors[2])
            fused, bindy = sch[cache].split(fused, factor=p_factors[2])
            fused, bindz = sch[cache].split(fused, factor=b_factors[2] * k_factors[2])       
            
            sch[cache].bind(bindx, tx)
            sch[cache].bind(bindy, ty)
            sch[cache].bind(bindz, tz)
        
        sch[conv_res].pragma(kernel_scope, 'auto_unroll_max_step', 1500)
        sch[conv_res].pragma(kernel_scope, 'unroll_explicit', 0)

        ###########################
        # schedule mean
        c_factors = args[7]
        rn_factors = args[8]
        rh_factors = args[9]
        rw_factors = args[10]
        c, = sch[mean_t].op.axis
        rn, rh, rw = sch[mean_t].op.reduce_axis

        # prepare thread_axis
        bx = tvm.te.thread_axis("blockIdx.x")
        by = tvm.te.thread_axis("blockIdx.y")
        bz = tvm.te.thread_axis("blockIdx.z")
        vx = tvm.te.thread_axis("vthread")
        vy = tvm.te.thread_axis("vthread")
        vz = tvm.te.thread_axis("vthread")
        tx = tvm.te.thread_axis("threadIdx.x")
        ty = tvm.te.thread_axis("threadIdx.y")
        tz = tvm.te.thread_axis("threadIdx.z")

        # rwo, rw = sch[mean_t].split(rw, nparts=rw_factors[0])
        # rho, rh = sch[mean_t].split(rh, nparts=rh_factors[0])
        # mean_t_rf = sch.rfactor(mean_t, rho)
        rhw = sch[mean_t].fuse(rh, rw)
        rhwo, rhwi = sch[mean_t].split(rhw, nparts=rh_factors[0] * rw_factors[0])
        mean_t_rf = sch.rfactor(mean_t, rhwo)

        c = sch[mean_t].op.axis[0]
        co, ci = sch[mean_t].split(c, nparts=c_factors[0])
        sch[mean_t].bind(co, bx)
        # sch[mean_t].bind(ci, bx)
        sch[mean_t_rf].compute_at(sch[mean_t], sch[mean_t].op.reduce_axis[0])
        sch[mean_t].bind(sch[mean_t].op.reduce_axis[0], tx)

        ###########################
        # schedule var
        c_factors = args[11]
        rn_factors = args[12]
        rh_factors = args[13]
        rw_factors = args[14]
        c, = sch[var_res].op.axis
        rn, rh, rw = sch[var_res].op.reduce_axis

        # prepare thread_axis
        bx = tvm.te.thread_axis("blockIdx.x")
        by = tvm.te.thread_axis("blockIdx.y")
        bz = tvm.te.thread_axis("blockIdx.z")
        vx = tvm.te.thread_axis("vthread")
        vy = tvm.te.thread_axis("vthread")
        vz = tvm.te.thread_axis("vthread")
        tx = tvm.te.thread_axis("threadIdx.x")
        ty = tvm.te.thread_axis("threadIdx.y")
        tz = tvm.te.thread_axis("threadIdx.z")

        rhw = sch[var_res].fuse(rh, rw)
        rhwo, rhwi = sch[var_res].split(rhw, nparts=rh_factors[0] * rw_factors[0])
        var_res_rf = sch.rfactor(var_res, rhwo)

        c = sch[var_res].op.axis[0]
        co, ci = sch[var_res].split(c, nparts=c_factors[0])
        sch[var_res].bind(co, bx)
        sch[var_res_rf].compute_at(sch[var_res], sch[var_res].op.reduce_axis[0])
        sch[var_res].bind(sch[var_res].op.reduce_axis[0], tx)

      #########################
      # schedule for s1
      _schedule_s(
        s1,
        conv_res1,
        var_res1,
        # conv
        [4, 1, 1, 1],  # 4
        [4, 1, 64, 1],  # 256
        [7, 1, 2, 1],  # 14
        [1, 1, 2, 7],  # 14
        [64, 1, 8],  # 512
        [1, 1, 1],  # 1
        [1, 1, 1],  # 1
        # mean
        [256, 1, 1, 1],  # 256
        [4, 1, 1, 1],  # 4
        [14, 1, 1, 1],  # 14
        [14, 1, 1, 1],  # 14
        # var
        [256, 1, 1, 1],  # 256
        [4, 1, 1, 1],  # 4
        [7, 1, 1, 2],  # 14
        [14, 1, 1, 1]  # 14
      )

      ###########################
      # schedule s2
      s2[bn_res1].compute_inline()
      s2[relu_res1].compute_inline()
      _schedule_s(
        s2,
        conv_res2,
        var_res2,
        # conv
        [4, 1, 1, 1],  # 4
        [8, 1, 32, 1],  # 256
        [7, 1, 2, 1],  # 14
        [1, 1, 2, 7],  # 14
        [16, 1, 16],  # 256
        [3, 1, 1],  # 3
        [1, 1, 3],  # 3
        # mean
        [256, 1, 1, 1],  # 256
        [4, 1, 1, 1],  # 4
        [14, 1, 1, 1],  # 14
        [14, 1, 1, 1],  # 14
        # var
        [256, 1, 1, 1],  # 256
        [4, 1, 1, 1],  # 4
        [7, 1, 1, 2],  # 14
        [14, 1, 1, 1]  # 14
      )

      ###########################
      # schedule s3
      s3[bn_res2].compute_inline()
      s3[relu_res2].compute_inline()
      _schedule_s(
        s3,
        conv_res3,
        var_res3,
        # conv
        [4, 1, 1, 1],  # 4
        [8, 1, 16, 8],  # 1024
        [7, 1, 2, 1],  # 14
        [1, 1, 14, 1],  # 14
        [16, 1, 16],  # 256
        [1, 1, 1],  # 1
        [1, 1, 1],  # 1
        # mean
        [256, 4, 1, 1],  # 1024
        [4, 1, 1, 1],  # 4
        [14, 1, 1, 1],  # 14
        [14, 1, 1, 1],  # 14
        # var
        [256, 4, 1, 1],  # 1024
        [4, 1, 1, 1],  # 4
        [7, 1, 1, 2],  # 14
        [14, 1, 1, 1]  # 14
      )

      ###########################
      # schedule s4
      _schedule_s(
        s4,
        conv_res4,
        var_res4,
        # conv
        [4, 1, 1, 1],  # 4
        [4, 1, 64, 4],  # 1024
        [7, 1, 2, 1],  # 14
        [1, 1, 2, 7],  # 14
        [32, 1, 16],  # 512
        [1, 1, 1],  # 1
        [1, 1, 1],  # 1
        # mean
        [256, 4, 1, 1],  # 1024
        [4, 1, 1, 1],  # 4
        [14, 1, 1, 1],  # 14
        [14, 1, 1, 1],  # 14
        # var
        [256, 4, 1, 1],  # 1024
        [4, 1, 1, 1],  # 4
        [7, 1, 1, 2],  # 14
        [14, 1, 1, 1]  # 14
      )

      ###########################
      # schedule s5
      s5[bn_res3].compute_inline()
      s5[bn_res4].compute_inline()
      s5[add_res].compute_inline()

      n, c, h, w = s5[output].op.axis
      n_factors = [4, 1, 1, 1]  # 4
      c_factors = [64, 1, 16, 1]  # 1024
      h_factors = [2, 1, 7, 1]  # 14
      w_factors = [7, 1, 2, 1]  # 14

      # prepare thread_axis
      bx = tvm.te.thread_axis("blockIdx.x")
      by = tvm.te.thread_axis("blockIdx.y")
      bz = tvm.te.thread_axis("blockIdx.z")
      vx = tvm.te.thread_axis("vthread")
      vy = tvm.te.thread_axis("vthread")
      vz = tvm.te.thread_axis("vthread")
      tx = tvm.te.thread_axis("threadIdx.x")
      ty = tvm.te.thread_axis("threadIdx.y")
      tz = tvm.te.thread_axis("threadIdx.z")

      no, ni = s5[output].split(n, nparts=n_factors[0])
      co, ci = s5[output].split(c, nparts=c_factors[0])
      ho, hi = s5[output].split(h, nparts=h_factors[0])
      wo, wi = s5[output].split(w, nparts=w_factors[0])

      vno, ni = s5[output].split(ni, nparts=n_factors[1])
      vco, ci = s5[output].split(ci, nparts=c_factors[1])
      vho, hi = s5[output].split(hi, nparts=h_factors[1])
      vwo, wi = s5[output].split(wi, nparts=w_factors[1])

      tno, ni = s5[output].split(ni, nparts=n_factors[2])
      tco, ci = s5[output].split(ci, nparts=c_factors[2])
      tho, hi = s5[output].split(hi, nparts=h_factors[2])
      two, wi = s5[output].split(wi, nparts=w_factors[2])

      # reorder
      s5[output].reorder(no, co, ho, wo, vno, vco, vho, vwo, tno, tco, tho, two, ni, ci, hi, wi)

      # bind
      nco = s5[output].fuse(no, co)
      s5[output].bind(nco, bz)
      s5[output].bind(ho, by)
      s5[output].bind(wo, bx)
      s5[output].bind(vco, vz)
      s5[output].bind(vho, vy)
      s5[output].bind(vwo, vx)
      s5[output].bind(tco, tz)
      s5[output].bind(tho, ty)
      s5[output].bind(two, tx)

    
    ctx = tvm.context(target, 0)
    img_tvm = tvm.nd.array(img_np, ctx)
    weight1_np, weight2_np, weight3_np, weight4_np = weights_np
    weight1_tvm = tvm.nd.array(weight1_np, ctx)
    weight2_tvm = tvm.nd.array(weight2_np, ctx)
    weight3_tvm = tvm.nd.array(weight3_np, ctx)
    weight4_tvm = tvm.nd.array(weight4_np, ctx)
    inputs_p1_tvm = tvm.nd.array(np.zeros([N, K, H//2, W//2]).astype(img_tvm.dtype), ctx)
    inputs_p2_tvm = tvm.nd.array(np.zeros([N, K, H//2, W//2]).astype(img_tvm.dtype), ctx)
    inputs_p3_tvm = tvm.nd.array(np.zeros([N, L, H//2, W//2]).astype(img_tvm.dtype), ctx)
    inputs_p4_tvm = tvm.nd.array(np.zeros([N, L, H//2, W//2]).astype(img_tvm.dtype), ctx)
    mean_p1_tvm = tvm.nd.array(np.zeros([K]).astype(img_tvm.dtype), ctx)
    mean_p2_tvm = tvm.nd.array(np.zeros([K]).astype(img_tvm.dtype), ctx)
    mean_p3_tvm = tvm.nd.array(np.zeros([L]).astype(img_tvm.dtype), ctx)
    mean_p4_tvm = tvm.nd.array(np.zeros([L]).astype(img_tvm.dtype), ctx)
    var_p1_tvm = tvm.nd.array(np.zeros([K]).astype(img_tvm.dtype), ctx)
    var_p2_tvm = tvm.nd.array(np.zeros([K]).astype(img_tvm.dtype), ctx)
    var_p3_tvm = tvm.nd.array(np.zeros([L]).astype(img_tvm.dtype), ctx)
    var_p4_tvm = tvm.nd.array(np.zeros([L]).astype(img_tvm.dtype), ctx)
    alpha1_tvm = tvm.nd.array(alphas_np[0], ctx)
    alpha2_tvm = tvm.nd.array(alphas_np[1], ctx)
    alpha3_tvm = tvm.nd.array(alphas_np[2], ctx)
    alpha4_tvm = tvm.nd.array(alphas_np[3], ctx)
    beta1_tvm = tvm.nd.array(betas_np[0], ctx)
    beta2_tvm = tvm.nd.array(betas_np[1], ctx)
    beta3_tvm = tvm.nd.array(betas_np[2], ctx)
    beta4_tvm = tvm.nd.array(betas_np[3], ctx)
    output_tvm = tvm.nd.array(output_np, ctx)

    print(tvm.lower(s1, [img, weight1, conv_res1, mean_res1, var_res1], simple_mode=True))
    print(tvm.lower(s2, [inputs_p1, mean_p1, var_p1, alpha1, beta1, weight2, conv_res2, mean_res2, var_res2], simple_mode=True))
    print(tvm.lower(s3, [inputs_p2, mean_p2, var_p2, alpha2, beta2, weight3, conv_res3, mean_res3, var_res3], simple_mode=True))
    print(tvm.lower(s4, [img, weight4, conv_res4, mean_res4, var_res4], simple_mode=True))
    print(tvm.lower(s5, [inputs_p3, mean_p3, var_p3, alpha3, beta3, inputs_p4, mean_p4, var_p4, alpha4, beta4, output], simple_mode=True))
    func1 = tvm.build(s1, [img, weight1, conv_res1, mean_res1, var_res1], target=target)
    func2 = tvm.build(s2, [inputs_p1, mean_p1, var_p1, alpha1, beta1, weight2, conv_res2, mean_res2, var_res2], target=target)
    func3 = tvm.build(s3, [inputs_p2, mean_p2, var_p2, alpha2, beta2, weight3, conv_res3, mean_res3, var_res3], target=target)
    func4 = tvm.build(s4, [img, weight4, conv_res4, mean_res4, var_res4], target=target)
    func5 = tvm.build(s5, [inputs_p3, mean_p3, var_p3, alpha3, beta3, inputs_p4, mean_p4, var_p4, alpha4, beta4, output], target=target)

    time_lst = []
    # warmup
    func1(img_tvm, weight1_tvm, inputs_p1_tvm, mean_p1_tvm, var_p1_tvm)
    func2(inputs_p1_tvm, mean_p1_tvm, var_p1_tvm, alpha1_tvm, beta1_tvm, weight2_tvm, inputs_p2_tvm, mean_p2_tvm, var_p2_tvm)
    func3(inputs_p2_tvm, mean_p2_tvm, var_p2_tvm, alpha2_tvm, beta2_tvm, weight3_tvm, inputs_p3_tvm, mean_p3_tvm, var_p3_tvm)
    func4(img_tvm,  weight4_tvm, inputs_p4_tvm, mean_p4_tvm, var_p4_tvm)
    func5(inputs_p3_tvm, mean_p3_tvm, var_p3_tvm, alpha3_tvm, beta3_tvm, inputs_p4_tvm, mean_p4_tvm, var_p4_tvm, alpha4_tvm, beta4_tvm, output_tvm)
    for i in range(repeat):
      begin = time.time()
      count = 0
      while (time.time() - begin < min_repeat_ms/1e3):
        func1(img_tvm, weight1_tvm, inputs_p1_tvm, mean_p1_tvm, var_p1_tvm)
        func2(inputs_p1_tvm, mean_p1_tvm, var_p1_tvm, alpha1_tvm, beta1_tvm, weight2_tvm, inputs_p2_tvm, mean_p2_tvm, var_p2_tvm)
        func3(inputs_p2_tvm, mean_p2_tvm, var_p2_tvm, alpha2_tvm, beta2_tvm, weight3_tvm, inputs_p3_tvm, mean_p3_tvm, var_p3_tvm)
        func4(img_tvm,  weight4_tvm, inputs_p4_tvm, mean_p4_tvm, var_p4_tvm)
        func5(inputs_p3_tvm, mean_p3_tvm, var_p3_tvm, alpha3_tvm, beta3_tvm, inputs_p4_tvm, mean_p4_tvm, var_p4_tvm, alpha4_tvm, beta4_tvm, output_tvm)
        ctx.sync()
        count += 1
      interval = time.time() - begin
      if count > 0:
        time_lst.append(interval/count)
      else:
        time_lst.append(float("inf"))

    time_lst = np.array(time_lst) * 1e3
    
    mean_time = time_lst.mean()
    dev = np.sqrt(time_lst.var())
    
    return output_tvm.asnumpy(), (mean_time, dev)

  return _func


if __name__ == "__main__":
  N = 4
  C = 512
  H = 28
  W = 28
  K = 256
  L = 1024
  dtype = "float32"
  target = "cuda"
  repeat = 20
  min_repeat_ms = 200

  img_np = np.random.uniform(-1, 1, [N, C, H, W]).astype(dtype)
  weight1_np = np.random.uniform(-1, 1, [K, C, 1, 1]).astype(dtype)
  weight2_np = np.random.uniform(-1, 1, [K, K, 3, 3]).astype(dtype)
  weight3_np = np.random.uniform(-1, 1, [L, K, 1, 1]).astype(dtype)
  weight4_np = np.random.uniform(-1, 1, [L, C, 1, 1]).astype(dtype)
  alpha1_np = np.random.uniform(-1, 1, [K]).astype(dtype)
  beta1_np = np.random.uniform(-1, 1, [K]).astype(dtype)
  alpha2_np = np.random.uniform(-1, 1, [K]).astype(dtype)
  beta2_np = np.random.uniform(-1, 1, [K]).astype(dtype)
  alpha3_np = np.random.uniform(-1, 1, [L]).astype(dtype)
  beta3_np = np.random.uniform(-1, 1, [L]).astype(dtype)
  alpha4_np = np.random.uniform(-1, 1, [L]).astype(dtype)
  beta4_np = np.random.uniform(-1, 1, [L]).astype(dtype)
  output_np = np.zeros([N, L, H//2, W//2]).astype(dtype)

  weights_np = [weight1_np, weight2_np, weight3_np, weight4_np]
  alphas_np = [alpha1_np, alpha2_np, alpha3_np, alpha4_np]
  betas_np = [beta1_np, beta2_np, beta3_np, beta4_np]

  pytorch_func = pytorch_mismatch_bottle_neck(
    N, C, H, W, K, L, dtype=dtype, target=target, training=True)

  tvm_func = tvm_conv_bn_relu_conv(
    N, C, H, W, K, L, dtype=dtype, target=target, training=True)

  torch_res, (torch_mean, torch_dev) = pytorch_func(
    img_np, weights_np, alphas_np, betas_np, output_np, repeat=repeat, min_repeat_ms=min_repeat_ms)

  tvm_res, (tvm_mean, tvm_dev) = tvm_func(
    img_np, weights_np, alphas_np, betas_np, output_np, repeat=repeat, min_repeat_ms=min_repeat_ms)

  tvm.testing.assert_allclose(tvm_res, torch_res, atol=1e-4, rtol=1e-5)

  print("PyTorch", target, "use time: ", torch_mean, "ms (dev=", torch_dev, "ms)")
  print("Mine", target, "use time: ", tvm_mean, "ms (dev=", tvm_dev, "ms)")

  
