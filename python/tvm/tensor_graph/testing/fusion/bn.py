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


def pytorch_conv_bn_relu_conv(N, C, H, W, K, stride=1, groups=1, dilation=1,
        dtype="float32", target="llvm", training=True):
  conv1 = torch.nn.Conv2d(C, K, 1, stride=stride, padding=0, dilation=dilation, groups=groups, bias=False)
  conv2 = torch.nn.Conv2d(K, K, 3, stride=stride, padding=1, dilation=dilation, groups=groups, bias=False)
  bn = torch.nn.BatchNorm2d(K, track_running_stats=training)
  relu = torch.relu 

  def _func(img_np, weight1_np, alpha_np, beta_np, weight2_np, output_np, repeat=10, min_repeat_ms=200):
    img_torch = torch.tensor(img_np)
    conv1.weight = torch.nn.Parameter(torch.tensor(weight1_np, requires_grad=training))
    bn.weight = torch.nn.Parameter(torch.tensor(alpha_np, requires_grad=training))
    bn.bias = torch.nn.Parameter(torch.tensor(beta_np, requires_grad=training))
    conv2.weight = torch.nn.Parameter(torch.tensor(weight2_np, requires_grad=training))
    conv_res = conv1(img_torch)
    bn_res = bn(conv_res)
    relu_res = relu(bn_res)
    output = conv2(relu_res)

    time_lst = []
    if target == "llvm":
      # warmup done before
      for i in range(repeat):
        begin = time.time()
        count = 0
        while (time.time() - begin < min_repeat_ms/1e3):
          conv_res = conv1(img_torch)
          bn_res = bn(conv_res)
          relu_res = relu(bn_res)
          output = conv2(relu_res)
          count += 1
        interval = time.time() - begin
        if count > 0:
          time_lst.append(interval/count)
        else:
          time_lst.append(float("inf"))
    elif target == "cuda":
      img_torch = img_torch.cuda(0)
      conv1.cuda(0)
      conv2.cuda(0)
      bn.cuda(0)
      # warmup
      conv_res = conv1(img_torch)
      bn_res = bn(conv_res)
      relu_res = relu(bn_res)
      output = conv2(relu_res)
      for i in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        conv_res = conv1(img_torch)
        bn_res = bn(conv_res)
        relu_res = relu(bn_res)
        output = conv2(relu_res)
        end.record()
        torch.cuda.synchronize()
        interval = start.elapsed_time(end)
        time_lst.append(interval/1e3)

    time_lst = np.array(time_lst) * 1e3
    mean_time = time_lst.mean()
    dev = np.sqrt(time_lst.var())

    return output.detach().cpu().numpy(), (mean_time, dev)

  return _func


def tvm_conv_bn_relu_conv(N, C, H, W, K, stride=1, groups=1, dilation=1,
        dtype="float32", target="llvm", training=True):
  img = tvm.te.placeholder([N, C, H, W], dtype=dtype, name="img")
  weight1 = tvm.te.placeholder([K, C, 1, 1], dtype=dtype, name="weight1", requires_grad=training)
  weight2 = tvm.te.placeholder([K, K, 3, 3], dtype=dtype, name="weight2", requires_grad=training)
  alpha = tvm.te.placeholder([K], dtype=dtype, name="alpha", requires_grad=training)
  beta = tvm.te.placeholder([K], dtype=dtype, name="beta", requires_grad=training)
  conv_res = conv2d_nchw(img, weight1, stride=stride, padding=0, dilation=dilation, groups=groups)
  (mean_res, var_res), (inputs_p, mean_p, var_p), bn_res = batch_norm(conv_res, alpha, beta)
  relu_res = relu(bn_res)
  output = conv2d_nchw(relu_res, weight2, stride=stride, padding=1, dilation=dilation, groups=groups)

  def _func(img_np, weight1_np, alpha_np, beta_np, weight2_np, output_np, repeat=10, min_repeat_ms=200):
    # naive schedule
    s1 = tvm.te.create_schedule([var_res.op])
    s2 = tvm.te.create_schedule([output.op])
    if target == "llvm":
      pass
    elif target == "cuda":
      pad1 = s1[conv_res].op.input_tensors[0]
      s1[pad1].compute_inline()
      pad2 = s2[output].op.input_tensors[0]
      s2[pad2].compute_inline()
      s2[relu_res].compute_inline()
      s2[bn_res].compute_inline()
      _, mean_t = s1[var_res].op.input_tensors
      
      ###########################
      # schedule for conv_res
      # create cache
      write_cache = s1.cache_write(conv_res, "local")
      weight = s1[write_cache].op.input_tensors[1]
      read_share_weight = s1.cache_read(weight, "shared", [write_cache])
      read_local_weight = s1.cache_read(read_share_weight, "local", [write_cache])
      read_share_inputs = s1.cache_read(pad1, "shared", [write_cache])
      read_local_inputs = s1.cache_read(read_share_inputs, "local", [write_cache])

      # tunable parameters
      b_factors = [4, 1, 1, 1]  # 4
      k_factors = [4, 1, 16, 1]  # 64
      p_factors = [7, 1, 8, 1]  # 56
      q_factors = [1, 1, 8, 7]  # 56
      rc_factors = [16, 1, 16]  # 256
      ry_factors = [1, 1, 1]  # 1
      rx_factors = [1, 1, 1]  # 1

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
      b, k, p, q = s1[conv_res].op.axis

      kernel_scope, b = s1[conv_res].split(b, nparts=1)

      bo, bi = s1[conv_res].split(b, nparts=b_factors[0])
      ko, ki = s1[conv_res].split(k, nparts=k_factors[0])
      po, pi = s1[conv_res].split(p, nparts=p_factors[0])
      qo, qi = s1[conv_res].split(q, nparts=q_factors[0])

      vbo, bi = s1[conv_res].split(bi, nparts=b_factors[1])
      vko, ki = s1[conv_res].split(ki, nparts=k_factors[1])
      vpo, pi = s1[conv_res].split(pi, nparts=p_factors[1])
      vqo, qi = s1[conv_res].split(qi, nparts=q_factors[1])

      tbo, bi = s1[conv_res].split(bi, nparts=b_factors[2])
      tko, ki = s1[conv_res].split(ki, nparts=k_factors[2])
      tpo, pi = s1[conv_res].split(pi, nparts=p_factors[2])
      tqo, qi = s1[conv_res].split(qi, nparts=q_factors[2])

      # reorder
      s1[conv_res].reorder(po, bo, ko, qo, vqo, vbo, vko, vpo, tbo, tko, tpo, tqo, bi, ki, pi, qi)

      # fuse
      bko = s1[conv_res].fuse(bo, ko)
      vbko = s1[conv_res].fuse(vbo, vko)
      tbko = s1[conv_res].fuse(tbo, tko)
      bki = s1[conv_res].fuse(bi, ki)

      # bind
      s1[conv_res].bind(bko, bz)
      s1[conv_res].bind(po, by)
      s1[conv_res].bind(qo, bx)
      s1[conv_res].bind(vbko, vz)
      s1[conv_res].bind(vpo, vy)
      s1[conv_res].bind(vqo, vx)
      s1[conv_res].bind(tbko, tz)
      s1[conv_res].bind(tpo, ty)
      s1[conv_res].bind(tqo, tx)

      # compute at write cache
      s1[write_cache].compute_at(s1[conv_res], tqo)

      rc, ry, rx = s1[write_cache].op.reduce_axis
      rco, rci = s1[write_cache].split(rc, nparts=rc_factors[0])
      rcm, rci = s1[write_cache].split(rci, nparts=rc_factors[1])
      ryo, ryi = s1[write_cache].split(ry, nparts=ry_factors[0])
      rym, ryi = s1[write_cache].split(ryi, nparts=ry_factors[1])
      rxo, rxi = s1[write_cache].split(rx, nparts=rx_factors[0])
      rxm, rxi = s1[write_cache].split(rxi, nparts=rx_factors[1])
      a, b, c, d = s1[write_cache].op.axis
      s1[write_cache].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, a, b, c, d)

      # compute at read cache
      s1[read_share_weight].compute_at(s1[write_cache], rxm)
      s1[read_local_weight].compute_at(s1[write_cache], rxi)
      s1[read_share_inputs].compute_at(s1[write_cache], rxm)
      s1[read_local_inputs].compute_at(s1[write_cache], rxi)

      # cooperative fetching
      for cache in [read_share_inputs, read_share_weight]:
          cb, ck, ch, cw = s1[cache].op.axis
          fused = s1[cache].fuse(cb, ck, ch, cw)
          fused, bindx = s1[cache].split(fused, factor=q_factors[2])
          fused, bindy = s1[cache].split(fused, factor=p_factors[2])
          fused, bindz = s1[cache].split(fused, factor=b_factors[2] * k_factors[2])       
          
          s1[cache].bind(bindx, tx)
          s1[cache].bind(bindy, ty)
          s1[cache].bind(bindz, tz)
      
      s1[conv_res].pragma(kernel_scope, 'auto_unroll_max_step', 1500)
      s1[conv_res].pragma(kernel_scope, 'unroll_explicit', 0)

      ###########################
      # schedule mean
      c_factors = [8, 8, 1, 1]  # 64
      rn_factors = [4, 1, 1, 1]  # 4
      rh_factors = [8, 1, 1, 7]  # 56
      rw_factors = [56, 1, 1, 1]  # 56
      c, = s1[mean_t].op.axis
      rn, rh, rw = s1[mean_t].op.reduce_axis

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

      # co, c = s1[mean_t].split(c, nparts=c_factors[0])
      # rno, rn = s1[mean_t].split(rn, nparts=rn_factors[0])
      # rho, rh = s1[mean_t].split(rh, nparts=rh_factors[0])
      rwo, rw = s1[mean_t].split(rw, nparts=rw_factors[0])
      mean_t_rf = s1.rfactor(mean_t, rwo)
      rho, rh = s1[mean_t_rf].split(rh, nparts=rh_factors[0])
      # mean_t_rf_rf = s1.rfactor(mean_t_rf, rho)
      # rno, rn = s1[mean_t_rf].split(s[mean_t_rf].op.reduce_axis[0], nparts=rn_factors[0])
      # rho, rh = s1[mean_t].split(s[mean_t_rf].op.reduce_axis[1], nparts=rh_factors[0])

      # cm, c = s1[mean_t].split(c, nparts=c_factors[1])
      # rnm, rn = s1[mean_t].split(rn, nparts=rn_factors[1])
      # rhm, rh = s1[mean_t].split(rh, nparts=rh_factors[1])
      # rwm, rw = s1[mean_t].split(rw, nparts=rw_factors[1])

      # ct, ci = s1[mean_t].split(c, nparts=c_factors[2])
      # rnt, rni = s1[mean_t].split(rn, nparts=rn_factors[2])
      # rht, rhi = s1[mean_t].split(rh, nparts=rh_factors[2])
      # rwt, rwi = s1[mean_t].split(rw, nparts=rw_factors[2])

      # s1[mean_t].reorder(co, cm, rno, rho, rwo, ct, ci, rnm, rhm, rwm, rnt, rht, rwt, rni, rhi, rwi)

      # s1[mean_t].bind(co, by)
      # s1[mean_t].bind(cm, bx)
      s1[mean_t].bind(s1[mean_t].op.axis[0], bx)
      s1[mean_t_rf].compute_at(s1[mean_t], s1[mean_t].op.reduce_axis[0])
      s1[mean_t].bind(s1[mean_t].op.reduce_axis[0], tx)
      # s1[mean_t_rf_rf].compute_at(s1[mean_t_rf], s1[mean_t_rf].op.reduce_axis[0])
      # s1[mean_t_rf].bind(s1[mean_t_rf].op.reduce_axis[0], ty)

      ###########################
      # schedule var
      c_factors = [8, 8, 1, 1]  # 64
      rn_factors = [4, 1, 1, 1]  # 4
      rh_factors = [4, 2, 1, 7]  # 56
      rw_factors = [56, 2, 1, 1]  # 56
      c, = s1[var_res].op.axis
      rn, rh, rw = s1[var_res].op.reduce_axis

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

      rwo, rwi = s1[var_res].split(rw, nparts=rw_factors[0])
      var_res_rf = s1.rfactor(var_res, rwo)
      s1[var_res].bind(s1[var_res].op.axis[0], bx)
      s1[var_res_rf].compute_at(s1[var_res], s1[var_res].op.reduce_axis[0])
      s1[var_res].bind(s1[var_res].op.reduce_axis[0], tx)

      # co, c = s1[var_res].split(c, nparts=c_factors[0])
      # rno, rn = s1[var_res].split(rn, nparts=rn_factors[0])
      # rho, rh = s1[var_res].split(rh, nparts=rh_factors[0])
      # rwo, rw = s1[var_res].split(rw, nparts=rw_factors[0])

      # cm, c = s1[var_res].split(c, nparts=c_factors[1])
      # rnm, rn = s1[var_res].split(rn, nparts=rn_factors[1])
      # rhm, rh = s1[var_res].split(rh, nparts=rh_factors[1])
      # rwm, rw = s1[var_res].split(rw, nparts=rw_factors[1])

      # ct, ci = s1[var_res].split(c, nparts=c_factors[2])
      # rnt, rni = s1[var_res].split(rn, nparts=rn_factors[2])
      # rht, rhi = s1[var_res].split(rh, nparts=rh_factors[2])
      # rwt, rwi = s1[var_res].split(rw, nparts=rw_factors[2])

      # s1[var_res].reorder(co, cm, rno, rho, rwo, ct, ci, rnm, rhm, rwm, rnt, rht, rwt, rni, rhi, rwi)

      # s1[var_res].bind(co, by)
      # s1[var_res].bind(cm, bx)

      # s1[conv_res].compute_at(s1[var_res], cm)
      # s1[mean_t].compute_at(s1[var_res_rf], s1[var_res_rf].op.axis[0])
      
      ###########################
      # schedule output
      write_cache = s2.cache_write(output, "local")
      weight = s2[write_cache].op.input_tensors[1]
      read_share_weight = s2.cache_read(weight, "shared", [write_cache])
      read_local_weight = s2.cache_read(read_share_weight, "local", [write_cache])
      read_share_inputs = s2.cache_read(pad2, "shared", [write_cache])
      read_local_inputs = s2.cache_read(read_share_inputs, "local", [write_cache])

      # tunable parameters
      b_factors = [4, 1, 1, 1]  # 4
      k_factors = [1, 1, 16, 4]  # 64
      p_factors = [7, 1, 8, 1]  # 56
      q_factors = [7, 1, 8, 1]  # 56
      rc_factors = [8, 1, 8]  # 64
      ry_factors = [1, 3, 1]  # 3
      rx_factors = [1, 1, 3]  # 3

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
      b, k, p, q = s2[output].op.axis

      kernel_scope, b = s2[output].split(b, nparts=1)

      bo, bi = s2[output].split(b, nparts=b_factors[0])
      ko, ki = s2[output].split(k, nparts=k_factors[0])
      po, pi = s2[output].split(p, nparts=p_factors[0])
      qo, qi = s2[output].split(q, nparts=q_factors[0])

      vbo, bi = s2[output].split(bi, nparts=b_factors[1])
      vko, ki = s2[output].split(ki, nparts=k_factors[1])
      vpo, pi = s2[output].split(pi, nparts=p_factors[1])
      vqo, qi = s2[output].split(qi, nparts=q_factors[1])

      tbo, bi = s2[output].split(bi, nparts=b_factors[2])
      tko, ki = s2[output].split(ki, nparts=k_factors[2])
      tpo, pi = s2[output].split(pi, nparts=p_factors[2])
      tqo, qi = s2[output].split(qi, nparts=q_factors[2])

      # reorder
      s2[output].reorder(po, bo, ko, qo, vqo, vbo, vko, vpo, tbo, tko, tpo, tqo, bi, ki, pi, qi)

      # fuse
      bko = s2[output].fuse(bo, ko)
      vbko = s2[output].fuse(vbo, vko)
      tbko = s2[output].fuse(tbo, tko)
      bki = s2[output].fuse(bi, ki)

      # bind
      s2[output].bind(bko, bz)
      s2[output].bind(po, by)
      s2[output].bind(qo, bx)
      s2[output].bind(vbko, vz)
      s2[output].bind(vpo, vy)
      s2[output].bind(vqo, vx)
      s2[output].bind(tbko, tz)
      s2[output].bind(tpo, ty)
      s2[output].bind(tqo, tx)

      # compute at write cache
      s2[write_cache].compute_at(s2[output], tqo)

      rc, ry, rx = s2[write_cache].op.reduce_axis
      rco, rci = s2[write_cache].split(rc, nparts=rc_factors[0])
      rcm, rci = s2[write_cache].split(rci, nparts=rc_factors[1])
      ryo, ryi = s2[write_cache].split(ry, nparts=ry_factors[0])
      rym, ryi = s2[write_cache].split(ryi, nparts=ry_factors[1])
      rxo, rxi = s2[write_cache].split(rx, nparts=rx_factors[0])
      rxm, rxi = s2[write_cache].split(rxi, nparts=rx_factors[1])
      a, b, c, d = s2[write_cache].op.axis
      s2[write_cache].reorder(rco, ryo, rxo, rcm, rym, rxm, rci, ryi, rxi, a, b, c, d)

      # compute at read cache
      s2[read_share_weight].compute_at(s2[write_cache], rxm)
      s2[read_local_weight].compute_at(s2[write_cache], rxi)
      s2[read_share_inputs].compute_at(s2[write_cache], rxm)
      s2[read_local_inputs].compute_at(s2[write_cache], rxi)

      # cooperative fetching
      for cache in [read_share_inputs, read_share_weight]:
          cb, ck, ch, cw = s2[cache].op.axis
          fused = s2[cache].fuse(cb, ck, ch, cw)
          fused, bindx = s2[cache].split(fused, factor=q_factors[2])
          fused, bindy = s2[cache].split(fused, factor=p_factors[2])
          fused, bindz = s2[cache].split(fused, factor=b_factors[2] * k_factors[2])       
          
          s2[cache].bind(bindx, tx)
          s2[cache].bind(bindy, ty)
          s2[cache].bind(bindz, tz)
      
      s2[output].pragma(kernel_scope, 'auto_unroll_max_step', 1500)
      s2[output].pragma(kernel_scope, 'unroll_explicit', 0)

    
    ctx = tvm.context(target, 0)
    img_tvm = tvm.nd.array(img_np, ctx)
    weight1_tvm = tvm.nd.array(weight1_np, ctx)
    weight2_tvm = tvm.nd.array(weight2_np, ctx)
    inputs_p_tvm = tvm.nd.array(np.zeros([N, K, H, W]).astype(img_tvm.dtype), ctx)
    mean_p_tvm = tvm.nd.array(np.zeros([K]).astype(img_tvm.dtype), ctx)
    var_p_tvm = tvm.nd.array(np.zeros([K]).astype(img_tvm.dtype), ctx)
    alpha_tvm = tvm.nd.array(alpha_np, ctx)
    beta_tvm = tvm.nd.array(beta_np, ctx)
    output_tvm = tvm.nd.array(output_np, ctx)

    print(tvm.lower(s1, [img, weight1, conv_res, mean_res, var_res], simple_mode=True))
    print(tvm.lower(s2, [inputs_p, mean_p, var_p, alpha, beta, weight2, output], simple_mode=True))
    func1 = tvm.build(s1, [img, weight1, conv_res, mean_res, var_res], target=target)
    func2 = tvm.build(s2, [inputs_p, mean_p, var_p, alpha, beta, weight2, output], target=target)
    # func1(img_tvm, weight1_tvm, inputs_p_tvm, mean_p_tvm, var_p_tvm)
    # func2(inputs_p_tvm, mean_p_tvm, var_p_tvm, alpha_tvm, beta_tvm, weight2_tvm, output_tvm)

    # evaluator = func.time_evaluator(func.entry_name, ctx, repeat=repeat, min_repeat_ms=min_repeat_ms)
    # time_lst = np.array(evaluator(img_tvm, weight1_tvm, alpha_tvm, beta_tvm, weight2_tvm, output_tvm).results) * 1e3
    # mean_cost = time_lst.mean()
    # dev = np.sqrt(time_lst.var())

    time_lst = []
    # warmup
    func1(img_tvm, weight1_tvm, inputs_p_tvm, mean_p_tvm, var_p_tvm)
    func2(inputs_p_tvm, mean_p_tvm, var_p_tvm, alpha_tvm, beta_tvm, weight2_tvm, output_tvm)
    for i in range(repeat):
      begin = time.time()
      count = 0
      while (time.time() - begin < min_repeat_ms/1e3):
        func1(img_tvm, weight1_tvm, inputs_p_tvm, mean_p_tvm, var_p_tvm)
        # ctx.sync()
        func2(inputs_p_tvm, mean_p_tvm, var_p_tvm, alpha_tvm, beta_tvm, weight2_tvm, output_tvm)
        ctx.sync()
        count += 1
      interval = time.time() - begin
      if count > 0:
        time_lst.append(interval/count)
      else:
        time_lst.append(float("inf"))
    
    # evaluator1 = func1.time_evaluator(func1.entry_name, ctx, number=1, repeat=1)
    # time_lst1 = np.array(evaluator1(img_tvm, weight1_tvm, inputs_p_tvm, mean_p_tvm, var_p_tvm).results) * 1e3
    # evaluator2 = func2.time_evaluator(func2.entry_name, ctx, number=1, repeat=1)
    # time_lst2 = np.array(evaluator2(inputs_p_tvm, mean_p_tvm, var_p_tvm, alpha_tvm, beta_tvm, weight2_tvm, output_tvm).results) * 1e3

    # print(time_lst1)
    # print(time_lst2)
    # time_lst = time_lst1 + time_lst2

    time_lst = np.array(time_lst) * 1e3
    
    mean_time = time_lst.mean()
    dev = np.sqrt(time_lst.var())
    
    return output_tvm.asnumpy(), (mean_time, dev)

  return _func




if __name__ == "__main__":
  N = 4
  C = 256
  H = 56
  W = 56
  K = 64
  stride = 1
  groups = 1
  dilation = 1
  dtype = "float32"
  target = "cuda"
  repeat = 10
  min_repeat_ms = 200

  img_np = np.random.uniform(-1, 1, [N, C, H, W]).astype(dtype)
  weight1_np = np.random.uniform(-1, 1, [K, C, 1, 1]).astype(dtype)
  weight2_np = np.random.uniform(-1, 1, [K, K, 3, 3]).astype(dtype)
  alpha_np = np.random.uniform(-1, 1, [K]).astype(dtype)
  beta_np = np.random.uniform(-1, 1, [K]).astype(dtype)
  output_np = np.zeros([N, K, H, W]).astype(dtype)

  pytorch_func = pytorch_conv_bn_relu_conv(
    N, C, H, W, K, stride=stride, groups=groups, dilation=dilation, dtype=dtype, target=target, training=True)

  tvm_func = tvm_conv_bn_relu_conv(
    N, C, H, W, K, stride=stride, groups=groups, dilation=dilation, dtype=dtype, target=target, training=True)

  torch_res, (torch_mean, torch_dev) = pytorch_func(
    img_np, weight1_np, alpha_np, beta_np, weight2_np, output_np, repeat=repeat, min_repeat_ms=min_repeat_ms)

  tvm_res, (tvm_mean, tvm_dev) = tvm_func(
    img_np, weight1_np, alpha_np, beta_np, weight2_np, output_np, repeat=repeat, min_repeat_ms=min_repeat_ms)

  tvm.testing.assert_allclose(tvm_res, torch_res, atol=1e-4, rtol=1e-5)

  print("PyTorch", target, "use time: ", torch_mean, "ms (dev=", torch_dev, "ms)")
  print("Mine", target, "use time: ", tvm_mean, "ms (dev=", tvm_dev, "ms)")

  
