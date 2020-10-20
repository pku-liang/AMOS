from enum import Enum
from collections import namedtuple

import tvm
import tvm.te as te
import tvm.tg as tg
import tvm.tir as tir

import numpy as np


def pprint_dict(d):
  import json
  print(json.dumps(d, indent=2, sort_keys=False))


def get_vector_add(n):
  """TVM expression for vector add"""
  A = te.placeholder((n,), name='a')
  B = te.placeholder((n,), name='b')
  C = te.compute(A.shape, lambda i: A[i] + B[i], name='c')
  return A, B, C


def get_gemm(n, m, l):
  """Return the computing expression of matrix multiplication
  A : n x l matrix
  B : l x m matrix
  C : n x m matrix with C = A B
  """
  k = te.reduce_axis((0, l), name='k')
  A = te.placeholder((n, l), name='A')
  B = te.placeholder((l, m), name='B')
  C = te.compute((n, m),
                  lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
                  name='C')
  return A, B, C


def get_padding(X, ph, pw, val=0):
  """Pad X with the given value in 2-D

  ph, pw : height and width padding
  val : padding value, default 0
  """
  assert len(X.shape) >= 2
  nh, nw = X.shape[-2], X.shape[-1]
  return te.compute(
          (*X.shape[0:-2], nh+ph*2, nw+pw*2),
          lambda *i: te.if_then_else(
              te.any(i[-2]<ph, i[-2]>=nh+ph, i[-1]<pw, i[-1]>=nw+pw),
              val, X[i[:-2]+(i[-2]-ph, i[-1]-pw)]),
          name='PaddedX')


def conv_out_size(n, k, p, s):
  """Compute the output size by given input size n (width or height),
  kernel size k, padding p, and stride s
  Return output size (width or height)
  """
  return (n - k + 2 * p) // s + 1


def get_conv2d(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
  """Convolution

  oc, ic : output and input channels
  nh, nw : input width and height
  kh, kw : kernel width and height
  ph, pw : height and width padding sizes, default 0
  sh, sw : height and width strides, default 1
  """
  # reduction axes
  ric = te.reduce_axis((0, ic), name='ric')
  rkh = te.reduce_axis((0, kh), name='rkh')
  rkw = te.reduce_axis((0, kw), name='rkw')
  # output height and weights
  oh = conv_out_size(nh, kh, ph, sh)
  ow = conv_out_size(nw, kw, pw, sw)
  # pad X and then compute Y
  X = te.placeholder((ic, nh, nw), name='X')
  K = te.placeholder((oc, ic, kh, kw), name='K')
  PaddedX = get_padding(X, ph, pw) if ph * pw != 0 else X
  Y = te.compute(
      (oc, oh, ow),
      lambda c, i, j: te.sum(
          PaddedX[ric, i*sh+rkh, j*sw+rkw] * K[c, ric, rkh, rkw],
          axis=[ric, rkh, rkw]), name='Y')
  return X, K, Y, PaddedX


def get_conv2d_unroll(oc, ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
  """Convolution

  oc, ic : output and input channels
  nh, nw : input width and height
  kh, kw : kernel width and height
  ph, pw : height and width padding sizes, default 0
  sh, sw : height and width strides, default 1
  """
  # reduction axes
  ric = te.reduce_axis((0, ic), name='ric')
  rkh = te.reduce_axis((0, kh), name='rkh')
  rkw = te.reduce_axis((0, kw), name='rkw')
  # output height and weights
  oh = conv_out_size(nh, kh, ph, sh)
  ow = conv_out_size(nw, kw, pw, sw)
  # pad X and then compute Y
  X = te.placeholder((ic, nh, nw), name='X')
  K = te.placeholder((oc, ic, kh, kw), name='K')
  PaddedX = get_padding(X, ph, pw) if ph * pw != 0 else X
  Y = te.compute(
      (oc, oh, ow),
      lambda c, i, j: te.sum(
          PaddedX[ric, i*sh+rkh, j*sw+rkw] * K[c, ric, rkh, rkw],
          axis=[ric, rkh, rkw]), name='Y')
  
  sch = te.create_schedule(Y.op)
  sch[Y].pragma(Y.op.axis[0], 'auto_unroll_max_step', 4)
  return sch, (X, K, Y, PaddedX)


def depthwise_conv_pack(c, nh, nw, kh, kw, ph, pw, tc):
    """Pack data and weight for depthwise convolution
       Note that the input channel of kernel is specified as 1,
       and the output channel of kernel equals the input channel of data

    c : input channel of data and output channel of kernel
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    tc : the tiling size of channels
    """
    X = te.placeholder((c, nh, nw), name='X')
    K = te.placeholder((c, 1, kh, kw), name='K')
    PaddedX = get_padding(X, ph, pw) if ph * pw != 0 else X
    # make sure the channel tiling is valid
    if c < tc:
        tc = c
    assert c % tc == 0
    # pack X and K
    PackedX = te.compute(
        (c//tc, nh+ph*2, nw+pw*2, tc),
        lambda c_out, x, y, c_in: PaddedX[c_out*tc + c_in, x, y],
        name='PackedX')
    PackedK = te.compute(
        (c//tc, 1, kh, kw, 1, tc),
        lambda c_out, _, x, y, __, c_in: K[
            c_out*tc + c_in, 0, x, y],
        name='PackedK')
    return X, K, PaddedX, PackedX, PackedK


def get_depthwise_conv2d(c, nh, nw, kh, kw, ph, pw, sh, sw, tc):
    """depthwise conv

    c : number of channels for both input and output.
    nh, nw : input width and height
    kh, kw : kernel width and height
    ph, pw : height and width padding
    sh, sw : height and width strides
    tc : the tiling sizes of channels
    """
    X, K, PaddedX, PackedX, PackedK = depthwise_conv_pack(
        c, nh, nw, kh, kw, ph, pw, tc)
    # reduction axes
    rkh = te.reduce_axis((0, kh), name='rkh')
    rkw = te.reduce_axis((0, kw), name='rkw')
    # output height and weights
    oh = conv_out_size(nh, kh, ph, sh)
    ow = conv_out_size(nw, kw, pw, sw)
    # compute Y in the packed layout
    PackedY = te.compute(
        (c//tc, oh, ow, tc),
        lambda c_out, x, y, c_in: te.sum(
            (PackedX[c_out, x*sh+rkh, y*sw+rkw, c_in] *
             PackedK[c_out, 0, rkh, rkw, 0, c_in]),
            axis=[rkh, rkw]), name='PackedY')

    # Unpack the result
    Y = te.compute((c, oh, ow),
                    lambda c, x, y: PackedY[c//tc, x, y, c%tc],
                    name='Y')
    return X, K, Y, PaddedX, PackedX, PackedK, PackedY


def get_feature(inputs, outputs, sch=None, target='llvm'):
  sch = sch or te.create_schedule([o.op for o in outputs])
  target = tvm.target.create(target)
  args = [*inputs, *outputs]

  print(tvm.lower(sch, args, simple_mode=True))

  features = tg.auto_schedule.get_feature(sch, args, target, flatten=True)
  features = np.array(features)

  structured_features = tg.auto_schedule.get_feature(sch, args, target, flatten=False)
  return features, structured_features


def nelem(tensor: te.tensor.Tensor):
  return np.prod([t.value for t in tensor.shape])


AccessType = Enum('AccessType', ['kNone', 'kRead', 'kWrite', 'kReadWrite'])
ReuseType = Enum('ReuseType', ['kNoReuse', 'kLoopMultipleRead', 'kSerialMultipleRead', 'kBothReuse'])


def structural_equal(feature, feature_ref, check_features:list):
  assert len(feature) == len(feature_ref), \
    f'mismatch in len(feature): {len(feature)} != {len(feature_ref)}(ref)'
  for fea, fea_ref in zip(feature, feature_ref):
    for buf_key in fea.keys():
      if buf_key == '_stmt_': continue
      for fea_key in check_features:
        res, ref = fea[buf_key][fea_key], fea_ref[buf_key][fea_key]
        enum_feas = ['access_type', 'reuse_type']
        assert res == str(ref) if fea_key in enum_feas else res == ref, \
          f'mismatch in {key}: {res} != {ref}(ref)'


BUFFER_ACCESS_FEATURE_KEYS = ['access_type', 'bytes', 'unique_bytes', 'lines',
  'unique_lines', 'reuse_type', 'reuse_distance', 'reuse_counter', 'stride']


Feature = namedtuple('Feature', ['access_type', 'bytes', 'unique_bytes', 'lines',
  'unique_lines', 'reuse_type', 'reuse_distance', 'reuse_counter', 'stride'])


def build_structured_feature(d):
  from copy import deepcopy
  d = deepcopy(d)
  for dd in d:
    for k, v in dd.items():
      dd[k] = {kk: getattr(v, kk) for kk in BUFFER_ACCESS_FEATURE_KEYS}
  return d


def conv2d_gpu_default(oc, ic, n, k, p, s):
  X, K, Y, PaddedX = get_conv2d(oc, ic, n, n, k, k, p, p, s, s)
  sch = te.create_schedule(Y.op)
  if p != 0: sch[PaddedX].compute_inline()
  _, y, x = sch[Y].op.axis
  sch[Y].bind(y, te.thread_axis("blockIdx.x"))
  sch[Y].bind(x, te.thread_axis("threadIdx.x"))
  print(tvm.lower(sch, [X, K, Y], simple_mode=True))
  return sch, (X, K, Y)


def split_axis(factors, sch, op, axis):
  """Splitting an axis into factors

  Parameters
  ----------
  factors: array of integers
      The factors that the split applies
  sch: tvm.te.schedule.Schedule
      The tvm schedule
  op: tvm.te.tensor.Operation
      The stage to be applied
  axis: tvm.te.schedule.IterVar
      axis to split

  Returns
  -------
  axes : list of Axis
      The transformed axes.
  """
  ret = []
  for i in range(0, len(factors)):
      ax0, ax1 = sch[op].split(axis, factor=int(np.prod(factors[i:])))
      ret.append(ax0)
      axis = ax1
  return ret + [axis]


def conv2d_gpu_tiled(oc, ic, n, k, p, s):
  tile_c = [4, 8]
  tile_h = [2, 2]
  tile_w = [16, 4]
  tile_rc = [1, 1]
  tile_rh = [1, 1]
  tile_rw = [1, 3]

  X, K, Y, PaddedX = get_conv2d(oc, ic, n, n, k, k, p, p, s, s)
  sch = te.create_schedule(Y.op)
  if p != 0: sch[PaddedX].compute_inline()

  YL = sch.cache_write(Y, 'local')

  # create cache stage
  XX = sch.cache_read(PaddedX, 'shared', [YL])
  KK = sch.cache_read(K, 'shared', [YL])
  XL = sch.cache_read(XX, 'local', [YL])
  KL = sch.cache_read(KK, 'local', [YL])

  c, h, w = sch[Y].op.axis

  bc, tc, ic = split_axis(tile_c, sch, Y, c)
  bh, th, ih = split_axis(tile_h, sch, Y, h)
  bw, tw, iw = split_axis(tile_w, sch, Y, w)

  sch[Y].bind(bc, te.thread_axis("blockIdx.z"))
  sch[Y].bind(bh, te.thread_axis("blockIdx.y"))
  sch[Y].bind(bw, te.thread_axis("blockIdx.x"))
  sch[Y].bind(tc, te.thread_axis("threadIdx.z"))
  sch[Y].bind(th, te.thread_axis("threadIdx.y"))
  sch[Y].bind(tw, te.thread_axis("threadIdx.x"))
  sch[Y].reorder(bc, bh, bw, tc, th, tw, ic, ih, iw)

  sch[YL].compute_at(sch[Y], tw)

  # tile reduction axes
  c, h, w = sch[YL].op.axis
  rc, rh, rw = sch[YL].op.reduce_axis
  rco, rcm, rci = split_axis(tile_rc, sch, YL, rc)
  rho, rhm, rhi = split_axis(tile_rh, sch, YL, rh)
  rwo, rwm, rwi = split_axis(tile_rw, sch, YL, rw)
  sch[YL].reorder(rco, rho, rwo, rcm, rhm, rwm, rci, rhi, rwi, c, h, w)

  sch[XX].compute_at(sch[YL], rwo)
  sch[KK].compute_at(sch[YL], rwo)
  sch[XL].compute_at(sch[YL], rwm)
  sch[KL].compute_at(sch[YL], rwm)

  # cooperative fetching
  for load in [XX, KK]:
      args = sch[load].op.axis
      fused = sch[load].fuse(*args)
      # align thread layout
      tz, fused = sch[load].split(fused, nparts=tile_c[0])
      ty, fused = sch[load].split(fused, nparts=tile_h[0])
      tx, _ = sch[load].split(fused, nparts=tile_w[0])
      sch[load].bind(tz, te.thread_axis("threadIdx.z"))
      sch[load].bind(ty, te.thread_axis("threadIdx.y"))
      sch[load].bind(tx, te.thread_axis("threadIdx.x"))

  return sch, (X, K, Y)


def conv2d_gpu_tiled_vthread(oc, ic, n, k, p, s):
  tile_c = [4, 8]
  tile_h = [2, 2]
  tile_w = [16, 4]
  tile_rc = [1, 1]
  tile_rh = [1, 1]
  tile_rw = [1, 3]

  X, K, Y, PaddedX = get_conv2d(oc, ic, n, n, k, k, p, p, s, s)
  sch = te.create_schedule(Y.op)
  if p != 0: sch[PaddedX].compute_inline()

  YL = sch.cache_write(Y, 'local')

  # create cache stage
  XX = sch.cache_read(PaddedX, 'shared', [YL])
  KK = sch.cache_read(K, 'shared', [YL])
  XL = sch.cache_read(XX, 'local', [YL])
  KL = sch.cache_read(KK, 'local', [YL])

  c, h, w = sch[Y].op.axis

  bc, vc, tc, ic = split_axis(tile_c, sch, Y, c)
  bh, vh, th, ih = split_axis(tile_h, sch, Y, h)
  bw, vw, tw, iw = split_axis(tile_w, sch, Y, w)

  sch[Y].bind(bc, te.thread_axis("blockIdx.z"))
  sch[Y].bind(bh, te.thread_axis("blockIdx.y"))
  sch[Y].bind(bw, te.thread_axis("blockIdx.x"))
  sch[Y].bind(vc, te.thread_axis("vthread"))
  sch[Y].bind(vh, te.thread_axis("vthread"))
  sch[Y].bind(vw, te.thread_axis("vthread"))
  sch[Y].bind(tc, te.thread_axis("threadIdx.z"))
  sch[Y].bind(th, te.thread_axis("threadIdx.y"))
  sch[Y].bind(tw, te.thread_axis("threadIdx.x"))
  sch[Y].reorder(bc, bh, bw, vc, vh, vw, tc, th, tw, ic, ih, iw)

  sch[YL].compute_at(sch[Y], tw)

  # tile reduction axes
  c, h, w = sch[YL].op.axis
  rc, rh, rw = sch[YL].op.reduce_axis
  rco, rcm, rci = split_axis(tile_rc, sch, YL, rc)
  rho, rhm, rhi = split_axis(tile_rh, sch, YL, rh)
  rwo, rwm, rwi = split_axis(tile_rw, sch, YL, rw)
  sch[YL].reorder(rco, rho, rwo, rcm, rhm, rwm, rci, rhi, rwi, c, h, w)

  sch[XX].compute_at(sch[YL], rwo)
  sch[KK].compute_at(sch[YL], rwo)
  sch[XL].compute_at(sch[YL], rwm)
  sch[KL].compute_at(sch[YL], rwm)

  # cooperative fetching
  for load in [XX, KK]:
      args = sch[load].op.axis
      fused = sch[load].fuse(*args)
      # align thread layout
      tz, fused = sch[load].split(fused, nparts=tile_c[1])
      ty, fused = sch[load].split(fused, nparts=tile_h[1])
      tx, _ = sch[load].split(fused, nparts=tile_w[1])
      sch[load].bind(tz, te.thread_axis("threadIdx.z"))
      sch[load].bind(ty, te.thread_axis("threadIdx.y"))
      sch[load].bind(tx, te.thread_axis("threadIdx.x"))

  return sch, (X, K, Y)


def depthwise_cached_block(c, n, k, p, s, tc, tw):
    X, K, Y, PaddedX, PackedX, PackedK, PackedY = get_depthwise_conv2d(
        c, n, n, k, k, p, p, s, s, tc)
    sch = te.create_schedule(Y.op)

    CachedY = sch.cache_write(PackedY, 'global')

    c_out, h, w, c_in = sch[PackedY].op.axis
    w_out, w_in = sch[PackedY].split(w, factor=tw)
    sch[PackedY].reorder(c_out, h, w_out, w_in, c_in)
    c_out_h = sch[PackedY].fuse(c_out, h)
    sch[PackedY].parallel(c_out_h)
    sch[CachedY].compute_at(sch[PackedY], w_out)

    cc_out, ch, cw, cc_in = sch[CachedY].op.axis
    kh, kw = sch[CachedY].op.reduce_axis
    sch[CachedY].reorder(cc_out, ch, kh, kw, cw, cc_in)
    sch[CachedY].vectorize(cc_in)
    sch[CachedY].unroll(cw)

    # Schedule the padding by adding thread-level parallelism
    if PaddedX != X:
        sch[PaddedX].parallel(PaddedX.op.axis[0])
    # Optimize the packing of X and K
    sch[PackedX].parallel(sch[PackedX].fuse(*PackedX.op.axis[0:2]))
    sch[PackedX].unroll(PackedX.op.axis[-1])
    sch[PackedK].parallel(sch[PackedK].fuse(*PackedK.op.axis[0:2]))
    sch[PackedK].unroll(PackedK.op.axis[-1])
    # Optimize the unpacking of Y
    sch[Y].parallel(sch[Y].fuse(*Y.op.axis[0:2]))
    sch[Y].unroll(Y.op.axis[-1])
    return sch, (X, K, Y)
