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


def get_depthwise_conv2d(ic, nh, nw, kh, kw, ph=0, pw=0, sh=1, sw=1):
  """Convolution

  ic : number of channels for both input and output
  nh, nw : input width and height
  kh, kw : kernel width and height
  ph, pw : height and width padding sizes, default 0
  sh, sw : height and width strides, default 1
  """
  # reduction axes
  rkh = te.reduce_axis((0, kh), name='rkh')
  rkw = te.reduce_axis((0, kw), name='rkw')
  # output height and weights
  oh = conv_out_size(nh, kh, ph, sh)
  ow = conv_out_size(nw, kw, pw, sw)
  # pad X and then compute Y
  X = te.placeholder((ic, nh, nw), name='X')
  K = te.placeholder((ic, 1, kh, kw), name='K')
  PaddedX = padding(X, ph, pw) if ph * pw != 0 else X
  Y = te.compute(
      (ic, oh, ow),
      lambda c, i, j: te.sum(
          (PaddedX[c, i*sh+rkh, j*sw+rkw] * K[c, 0, rkh, rkw]),
          axis=[rkh, rkw]), name='Y')

  return X, K, Y, PaddedX


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

Feature = namedtuple('Feature', BUFFER_ACCESS_FEATURE_KEYS)

def build_structured_feature(d: list[dict[str, Feature]]):
  from copy import deepcopy
  d = deepcopy(d)
  for dd in d:
    for k, v in dd.items():
      dd[k] = {kk: getattr(v, kk) for kk in BUFFER_ACCESS_FEATURE_KEYS}
  return d
