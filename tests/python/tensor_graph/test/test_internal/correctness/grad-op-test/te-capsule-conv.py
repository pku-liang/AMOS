from tvm import testing
import tvm
import numpy as np 
import torch


N = 2
nC = 16
H = 14
W = 14
K = 8
R = 3
S = 3
X = 4
Y = 4
Z = 4

st = 1

P = (H - R + 1) // st
Q = (W - S + 1) // st

dtype = "float32"

A = tvm.te.placeholder([N, nC, H, W, X, Z], dtype=dtype, name="A")
B = tvm.te.placeholder([K, nC, R, S, Z, Y], dtype=dtype, name="B")
c = tvm.te.reduce_axis([0, nC], name="c")
r = tvm.te.reduce_axis([0, R], name="r")
s = tvm.te.reduce_axis([0, S], name="s")
z = tvm.te.reduce_axis([0, Z], name="z")
C = tvm.te.compute([N, K, P, Q, X, Y],
  lambda n, k, h, w, x, y:
    tvm.te.sum(A[n, c, h * st + r, w * st + s, x, z] * B[k, c, r, s, z, y], axis=[c,r,s,z]), name="C")

dC = tvm.te.placeholder([N, K, P, Q, X, Y], dtype=dtype, name="dC")

print(C.op.body)

print(dir(C.op.body[0].source[0]))

#print(tvm.tg.expr_equal(C.op.body[0].source[0].b.args[0], C.op.body[0].source[0].b.args[1]))

dB = tvm.tg.grad_op(B, C, dC)

s = tvm.te.create_schedule(dB.op)

print(tvm.lower(s, [A, B, dC, dB], simple_mode=True))

func = tvm.build(s, [A, B, dC, dB], target="llvm")

