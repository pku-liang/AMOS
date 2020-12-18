from tvm import testing
import tvm
import numpy as np 
import torch
from tvm from tvm import topi
# Set enable_gradient = True:   
# File "tvm-expr/src/te/myautodiff/arg_util.h", line 385
#  TVMError: Unexpected visit: B(i0, j0)
enable_gradient = True
batch = 10
word_per_batch = 5
total_word = 100
embed_dim = 20 #embedding dimension

dtype = "float32"

A = tvm.te.placeholder([total_word, embed_dim], dtype=dtype, name="A")
B = tvm.te.placeholder([batch, word_per_batch], dtype="int32", name="B")
C = tvm.te.compute([batch, word_per_batch, embed_dim], lambda i, j, k: A[B[i, j], k], name="C")
b = tvm.te.reduce_axis([0, batch], name="b")
w = tvm.te.reduce_axis([0, word_per_batch], name="w")
e = tvm.te.reduce_axis([0, embed_dim], name="e")
D = tvm.te.compute([1], lambda a: tvm.te.sum(C[a+b, w, e], axis=[b, w, e]), name="D")

if enable_gradient:
    print(D.op.body)
    print(dir(D.op.body[0].source[0]))

    print("all correct so far1") # pass
    dA, = tvm.tg.gradient(D, [A])
    print("all correct so far2") # fail
    s = tvm.te.create_schedule(dA.op)

    print(tvm.lower(s, [A, B, C, D, dA], simple_mode=True))
    func = tvm.build(s, [A, B, C, D, dA], target="llvm")

    A_np = np.random.uniform(-10, 10, [total_word, embed_dim]).astype("float32")
    B_np = np.random.uniform(0, 100, [batch, word_per_batch]).astype("int32")
    C_np = np.zeros([batch, word_per_batch, embed_dim]).astype("float32")
    D_np = np.zeros([1]).astype("float32")
    dA_np = np.zeros([total_word, embed_dim]).astype("float32")

    ctx = tvm.context("llvm", 0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)
    D_tvm = tvm.nd.array(D_np, ctx)
    dA_tvm = tvm.nd.array(dA_np, ctx)

    func(A_tvm, B_tvm, C_tvm, D_tvm, dA_tvm)

    print(dA_tvm)
else:
    print(D.op.body)
    print(dir(D.op.body[0].source[0]))
    # print("all correct so far")
    s = tvm.te.create_schedule(D.op)

    print(tvm.lower(s, [A, B, C, D], simple_mode=True))
    func = tvm.build(s, [A, B, C, D], target="llvm")

    A_np = np.random.uniform(-10, 10, [total_word, embed_dim]).astype("float32")
    B_np = np.random.uniform(0, 100, [batch, word_per_batch]).astype("int32")
    C_np = np.zeros([batch, word_per_batch, embed_dim]).astype("float32")
    D_np = np.zeros([1]).astype("float32")

    ctx = tvm.context("llvm", 0)
    A_tvm = tvm.nd.array(A_np, ctx)
    B_tvm = tvm.nd.array(B_np, ctx)
    C_tvm = tvm.nd.array(C_np, ctx)
    D_tvm = tvm.nd.array(D_np, ctx)

    func(A_tvm, B_tvm, C_tvm, D_tvm)

    print(D_tvm)

