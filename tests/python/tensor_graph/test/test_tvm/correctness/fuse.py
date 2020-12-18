import tvm


A = tvm.te.compute([4, 4, 4], lambda i, j, k: 1.0)

s = tvm.te.create_schedule(A.op)
i, j, k = s[A].op.axis
s[A].reorder(i, k, j)
s[A].fuse(i, k)