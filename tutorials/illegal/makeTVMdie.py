import tvm

A = tvm.te.placeholder([6, 6], name="A")
C = tvm.te.compute([6, 6], lambda i, j: A[i, j] + 1, name="C")
sch = tvm.te.create_schedule(C.op)
print(tvm.lower(sch, [A, C], simple_mode=True))

sch[C].compute_at(sch[C], C.op.axis[0])
print("compute_at pass")

print(tvm.lower(sch, [A, C], simple_mode=True))
print("lower pass")


# func = tvm.build(sch, [A, C], "llvm")
# print("build pass")
