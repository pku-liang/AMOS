import tvm 

dtype = "float32"

A = tvm.te.placeholder([4, 1024], dtype=dtype, name="A")
B = tvm.te.placeholder([1024, 16], dtype=dtype, name="B")
k = tvm.te.reduce_axis([0, 1024], name="k")
C = tvm.te.compute([4, 16], lambda i, j: tvm.te.sum(A[i, k] * B[k, j], axis=[k]), name="C")

A1 = tvm.te.placeholder([4, 1024, 64], dtype=dtype, name="A1")
B1 = tvm.te.placeholder([1024, 16], dtype=dtype, name="B1")
C1 = tvm.te.placeholder([64, 16], dtype=dtype, name="C1")
k1 = tvm.te.reduce_axis([0, 1024], name="k1")
k2 = tvm.te.reduce_axis([0, 64], name="k2")
D1 = tvm.te.compute([4, 16], lambda i, j: tvm.te.sum(A1[i, k1, k2] * B1[k1, j] * C1[k2, j], axis=[k1, k2]), name="D1")

def test1():
  print("########################################")
  s = tvm.te.create_schedule(C.op)
  k = s[C].op.reduce_axis[0]
  ko, ki = s[C].split(k, factor=8)
  C_rf = s.rfactor(C, ki)

  s[C_rf].parallel(s[C_rf].op.axis[0])
  s[C].parallel(s[C].op.axis[1])
  s[C_rf].compute_at(s[C], s[C].op.axis[0])


  print(tvm.lower(s, [A, B, C], simple_mode=True))

  func = tvm.build(s, [A, B, C], target="llvm")


def test2():
  print("########################################")
  s = tvm.te.create_schedule(C.op)
  k = s[C].op.reduce_axis[0]
  ko, ki = s[C].split(k, factor=8)
  C_rf = s.rfactor(C, ki)
  s[C].bind(s[C].op.axis[0], tvm.te.thread_axis("blockIdx.x"))
  s[C_rf].bind(s[C_rf].op.axis[0], tvm.te.thread_axis("blockIdx.x"))

  print(tvm.lower(s, [A, B, C], simple_mode=True))

  func = tvm.build(s, [A, B, C], target="cuda")


def test3():
  print("########################################")
  s = tvm.te.create_schedule(C.op)
  k = s[C].op.reduce_axis[0]
  ko, ki = s[C].split(k, factor=8)
  C_rf = s.rfactor(C, ki)
  s[C].bind(s[C].op.axis[0], tvm.te.thread_axis("blockIdx.x"))
  s[C].bind(s[C].op.reduce_axis[0], tvm.te.thread_axis("threadIdx.x"))
  s[C_rf].compute_at(s[C], s[C].op.reduce_axis[0])

  print(tvm.lower(s, [A, B, C], simple_mode=True))

  func = tvm.build(s, [A, B, C], target="cuda")


def test4():
  print("########################################")
  s = tvm.te.create_schedule(D1.op)
  k1, k2 = s[D1].op.reduce_axis
  k1o, k1i = s[D1].split(k1, nparts=8)
  D1_rf = s.rfactor(D1, k1o)
  s[D1].bind(s[D1].op.axis[0], tvm.te.thread_axis("blockIdx.x"))
  s[D1].bind(s[D1].op.reduce_axis[0], tvm.te.thread_axis("threadIdx.x"))
  s[D1_rf].compute_at(s[D1], s[D1].op.reduce_axis[0])

  print(tvm.lower(s, [A1, B1, C1, D1], simple_mode=True))

  func = tvm.build(s, [A1, B1, C1, D1], target="cuda")


def test5():
  print("########################################")
  s = tvm.te.create_schedule(D1.op)
  k1o, k1i = s[D1].split(s[D1].op.reduce_axis[0], nparts=8)
  D1_rf1 = s.rfactor(D1, k1o)
  k2o, k2i = s[D1_rf1].split(s[D1_rf1].op.reduce_axis[1], nparts=16)
  D1_rf2 = s.rfactor(D1_rf1, k2o)
  s[D1].bind(s[D1].op.axis[0], tvm.te.thread_axis("blockIdx.x"))
  s[D1].bind(s[D1].op.reduce_axis[0], tvm.te.thread_axis("threadIdx.x"))
  s[D1_rf1].bind(s[D1_rf1].op.reduce_axis[0], tvm.te.thread_axis("threadIdx.y"))
  s[D1_rf1].compute_at(s[D1], s[D1].op.reduce_axis[0])
  s[D1_rf2].compute_at(s[D1_rf1], s[D1_rf1].op.reduce_axis[0])

  print(tvm.lower(s, [A1, B1, C1, D1], simple_mode=True))

  func = tvm.build(s, [A1, B1, C1, D1], target="cuda")


def test6():
  print("########################################")
  s = tvm.te.create_schedule(D1.op)
  # k = s[D1].fuse(s[D1].op.reduce_axis[0], s[D1].op.reduce_axis[1])
  k = s[D1].op.reduce_axis[0]
  ko, ki = s[D1].split(k, nparts=8)
  D1_rf = s.rfactor(D1, ko)
  s[D1].bind(s[D1].op.axis[0], tvm.te.thread_axis("blockIdx.x"))
  s[D1].bind(s[D1].op.reduce_axis[0], tvm.te.thread_axis("threadIdx.x"))
  s[D1_rf].compute_at(s[D1], s[D1].op.reduce_axis[0])

  print(tvm.lower(s, [A1, B1, C1, D1], simple_mode=True))

  func = tvm.build(s, [A1, B1, C1, D1], target="cuda")


def test7():
  print("######################################## test 7")
  A = tvm.te.placeholder([4, 12, 32, 32], name="A")
  B = tvm.te.compute([4, 12, 34, 34], 
    lambda n, c, h, w: tvm.te.if_then_else(tvm.tir.all(h >= 1, h < 33, w >= 1, w < 33), A[n, c, h-1, w-1], 0.0), name="B")
  C = tvm.te.placeholder([24, 12, 3, 3], name="C")
  rc = tvm.te.reduce_axis([0, 12], name="rc")
  rh = tvm.te.reduce_axis([0, 3], name="rh")
  rw = tvm.te.reduce_axis([0, 3], name="rw")
  D = tvm.te.compute([4, 24, 32, 32],
    lambda n, k, p, q: tvm.te.sum(B[n, rc, p+rh, q+rw] * C[k, rc, rh, rw], axis=[rc, rh, rw]), name="D")
  E = tvm.te.compute([4, 24, 32, 32], lambda n, c, h, w: tvm.te.if_then_else(D[n, c, h, w] > 0, D[n, c, h, w], 0.0), name="E")

  s = tvm.te.create_schedule(E.op)
  rco, rci = s[D].split(rc, factor=4)
  D_rf = s.rfactor(D, rci)
  s[D].parallel(s[D].op.reduce_axis[0])
  s[D_rf].compute_at(s[D], s[D].op.reduce_axis[0])
  s[B].compute_at(s[D], s[D].op.axis[1])
  s[D].compute_at(s[E], s[E].op.axis[1])

  print(tvm.lower(s, [A, C, E]))
  f = tvm.build(s, [A, C, E], "llvm")
  


if __name__ == "__main__":
  # test1()
  # test2()
  # test3()
  # test4()
  # test5()
  test6()
  # test7()