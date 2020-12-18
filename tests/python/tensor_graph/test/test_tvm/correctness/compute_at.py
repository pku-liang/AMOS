import tvm 


def test1():
  print("test 1 #########################")
  A = tvm.te.compute([4, 4], lambda i, j: 2 * i + j, name="A")
  B = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1, name="B")
  C = tvm.te.compute([4, 4], lambda i, j: A[i, j] + B[i, j], name="C")

  s = tvm.te.create_schedule(C.op)

  s[A].compute_at(s[C], s[C].op.axis[0])
  s[B].compute_at(s[C], s[C].op.axis[0])

  print(tvm.lower(s, [A, C], simple_mode=True))

  func = tvm.build(s, [A, C], target="llvm")

  print("Success")


def test2():
  print("test 2 #########################")
  A = tvm.te.compute([4, 4], lambda i, j: 2 * i + j, name="A")
  B = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1, name="B")
  C = tvm.te.compute([4, 4], lambda i, j: B[i, j] * 2, name="C")

  s = tvm.te.create_schedule(C.op)
  print(tvm.lower(s, [A, C], simple_mode=True))

  s[A].compute_at(s[B], s[B].op.axis[1])
  s[B].compute_at(s[C], s[C].op.axis[1])

  print(tvm.lower(s, [A, C], simple_mode=True))

  func = tvm.build(s, [A, C], target="llvm")

  print("Success")


def test3():
  print("test 3 #########################")
  A = tvm.te.placeholder([4, 4], dtype="float32", name="A")
  B = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1, name="B")
  C = tvm.te.compute([4, 4], lambda i, j: B[i, j] * 2, name="C")
  D = tvm.te.placeholder([4, 4], dtype="float32", name="D")
  k = tvm.te.reduce_axis([0, 4], name="k")
  E = tvm.te.compute([4, 4], lambda i, j: tvm.te.sum(C[i, k] * D[k, j], axis=k), name="E")
  F = tvm.te.compute([4, 4], lambda i, j: E[i, j] - 2, name="F")
  G = tvm.te.compute([4, 4], lambda i, j: F[i, j] * 3, name="G")

  s = tvm.te.create_schedule(G.op)

  s[E].set_scope("local")

  xo, xi = s[G].split(s[G].op.axis[0], factor=2)
  yo, yi = s[G].split(s[G].op.axis[1], factor=2)
  s[G].reorder(xo, yo, xi, yi)

  AA = s.cache_read(A, "shared", [B.op])
  DD = s.cache_read(D, "shared", [E])
  s[E].compute_at(s[G], yo)
  k = s[E].op.reduce_axis[0]

  ex, ey = s[E].op.axis
  ko, ki = s[E].split(k, factor=2)
  s[E].reorder(ko, ki, ey)

  s[AA].compute_at(s[E], ex)
  s[DD].compute_at(s[E], ey)
  s[B].compute_at(s[E], ey)
  s[C].compute_at(s[E], ey)

  s[F].compute_at(s[G], yi)

  print(tvm.lower(s, [A, D, F], simple_mode=True))

  func = tvm.build(s, [A, D, F], target="llvm")

  print("Success")


def test4():
  print("test 4 #########################")
  A = tvm.te.placeholder([4, 4], dtype="float32", name="A")
  B = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1, name="B")
  C = tvm.te.compute([4, 4], lambda i, j: B[i, j] * 2, name="C")
  D = tvm.te.placeholder([4, 4], dtype="float32", name="D")
  k = tvm.te.reduce_axis([0, 4], name="k")
  E = tvm.te.compute([4, 4], lambda i, j: tvm.te.sum(C[i, k] * D[k, j], axis=k), name="E")
  F = tvm.te.compute([4, 4], lambda i, j: E[i, j] - 2, name="F")
  G = tvm.te.compute([4, 4], lambda i, j: F[i, j] * 3, name="G")

  s = tvm.te.create_schedule(G.op)

  s[E].set_scope("local")

  xo, xi = s[G].split(s[G].op.axis[0], factor=2)
  yo, yi = s[G].split(s[G].op.axis[1], factor=2)
  s[G].reorder(xo, yo, xi, yi)

  AA = s.cache_read(A, "shared", [B.op])
  DD = s.cache_read(D, "shared", [E])
  s[E].compute_at(s[G], yo)
  k = s[E].op.reduce_axis[0]

  ko, ki = s[E].split(k, factor=2)
  EF = s.rfactor(E, ko)
  s[E].parallel(s[E].op.reduce_axis[0])
  s[EF].compute_at(s[E], s[E].op.reduce_axis[0])
  ex, ey = s[E].op.axis

  s[AA].compute_at(s[E], ex)
  s[DD].compute_at(s[E], ey)
  s[B].compute_at(s[EF], s[EF].op.reduce_axis[-1])
  s[C].compute_at(s[EF], s[EF].op.reduce_axis[-1])

  s[F].compute_at(s[G], yi)

  print(tvm.lower(s, [A, D, F], simple_mode=True))

  func = tvm.build(s, [A, D, F], target="llvm")

  print("Success")


def test5():
  print("test 5 #########################")
  A = tvm.te.placeholder([4, 4], dtype="float32", name="A")
  B = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1, name="B")
  C = tvm.te.compute([4, 4], lambda i, j: B[i, j] * 2, name="C")
  D = tvm.te.placeholder([4, 4], dtype="float32", name="D")
  k = tvm.te.reduce_axis([0, 4], name="k")
  E = tvm.te.compute([4, 4], lambda i, j: tvm.te.sum(C[i, k] * D[k, j], axis=k), name="E")
  F = tvm.te.compute([4, 4], lambda i, j: E[i, j] - 2, name="F")
  G = tvm.te.compute([4, 4], lambda i, j: F[i, j] * 3, name="G")

  s = tvm.te.create_schedule([G.op])

  s[E].set_scope("local")

  xo, xi = s[G].split(s[G].op.axis[0], factor=2)
  yo, yi = s[G].split(s[G].op.axis[1], factor=2)
  s[G].reorder(xo, yo, xi, yi)

  AA = s.cache_read(A, "shared", [B.op])
  DD = s.cache_read(D, "shared", [E])
  s[E].compute_at(s[G], yo)
  k = s[E].op.reduce_axis[0]

  ko, ki = s[E].split(k, factor=2)
  EF = s.rfactor(E, ko)
  s[E].parallel(s[E].op.reduce_axis[0])
  s[EF].compute_at(s[E], s[E].op.reduce_axis[0])
  ex, ey = s[E].op.axis

  s[AA].compute_at(s[E], ex)
  s[DD].compute_at(s[E], ey)
  s[B].compute_at(s[EF], s[EF].op.reduce_axis[-1])
  s[C].compute_at(s[EF], s[EF].op.reduce_axis[-1])

  s[F].compute_at(s[G], yi)

  print(tvm.lower(s, [A, D, F], simple_mode=True))

  func = tvm.build(s, [A, D, F], target="llvm")

  print("Success")


def test6():
  print("test 6 #########################")
  A = tvm.te.placeholder([4, 4], dtype="float32", name="A")
  B = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1, name="B")
  C = tvm.te.compute([4, 4], lambda i, j: B[i, j] * 2, name="C")
  D = tvm.te.placeholder([4, 4], dtype="float32", name="D")
  k = tvm.te.reduce_axis([0, 4], name="k")
  E = tvm.te.compute([4, 4], lambda i, j: tvm.te.sum(C[i, k] * D[k, j], axis=k), name="E")
  F = tvm.te.compute([4, 4], lambda i, j: E[i, j] - 2, name="F")
  G = tvm.te.compute([4, 4], lambda i, j: F[i, j] * 3, name="G")

  s = tvm.te.create_schedule([G.op])

  s[E].set_scope("local")

  xo, xi = s[G].split(s[G].op.axis[0], factor=2)
  yo, yi = s[G].split(s[G].op.axis[1], factor=2)
  s[G].reorder(xo, yo, xi, yi)

  AA = s.cache_read(A, "shared", [B.op])
  DD = s.cache_read(D, "shared", [E])
  s[E].compute_at(s[G], yo)
  k = s[E].op.reduce_axis[0]

  ko, ki = s[E].split(k, factor=2)
  EF = s.rfactor(E, ko)
  s[E].parallel(s[E].op.reduce_axis[0])
  s[EF].compute_at(s[E], s[E].op.reduce_axis[0])
  ex, ey = s[E].op.axis

  s[AA].compute_at(s[G], xo)
  s[DD].compute_at(s[G], xo)
  s[B].compute_at(s[G], xo)
  s[C].compute_at(s[G], xo)

  s[F].compute_at(s[G], yi)

  print(tvm.lower(s, [A, D, F], simple_mode=True))

  func = tvm.build(s, [A, D, F], target="llvm")

  print("Success")


if __name__ == "__main__":
  # test1()
  test2()
  # test3()
  # test4()
  # test5()
  # test6()