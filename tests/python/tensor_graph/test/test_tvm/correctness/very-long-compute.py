import tvm


def test1(repeat=10):
  A = tvm.te.placeholder([4, 4])
  Output = tvm.te.compute([4, 4], lambda i, j: A[i, j] + 1)
  for i in range(repeat):
    Output = tvm.te.compute([4, 4], lambda i, j: Output[i, j] + 1)

  sch = tvm.te.create_schedule(Output.op)
  func = tvm.build(sch, [A, Output], target="llvm")


if __name__ == "__main__":
  test1(680)