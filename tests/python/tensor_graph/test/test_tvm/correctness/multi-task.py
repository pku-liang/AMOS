import tvm 


def gemm(batch, in_dim, out_dim):
  data = tvm.te.placeholder([batch, in_dim], dtype="float32", name="data")
  weight = tvm.te.placeholder([out_dim, in_dim], dtype="float32", name="data")

  batch, in_dim = data.shape
  out_dim, _ = weight.shape
  k = tvm.te.reduce_axis((0, in_dim), name='k')
  matmul = tvm.te.compute((batch, out_dim), \
                      lambda i, j: tvm.te.sum(data[i, k] * \
                                          weight[j, k], axis=k), \
                      name='gemm')

  return data, weight, matmul


M = 256
N = 256
K1 = 256
K2 = 64

def main():
  A1, B1, C1 = gemm(M, K1, N)
  A2, B2, C2 = gemm(M, K2, N)

  multi = tvm.te.compute([M+M, N], lambda i, j: tvm.tir.if_then_else(i < M, C1[i, j], C2[i, j]), name="cat")
  add = tvm.te.compute([M, N], lambda i, j: multi[i, j] + multi[i+M, j], name="add")


  s = tvm.te.create_schedule(add.op)

  s[C1].compute_at(s[multi], s[multi].op.axis[0])
  s[C2].compute_at(s[multi], s[multi].op.axis[0])
  print(tvm.lower(s, [A1, A2, B1, B2, add], simple_mode=True))


if __name__ == "__main__":
  main()