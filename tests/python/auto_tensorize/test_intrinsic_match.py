import tvm
import tvm.te as te
import tvm.auto_tensorize as at


N = 4
C = 1024
P = 14
Q = 14
K = 512
R = 3
S = 3
H = P + R//2*2
W = Q + S//2*2

input_dtype = "float16"
output_dtype = "float32"


def gemm_intrinsic_compute():
  A = te.placeholder([32, 32], name="AA")
  B = te.placeholder([32, 32], name="BB")
  k = te.reduce_axis([0, 32], name="kk")
  Out = te.compute([32, 32], lambda i, j: te.sum(A[i, k] * B[k, j], axis=[k]), name="OO")
  return Out


def test1():
  A = te.placeholder([N, H, W, C], dtype=input_dtype, name="A")
  Weight = te.placeholder([R, S, C, K], dtype=input_dtype, name="W")
  rc = te.reduce_axis([0, C], name="rc")
  rr = te.reduce_axis([0, R], name="rr")
  rs = te.reduce_axis([0, S], name="rs")
  Out = te.compute([N, P, Q, K],
    lambda b, p, q, k: te.sum(
      (A[b, p+rr, q+rs, rc] * Weight[rr, rs, rc, k]).astype(output_dtype), 
    axis=[rc, rr, rs]), name="Out")

  intrin_t = gemm_intrinsic_compute()

  print("Target compute:")
  print(Out.op.body[0])

  print("Intrin compute:")
  print(intrin_t.op.body[0])

  recipe = at.WMMAFp16Fp32()
  main_capsule = recipe.get_capsule_compute_expression(
    'nnn', '16x16x16', recipe.main_capsule_name)

  print("Intrinsic match:")
  print(at.intrinsic_match(Out, intrin_t, main_capsule.op))


def test2():
  A = te.placeholder([H, C], dtype=input_dtype)
  Weight = te.placeholder([C, W], dtype=input_dtype)
  rc = te.reduce_axis([0, C], name="rc")
  Out = te.compute([H, W],
    lambda i, j: te.sum(
      (A[i, rc] * Weight[rc, j]).astype(output_dtype), 
  axis=[rc]), name="Out")

  intrin_t = gemm_intrinsic_compute()

  print("Target compute:")
  print(Out.op.body[0])

  print("Intrin compute:")
  print(intrin_t.op.body[0])

  recipe = at.WMMAFp16Fp32()
  main_capsule = recipe.get_capsule_compute_expression(
    'nnn', '16x16x16', recipe.main_capsule_name)

  print("Intrinsic match:")
  print(at.intrinsic_match(Out, intrin_t, main_capsule.op))


if __name__ == "__main__":
  test1()
  test2()
