#include <tvm/auto_tensorize/matcher.h>

using namespace tvm;
using namespace tvm::te;
using namespace tvm::tir;
using namespace tvm::auto_tensorize;

void test_infer_bounds() {
  // auto n = var("n");
  Array<PrimExpr> shape {3,};

  auto A = placeholder(shape, DataType::Float(32), "A");
  auto B = placeholder(shape, DataType::Float(32), "B");
  auto C = compute(A->shape, [&A, &B](PrimExpr i) { return A[i] + B[i]; }, "C");

  auto matcher = RecipeDAGMatcher();
  // Map<IterVar, Range> bounds = matcher._infer_bounds(C->op);
  // std::cout << bounds << std::endl;
}

int main() {
  test_infer_bounds();
  return 0;
}
