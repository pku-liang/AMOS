#include <tvm/auto_tensorize/matcher.h>

using namespace tvm;
using namespace tvm::te;
using namespace tvm::tir;
using namespace tvm::auto_tensorize;


Tensor get_gemm(int m, int n, int k) {
  auto input = placeholder({m, k});
  auto weight = placeholder({k, n});
  auto axis_k = reduce_axis({0, k});
  auto output = compute({m, n}, [=](auto i, auto j) { 
    return sum(input(i, axis_k) * weight(axis_k, j), {axis_k}); 
  });
  return output;
}

void test_recipe_dag_matcher_gemm() {
  auto target = get_gemm(128, 128, 128);
  auto intrin = get_gemm(16, 16, 16);
  auto main_capsule = intrin->op;
  auto matcher = RecipeDAGMatcher();
  auto results = matcher.match(target, intrin, main_capsule);
  std::cout << results << std::endl;
}


Tensor get_conv(int h, int w, int c, int K, int f) {
  auto input = placeholder({h, w, c});
  auto filter = placeholder({K, f, f, c});
  auto di = reduce_axis({0, f});
  auto dj = reduce_axis({0, f});
  auto dk = reduce_axis({0, c});

  auto conv = compute({h - 2, w - 2, K}, [=](auto i, auto j, auto k){ 
    return sum(input(i + di, j + dj, dk) * filter(k, di, dj, dk), {di, dj, dk}); 
  });
  return conv;
}

void test_recipe_dag_matcher_conv() { 
  auto target = get_conv(256, 256, 32, 32, 3);
  auto intrin = get_gemm(16, 16, 16);
  auto main_capsule = intrin->op;
  auto matcher = RecipeDAGMatcher();
  auto results = matcher.match(target, intrin, main_capsule);
  std::cout << results << std::endl;
}

Tensor get_dot(int n) {
  // TODO
}

void test_recipe_dag_matcher_dot() {
  // TODO
}

int main() {
  test_recipe_dag_matcher_gemm();
  test_recipe_dag_matcher_conv();
  test_recipe_dag_matcher_dot();
  return 0;
}
