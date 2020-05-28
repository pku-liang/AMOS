
#ifndef TVM_TE_LONGTAIL_UTILS_H_
#define TVM_TE_LONGTAIL_UTILS_H_

#include <tvm/runtime/registry.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/te/operation.h>
#include <tvm/te/longtail.h>

#include <unordered_map>
#include <vector>


namespace tvm {

namespace te {


class FindBatchLikeDim : public ExprVisitor {
 private:
  Array<Var> spatial_indices_;
 public:
  std::unordered_map<int, std::vector<int>> records;
  using ExprVisitor::VisitExpr;
  FindBatchLikeDim(Array<Var> spatial_indices) : spatial_indices_(spatial_indices) {
    for (int i = 0; i < (int)spatial_indices_.size(); ++i) {
      std::vector<int> tmp;
      records[i] = tmp;
    }
  }
 protected:
  using ExprVisitor::VisitExpr_;
  void VisitExpr_(const CallNode* op) override;
};


class FindAxisPosition : public ExprVisitor {
 private:
  Array<Var> spatial_indices_;
  const Tensor &tensor_;
 public:
  std::unordered_map<int, std::vector<int>> records;
  using ExprVisitor::VisitExpr;
  FindAxisPosition(Array<Var> spatial_indices, const Tensor &tensor) :
      spatial_indices_(spatial_indices), tensor_(tensor) {
    for (int i = 0; i < (int)spatial_indices_.size(); ++i) {
      std::vector<int> tmp;
      records[i] = tmp;
    }
  }
 protected:
  using ExprVisitor::VisitExpr_;
  void VisitExpr_(const CallNode* op) override;
};


}  // namespace tvm

}  // namespace te
#endif  // TVM_TE_LONGTAIL_UTILS_H_
