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


class CountOperationNum : public ExprVisitor {
 public:
  using ExprVisitor::VisitExpr;
  using ExprVisitor::VisitExpr_;

  int num_add;
  int num_mul;
  int num_div;
  int num_branch;
  int num_logic;
  int num_special;

  CountOperationNum() {
    num_add = 0;
    num_mul = 0;
    num_div = 0;
    num_branch = 0;
    num_logic = 0;
    num_special = 0;
  }

  void VisitExpr_(const CallNode* op) override;
  void VisitExpr_(const AddNode* op) override;
  void VisitExpr_(const SubNode* op) override;
  void VisitExpr_(const MulNode* op) override;
  void VisitExpr_(const DivNode* op) override;
  void VisitExpr_(const ModNode* op) override;
  void VisitExpr_(const FloorDivNode* op) override;
  void VisitExpr_(const FloorModNode* op) override;
  void VisitExpr_(const MinNode* op) override;
  void VisitExpr_(const MaxNode* op) override;
  void VisitExpr_(const AndNode* op) override;
  void VisitExpr_(const OrNode* op) override;
  void VisitExpr_(const ReduceNode* op) override;
  void VisitExpr_(const CastNode* op) override;
  void VisitExpr_(const NotNode* op) override;
  void VisitExpr_(const SelectNode* op) override;
};


class CountInputOccur : public ExprVisitor { 
 private:
  Array<Tensor> inputs_;
 public:
  using ExprVisitor::VisitExpr;
  using ExprVisitor::VisitExpr_;

  std::vector<int> count_occur;

  CountInputOccur(Array<Tensor> inputs) : inputs_(inputs) {
    for (auto t : inputs) {
      count_occur.push_back(0);
    }
  }

  void VisitExpr_(const CallNode* op) override;
};

}  // namespace tvm

}  // namespace te
#endif  // TVM_TE_LONGTAIL_UTILS_H_
