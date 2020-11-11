#ifndef TVM_TG_GRAPH_UTILS_H_
#define TVM_TG_GRAPH_UTILS_H_

#include <tvm/runtime/registry.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
// #include <tvm/tir/ir_pass.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/te/operation.h>
#include <tvm/tg/graph.h>

#include <sstream>
#include <unordered_map>
#include <set>
#include <map>
#include <unordered_set>
#include <vector>
#include <deque>

#include "../utils.h"


namespace tvm {
using namespace te;
namespace tg {


class IntAndTensor {
 public:
  int key;
  Tensor t;

  bool operator== (const IntAndTensor& another) const {
    return (key == another.key) && (t == another.t);
  }

  bool operator!= (const IntAndTensor& another) const {
    return !((*this) == another);
  }

  bool operator< (const IntAndTensor& another) const {
    return (key < another.key);
  }

  bool operator> (const IntAndTensor& another) const {
    return (key > another.key);
  }
};


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
  void VisitExpr_(const ProducerLoadNode* op) override;
};


class FindFusibleDim : public ExprVisitor {
private:
    Array<Var> spatial_indices_;
    Array<Tensor> weights_;
public:
    std::vector<std::pair<bool, int> > spatial_indices_in_weight;
    std::vector<bool> spatial_indices_in_input;
    using ExprVisitor::VisitExpr;

    FindFusibleDim(Array<Var> spatial_indices, Array<Tensor> &weights)
        :spatial_indices_(spatial_indices), weights_(weights) {
        int n = (int) spatial_indices_.size();
        spatial_indices_in_weight.resize(n, std::make_pair(false, -1));
        spatial_indices_in_input.resize(n, false);
    }

protected:
    using ExprVisitor::VisitExpr_;

    void VisitExpr_(const ProducerLoadNode *op) override;
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
  void VisitExpr_(const ProducerLoadNode* op) override;
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

  void VisitExpr_(const ProducerLoadNode* op) override;
};


std::pair<Array<Operation>, Map<Operation, Array<Operation> > >
  serialize_compute_dag(Array<Operation> root_ops, bool output_first=false);


}  // namespace tg

}  // namespace tvm

#endif  // TVM_TG_GRAPH_UTILS_H_
