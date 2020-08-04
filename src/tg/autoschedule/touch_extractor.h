/*!
 * \file touch_extractor.h
 * \brief Extract feature of touch pattern of buffers in lowered IR
 */

#ifndef TVM_AUTOTVM_TOUCH_EXTRACTOR_H_
#define TVM_AUTOTVM_TOUCH_EXTRACTOR_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/runtime/registry.h>
#include <tvm/node/structural_equal.h>

#include <stack>
#include <vector>
#include <map>
#include <string>
#include <deque>
#include <unordered_map>
#include "feature_visitor.h"  // src/tg/autoschedule/feature_visitor.h

namespace tvm {
namespace tg {

using TouchedBuffer = std::string;

template <typename P, typename Q>
using TvmMap = std::unordered_map<P, Q, tvm::ObjectHash, tvm::ObjectEqual>;

template <typename T>
using TvmSet = std::unordered_set<T, tvm::ObjectHash, tvm::ObjectEqual>;

enum ReuseType {
  kNoReuse = 0b00,
  kLoopMultipleRead = 0b01,
  kSerialMultipleRead = 0b10,
  kReuseTypeNum = 3,
};


struct BufferInfo {
  std::string scope;
  std::vector<int64_t> shape;
  DataType dtype;
};


struct BufferAccessFeature {
  // The type of access (read, write, read + write)
  AccessType access_type{AccessType::kNone};
  int64_t bytes{0};           // The total number of bytes accessed by this statement.
  int64_t unique_bytes{0};    // The total number of unique bytes accessed by this statement.
  int64_t lines{0};           // The total number of cache lines accessed by this statement.
  int64_t unique_lines{0};    // The total number of unique cache lines accessed by this statement.
  // The type of data reuse (LoopMulti- pleRead, SerialMultipleRead, NoReuse).
  ReuseType reuse_type{ReuseType::kNoReuse};       
  int64_t reuse_distance{0};  // The distance between data reuse in terms of number of for loop iterations and total accessed bytes.
  int64_t reuse_counter{0};   // The number of the happening of data reuse.
  int64_t stride{0};          // The stride of access.
};

// all the feature of an innermost statement
struct InnermostStatementFeature {
  InnermostStatementFeature() {}
  InnermostStatementFeature(int64_t counter): order(counter) {}

  int64_t order;

  std::unordered_map<TouchedBuffer, BufferAccessFeature> buffer_access_feature;
};

// extract iter vars and their touch pattern from ir
class TouchExtractor : public FeatureVisitor {
 public:
  void Analyze(const Stmt& stmt, const Map<te::Tensor, tir::Buffer> &out_binds) {
    for (auto &bind: out_binds) {
      auto &&buf = bind.second.as<BufferNode>();
      assert(buf->shape[0].as<IntImmNode>() && "data type of buffer shape is not IntImm");
      std::vector<int64_t> shape;
      for (auto x: buf->shape) shape.push_back(x.as<IntImmNode>()->value);
      this->buffer_info_.insert({buf->data, BufferInfo{buf->scope, shape, buf->dtype}});
    }
    operator()(stmt);
  }

  // error: no match for call to ‘(const std::hash<tvm::tir::ProvideNode>) (const tvm::tir::ProvideNode&)

  std::unordered_map<const StoreNode*, InnermostStatementFeature> innermost_stmt_map;
  // TvmMap<const ProvideNode*, InnermostStatementFeature> innermost_stmt_map;
  TvmMap<Var, std::unordered_set<const StoreNode*> > buffervar_stmt_map;


 private:
  bool EnterItervar_(Var var, int64_t min, int64_t length);
  void ExitItervar_();
  void EnterInnermostStmt_(const StoreNode &innermost_stmt);
  void ExitInnermostStmt_();
  void EnterMem_(Var buffer_var, PrimExpr index, AccessType access_type);
  void ExitMem_();

  void VisitStmt_(const StoreNode* op) final;

  const StoreNode *current_stmt {nullptr};
  AccessType current_buffer_access_type {AccessType::kNone};
  std::deque<Var> itervar_stack_;
  TvmMap<Var, int64_t> extent;
  TvmMap<Var, int64_t> loop_min;
  size_t innermost_stmt_counter_{0};

  TvmMap<tir::Var, BufferInfo> buffer_info_;

  using FeatureVisitor::VisitExpr_;
};

void GetInnerStatementFeatureFlatten(Stmt stmt, bool take_log, Array<FloatImm> *ret_feature, Map<te::Tensor, tir::Buffer> &out_binds);

void GetInnerStatementFeature(Stmt stmt, bool take_log, Array<Array<Array<PrimExpr> > > *ret_feature, Map<te::Tensor, tir::Buffer> &out_binds);
}  // namespace tg
}  // namespace tvm

#endif  // TVM_AUTOTVM_TOUCH_EXTRACTOR_H_
