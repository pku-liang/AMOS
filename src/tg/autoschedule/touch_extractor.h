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

// const char* INTRIN_KEYS[17]{
//     "exp", "exp2", "exp10", "erf", "tanh", "sigmoid", "log", "log2", "log10",
//     "tan", "cos", "cosh", "sin", "sinh", "atan", "sqrt", "rsqrt",
// };

enum LoopPositionType { 
  kNonePosition = 0,
  kInnerSpatial = 1,
  kMiddleSpatial = 2,
  kOuterSpatial = 3,
  kInnerReduce = 4,
  kMiddleReduce = 5,
  kOuterReduce = 6,
  kMixedPosition = 7,
};

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

struct IterVarInfo {
  Var var;
  bool is_attr_stmt;
  AnnotationType ann;
  const char *pragma_key;
  const PrimExpr *pragma_val;
  bool is_reduce;
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
  int64_t topdown{1};
};

// all the feature of an innermost statement
struct InnermostStatementFeature {
  InnermostStatementFeature() {}
  InnermostStatementFeature(int64_t counter): order(counter) {}

  int64_t order;

  int64_t int_add_ct{0};
  int64_t int_sub_ct{0};
  int64_t int_mul_ct{0};
  int64_t int_div_ct{0};
  int64_t int_mod_ct{0};
  int64_t int_cmp_ct{0};
  std::unordered_map<const char*, int64_t> int_intrin_ct;

  int64_t flt_add_ct{0};
  int64_t flt_sub_ct{0};
  int64_t flt_mul_ct{0};
  int64_t flt_div_ct{0};
  int64_t flt_mod_ct{0};
  int64_t flt_cmp_ct{0};
  std::unordered_map<const char*, int64_t> flt_intrin_ct;

  std::unordered_map<AnnotationType, int64_t> thread_bind_len;

  int64_t vectorize_len_imost{0};
  int64_t vectorize_len_prod{1};
  int64_t vectorize_loop_num{0};
  LoopPositionType vectorize_loop_pos{kNonePosition};

  int64_t unroll_len_imost{0};
  int64_t unroll_len_prod{1};
  int64_t unroll_loop_num{0};
  LoopPositionType unroll_loop_pos{kNonePosition};

  int64_t parallel_len_imost{0};
  int64_t parallel_len_prod{1};
  int64_t parallel_loop_num{0};
  LoopPositionType parallel_loop_pos{kNonePosition};

  int64_t num_outer_loops{0};
  int64_t prod_outer_loops{1};
  int64_t auto_unroll_max_step{0};

  std::vector<int64_t> output_buffer_size;
  TvmSet<Var> accessed_buffers;
  int64_t num_allocation;

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
      // TODO: retrieve buffer info from AllocateNode
    }
    operator()(stmt);
  }

  void VisitExpr_(const AddNode* op) final {
    if (current_stmt) {
      if (op->dtype.is_float())
        innermost_stmt_map[current_stmt].flt_add_ct++;
      else
        innermost_stmt_map[current_stmt].int_add_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const SubNode* op) final {
    if (current_stmt) {
      if (op->dtype.is_float())
        innermost_stmt_map[current_stmt].flt_sub_ct++;
      else
        innermost_stmt_map[current_stmt].int_sub_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const MulNode* op) final {
    if (current_stmt) {
      if (op->dtype.is_float())
        innermost_stmt_map[current_stmt].flt_mul_ct++;
      else
        innermost_stmt_map[current_stmt].int_mul_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const DivNode* op) final {
    assert(!op->dtype.is_float());
    if (current_stmt)
      innermost_stmt_map[current_stmt].int_div_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const ModNode* op) final {
    assert(!op->dtype.is_float());
    if (current_stmt)
      innermost_stmt_map[current_stmt].int_mod_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const FloorDivNode* op) final {
    assert(op->dtype.is_float());
    if (current_stmt)
      innermost_stmt_map[current_stmt].flt_div_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const FloorModNode* op) final {
    assert(op->dtype.is_float());
    if (current_stmt)
      innermost_stmt_map[current_stmt].flt_mod_ct++;
    FeatureVisitor::VisitExpr_(op);
  }

  template<typename T>
  void VisitExpr_(const CmpOpNode<T>* op) {
    if (current_stmt) {
      if (op->dtype.is_float())
        innermost_stmt_map[current_stmt].flt_cmp_ct++;
      else
        innermost_stmt_map[current_stmt].int_cmp_ct++;
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode* op) {
    if (op->call_type == CallNode::PureIntrinsic) {
      if (current_stmt) {
        if (op->dtype.is_float()) {
          innermost_stmt_map[current_stmt].flt_intrin_ct[op->name.c_str()]++;
        } else {
          innermost_stmt_map[current_stmt].int_intrin_ct[op->name.c_str()]++;
        }
      }
    }
    FeatureVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const IterVarNode* op) {
    std::cout << "Found IterVarNode: " << op->iter_type << " " << op->dom << std::endl;
  }

  std::unordered_map<const StoreNode*, InnermostStatementFeature> innermost_stmt_map;
  TvmMap<Var, std::unordered_set<const StoreNode*> > buffervar_stmt_map;

 private:
  bool EnterItervar_(Var var, int64_t min, int64_t length, bool is_attr_stmt, AnnotationType ann, 
                     const char *pragma_key, const PrimExpr *pragma_val);
  void ExitItervar_();
  void EnterInnermostStmt_(const StoreNode &innermost_stmt);
  void ExitInnermostStmt_();
  void EnterMem_(Var buffer_var, PrimExpr index, AccessType access_type);
  void ExitMem_();
  void EnterAllocateNode_(std::string scope);
  void ExitAllocateNode_();

  void VisitStmt_(const StoreNode* op) final;
  void VisitStmt_(const AllocateNode* op) final;

  const StoreNode *current_stmt {nullptr};
  std::deque<IterVarInfo> itervar_stack_;
  TvmMap<Var, int64_t> extent;
  TvmMap<Var, int64_t> loop_min;
  size_t innermost_stmt_counter_{0};
  std::string next_allocation_scope_;

  TvmMap<tir::Var, BufferInfo> buffer_info_;

  using FeatureVisitor::VisitExpr_;
};

void GetInnerStatementFeatureFlatten(Stmt stmt, bool take_log, Array<Array<FloatImm>> *ret_feature, Map<te::Tensor, tir::Buffer> &out_binds);

void GetInnerStatementFeature(Stmt stmt, bool take_log, Array<Array<Array<PrimExpr> > > *ret_feature, Map<te::Tensor, tir::Buffer> &out_binds);
}  // namespace tg
}  // namespace tvm

#endif  // TVM_AUTOTVM_TOUCH_EXTRACTOR_H_
