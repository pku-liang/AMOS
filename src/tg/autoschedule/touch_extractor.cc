/*!
 * \file touch_extractor.cc
 * \brief Extract feature of touch pattern of axes in lowered IR
 */

#include "touch_extractor.h"
#include "feature.h"

#include <set>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace tvm {
namespace tg {

class IndexMutator : public ExprMutator {
public:
  PrimExpr VisitExpr_(const FloorDivNode* op) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    return DivNode::make(a, b);
  }
};

// get touch pattern from index expression
class IndexParser: public ExprVisitor {
 public:
  void Parse(PrimExpr expr) {
    pattern_map.clear();

    expr = IndexMutator()(expr);
    expr = tvm::tir::CanonicalSimplify(expr);
    
    this->VisitExpr(expr);
  }

  void VisitExpr_(const VarNode* op) final {
    // TODO(lmzheng): handle more index types (multiple occurrence)
    if (pattern_map.count(op) == 0) {
      pattern_map[op] = next_stride_;
      next_stride_ = 1.;
    }
  }

  void VisitExpr_(const MulNode* op) final {
    if (op->a.as<VarNode>()) {
      if (const auto stride = op->b.as<IntImmNode>()) {
        next_stride_ = stride->value;
      } else if (const auto stride = op->b.as<FloatImmNode>()) {
        next_stride_ = stride->value;
      }
    }
    ExprVisitor::VisitExpr_(op);
  }

  std::unordered_map<const VarNode*, int64_t> pattern_map;

 private:
  float next_stride_ = 1.;
};


bool TouchExtractor::EnterItervar_(Var var, int64_t min, int64_t length) {
  itervar_stack_.push_back(var);
  extent[var] = length;
  loop_min[var] = min;
  return true;
}


void TouchExtractor::ExitItervar_() {
  Var var = itervar_stack_.back();
  itervar_stack_.pop_back();
}


void TouchExtractor::EnterInnermostStmt_(const StoreNode &innermost_stmt) {
  this->current_stmt = &innermost_stmt;
  innermost_stmt_map[&innermost_stmt] = InnermostStatementFeature(this->innermost_stmt_counter_++);
}


void TouchExtractor::ExitInnermostStmt_() { this->current_stmt = nullptr; }


void TouchExtractor::EnterMem_(Var buffer_var, PrimExpr index, AccessType access_type) {
  TouchedBuffer buf = buffer_var.get()->name_hint;
  auto& feature = innermost_stmt_map[current_stmt].buffer_access_feature;

  std::vector<int64_t> buffer_shape;
  std::string buffer_scope;
  int64_t buffer_elem_bytes = -1;

  for (auto item : this->buffer_info_) {
    auto& s1 = buffer_var->name_hint;
    auto& s2 = item.first->name_hint;
    auto res = std::mismatch(s2.begin(), s2.end(), s1.begin());
    if (res.first == s2.end()) {
      buffer_shape = item.second.shape;
      buffer_elem_bytes = item.second.dtype.bytes();
      if (s1 == s2)
        buffer_scope = item.second.scope;
      else buffer_scope = s1.substr(s1.rfind(".") + 1, s1.size());
      break;
    }
  }

  int64_t buffer_nelems =
      std::accumulate(buffer_shape.begin(), buffer_shape.end(), 1, std::multiplies<int64_t>());

  IndexParser parser;
  parser.Parse(index);

  // access type
  feature[buf].access_type = AccessType(feature[buf].access_type | access_type);

  // reuse type
  auto& reuse_type = feature[buf].reuse_type;

  bool loop_reuse_tag = false;
  int64_t bytes = buffer_elem_bytes;
  int64_t unique_bytes = buffer_nelems * buffer_elem_bytes;
  int64_t reuse_counter = 1;
  int64_t &stride = feature[buf].stride;

  for (auto var : itervar_stack_) {
    auto x = parser.pattern_map.find(var.get());

    auto length = extent[var];
    bytes *= length;
    if (x != parser.pattern_map.end()) {
      // unique_bytes *= length;
      if (stride == 0) {
        stride = x->second;
      } else {
        stride = std::min(stride, x->second);
      }
    } else {
      loop_reuse_tag = true;
      reuse_counter *= length;
    }
    if (loop_reuse_tag) reuse_type = ReuseType(reuse_type | ReuseType::kLoopMultipleRead);
  }
  feature[buf].bytes += bytes;
  // feature[buf].unique_bytes += unique_bytes;
  feature[buf].unique_bytes = unique_bytes;
  feature[buf].reuse_counter += reuse_counter;

  auto& appearances = buffervar_stmt_map[buffer_var];
  appearances.insert({this->current_stmt});
  bool serial_reuse_tag = appearances.size() > 1;
  if (serial_reuse_tag) {
    for (auto stmt: buffervar_stmt_map[buffer_var]) {
      auto& f = innermost_stmt_map[stmt].buffer_access_feature;
      auto& rt = f[buf].reuse_type;
      rt = ReuseType(rt | ReuseType::kSerialMultipleRead);
    }
  }

  int64_t topdown = 1;
  for (auto var : itervar_stack_) {
    auto length = this->extent[var];
    topdown *= length;
  }

  if (buffer_scope == "global") {
    feature[buf].lines += topdown;
    const int CACHELINE_SIZE = 128;  // 128 bytes per L1 cache line

    feature[buf].unique_lines = buffer_nelems * buffer_elem_bytes / CACHELINE_SIZE;
  }

  if (loop_reuse_tag) {
    int64_t bottomup = 1;
    for (auto var = itervar_stack_.rbegin(); var != itervar_stack_.rend(); ++var) {
      auto x = parser.pattern_map.find(var->get());
      auto length = extent[*var];
      if (x != parser.pattern_map.end()) {
        bottomup *= length;
      } else {
        break;
      }
    }
    feature[buf].reuse_distance = bottomup;
  }
}


void TouchExtractor::ExitMem_() { }


void TouchExtractor::VisitStmt_(const StoreNode* op) {
  EnterInnermostStmt_(*op);
  EnterMem_(op->buffer_var, op->index, AccessType::kWrite);
  StmtExprVisitor::VisitStmt_(op);
  ExitMem_();
  ExitInnermostStmt_();
}


void GetInnerStatementFeature(
  Stmt stmt, bool take_log, 
  Array<Array<Array<PrimExpr> > > *ret_feature, 
  Map<te::Tensor, tir::Buffer> &out_binds) {
  // extract
  TouchExtractor touch_analyzer;
  touch_analyzer.Analyze(stmt, out_binds);

  // sort according to order
  std::vector<const StoreNode*> innermost_stmts;
  for (auto kv : touch_analyzer.innermost_stmt_map) {
    innermost_stmts.push_back(kv.first);
  }
  std::sort(innermost_stmts.begin(), innermost_stmts.end(),
            [&](const StoreNode *lhs, const StoreNode *rhs) -> bool {
              return touch_analyzer.innermost_stmt_map[lhs].order <
                     touch_analyzer.innermost_stmt_map[rhs].order;
            });

  // whether take log for numerical feature
  std::function<double(int64_t)> trans;
  if (take_log) {
    trans = [](int64_t x) {
      if (x < 0)
        return -std::log(-x+1) / std::log(2);
      x = x + 1;
      return std::log(x) / std::log(2);
    };
  } else {
    trans = [](int64_t x) {
      return x;
    };
  }

  // serialize for front end
  for (auto stmt : innermost_stmts) {
    Array<Array<PrimExpr> > feature_row;
    InnermostStatementFeature &fea = touch_analyzer.innermost_stmt_map[stmt];

    std::stringstream buffer;
    buffer << stmt->buffer_var << "[" << stmt->index << "] = " << stmt->value;
    feature_row.push_back(Array<PrimExpr>{
        std::string("_stmt_"),
        buffer.str(),
    });

    // buffer access feature
    std::vector<TouchedBuffer> bufs;
    for (auto kv : fea.buffer_access_feature) {
      bufs.push_back(kv.first);
    }
    std::sort(bufs.begin(), bufs.end());

    for (auto k : bufs) {
      BufferAccessFeature &v = fea.buffer_access_feature[k];
      feature_row.push_back(
          Array<PrimExpr>{k,
                v.access_type,
                FloatImm(DataType::Float(32), trans(v.bytes)),
                FloatImm(DataType::Float(32), trans(v.unique_bytes)),
                FloatImm(DataType::Float(32), trans(v.lines)),
                FloatImm(DataType::Float(32), trans(v.unique_lines)),
                v.reuse_type,
                FloatImm(DataType::Float(32), trans(v.reuse_distance)),
                FloatImm(DataType::Float(32), trans(v.reuse_counter)),
                FloatImm(DataType::Float(32), trans(v.stride)),
                });
    }

    ret_feature->push_back(feature_row);
  }
}


void GetInnerStatementFeatureFlatten(
  Stmt stmt, bool take_log, 
  Array<FloatImm> *ret_feature, 
  Map<te::Tensor, tir::Buffer> &out_binds) {
  // extract touch feature
  TouchExtractor touch_analyzer;
  touch_analyzer.Analyze(stmt, out_binds);

  // sort according to order
  std::vector<const StoreNode *> innermost_stmts;
  for (auto kv : touch_analyzer.innermost_stmt_map) {
    innermost_stmts.push_back(kv.first);
  }
  std::sort(innermost_stmts.begin(), innermost_stmts.end(),
            [&](const StoreNode *lhs, const StoreNode *rhs) -> bool {
              return touch_analyzer.innermost_stmt_map[lhs].order <
                     touch_analyzer.innermost_stmt_map[rhs].order;
            });

  // whether take log for numerical feature
  std::function<double(int64_t)> trans;
  if (take_log) {
    trans = [](int64_t x) {
      if (x < 0)
        return -std::log(-x+1) / std::log(2);
      x = x + 1;
      return std::log(x) / std::log(2);
    };
  } else {
    trans = [](int64_t x) {
      return x;
    };
  }

  // serialize for front end
  for (auto stmt : innermost_stmts) {
    InnermostStatementFeature &fea = touch_analyzer.innermost_stmt_map[stmt];

    // buffer access feature
    std::vector<TouchedBuffer> bufs;
    for (auto kv : fea.buffer_access_feature) {
      bufs.push_back(kv.first);
    }
    std::sort(bufs.begin(), bufs.end());

    // feature vector length: 15
    for (auto i = 0; i < std::min(int(bufs.size()), 5); i++) {
      BufferAccessFeature &v = fea.buffer_access_feature[bufs[i]];
      for (auto j = 0; j < 4; j++)  // one-hot encoding
        ret_feature->push_back(FloatImm(DataType::Float(32), j == v.access_type));
      ret_feature->push_back(FloatImm(DataType::Float(32), trans(v.bytes)));
      ret_feature->push_back(FloatImm(DataType::Float(32), trans(v.unique_bytes)));
      ret_feature->push_back(FloatImm(DataType::Float(32), trans(v.lines)));
      ret_feature->push_back(FloatImm(DataType::Float(32), trans(v.unique_lines)));
      for (auto j = 0; j < 4; j++)  // one-hot encoding
        ret_feature->push_back(FloatImm(DataType::Float(32), j == v.reuse_type));
      ret_feature->push_back(FloatImm(DataType::Float(32), trans(v.reuse_distance)));
      ret_feature->push_back(FloatImm(DataType::Float(32), trans(v.reuse_counter)));
      ret_feature->push_back(FloatImm(DataType::Float(32), trans(v.stride)));
    }

    for (auto i = 0; i < 5 - int(bufs.size()); i++)
      for (auto j = 0; j < 15; j++)
        ret_feature->push_back(FloatImm(DataType::Float(32), 0));
  }
}

// register API for front end
TVM_REGISTER_GLOBAL("tg.GetInnerStatementFeature")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  Stmt stmt = args[0];
  bool take_log = args[1];
  Map<te::Tensor, tir::Buffer> out_binds = args[2];
  Array<Array<Array<PrimExpr> > > ret_feature;

  GetInnerStatementFeature(stmt, take_log, &ret_feature, out_binds);

  *ret = ret_feature;
});


TVM_REGISTER_GLOBAL("tg.GetInnerStatementFeatureFlatten")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  Stmt stmt = args[0];
  bool take_log = args[1];
  Map<te::Tensor, tir::Buffer> out_binds = args[2];
  Array<FloatImm> ret_feature;

  GetInnerStatementFeatureFlatten(stmt, take_log, &ret_feature, out_binds);

  // TODO: cast ret_feature into a byte array
  /* TVMByteArray arr;
  arr.size = sizeof(float) * ret_feature.size();
  arr.data = reinterpret_cast<char *>(ret_feature.data()); */
  *ret = ret_feature;  // arr
});

}  // namespace autotvm
}  // namespace tvm
