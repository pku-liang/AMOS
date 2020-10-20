/*!
 * \file touch_extractor.cc
 * \brief Extract feature of touch pattern of axes in lowered IR
 */

#include <tvm/arith/analyzer.h>

#include "touch_extractor.h"
#include "feature.h"

#include <set>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>

namespace tvm {
namespace tg {

const char* INTRIN_KEYS[17]{
  "exp", "exp2", "exp10", "erf", "tanh", "sigmoid", "log", "log2", "log10",
  "tan", "cos", "cosh", "sin", "sinh", "atan", "sqrt", "rsqrt",
};

const AnnotationType THREAD_BIND_KEYS[]{
  kBlockX, kBlockY, kBlockZ, kThreadX, kThreadY, kThreadZ, kVirtualThread,
};

class IndexMutator : public ExprMutator {
public:
  PrimExpr VisitExpr_(const FloorDivNode* op) {
    PrimExpr a = this->VisitExpr(op->a);
    PrimExpr b = this->VisitExpr(op->b);
    return Div(a, b);
  }
};

// get touch pattern from index expression
class IndexParser: public ExprVisitor {
 public:
  void Parse(PrimExpr expr) {
    pattern_map.clear();
    arith::Analyzer ana;
    expr = IndexMutator()(expr);
    expr = ana.canonical_simplify(expr);
    
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


bool TouchExtractor::EnterItervar_(Var var, int64_t min, int64_t length, 
                                   bool is_attr_stmt, AnnotationType ann, 
                                   const char *pragma_key, const PrimExpr *pragma_val) {
  itervar_stack_.push_back({var, is_attr_stmt, ann, pragma_key, pragma_val, false});
  extent[var] = length;
  loop_min[var] = min;
  return true;
}


void TouchExtractor::ExitItervar_() {
  // Var var = itervar_stack_.back().var;
  itervar_stack_.pop_back();
}


void TouchExtractor::EnterInnermostStmt_(const StoreNode &innermost_stmt) {
  this->current_stmt = &innermost_stmt;
  innermost_stmt_map[current_stmt] = InnermostStatementFeature(this->innermost_stmt_counter_++);
  auto& fea = innermost_stmt_map[current_stmt];

  for (auto item : itervar_stack_) {
    Var var = item.var;

    if (item.ann == kPragma && !strcmp(item.pragma_key, "pragma_auto_unroll_max_step"))
      fea.auto_unroll_max_step = item.pragma_val->as<IntImmNode>()->value;

    fea.num_outer_loops ++;
    fea.prod_outer_loops *= extent[var];

    if (item.is_attr_stmt) {
      fea.thread_bind_len[item.ann] = extent[var];
    }
    else {
      if (item.ann == AnnotationType::kVectorized) {
        fea.vectorize_len_imost = extent[var];
        fea.vectorize_len_prod *= extent[var];
        fea.vectorize_loop_num ++;
      } else if (item.ann == AnnotationType::kUnrolled) {
        fea.unroll_len_imost = extent[var];
        fea.unroll_len_prod *= extent[var];
        fea.unroll_loop_num ++;
      } else if (item.ann == AnnotationType::kParallel) {
        fea.parallel_len_imost = extent[var];
        fea.parallel_len_prod *= extent[var];
        fea.parallel_loop_num ++;
      }
    }
  }
}


void TouchExtractor::ExitInnermostStmt_() {
  auto& fea = innermost_stmt_map[current_stmt];
  std::vector<LoopPositionType> loop_pos_types;

  for (auto item : itervar_stack_)
    loop_pos_types.push_back(item.is_reduce? kMiddleReduce: kMiddleSpatial);

  for (auto it = loop_pos_types.begin(); it != loop_pos_types.end(); it++)
    if (*it == kMiddleReduce) {
      *it = kOuterReduce;
      break;
    }

  for (auto it = loop_pos_types.rbegin(); it != loop_pos_types.rend(); it++)
    if (*it == kMiddleReduce) {
      *it = kInnerReduce;
      break;
    }

  for (auto it = loop_pos_types.begin(); it != loop_pos_types.end(); it++)
    if (*it == kMiddleSpatial) {
      *it = kOuterSpatial;
      break;
    }
  
  for (auto it = loop_pos_types.rbegin(); it != loop_pos_types.rend(); it++)
    if (*it == kMiddleSpatial) {
      *it = kInnerSpatial;
      break;
    }

  for (size_t i = 0; i < itervar_stack_.size(); i++) {
    auto& item = itervar_stack_[i];
    if (item.ann == kVectorized) {
      if (fea.vectorize_loop_pos) fea.vectorize_loop_pos = kMixedPosition;
      else fea.vectorize_loop_pos = loop_pos_types[i];
    } else if (item.ann == kUnrolled) {
      if (fea.unroll_loop_pos) fea.unroll_loop_pos = kMixedPosition;
      else fea.unroll_loop_pos = loop_pos_types[i];
    } else if (item.ann == kParallel) {
      if (fea.parallel_loop_pos) fea.parallel_loop_pos = kMixedPosition;
      else fea.parallel_loop_pos = loop_pos_types[i];
    }
  }
  this->current_stmt = nullptr; 
}


void TouchExtractor::EnterMem_(Var buffer_var, PrimExpr index, AccessType access_type) {
  TouchedBuffer buf = buffer_var.get()->name_hint;
  auto& stmt_feature = innermost_stmt_map[current_stmt];
  auto& feature = stmt_feature.buffer_access_feature;
  

  std::vector<int64_t> buffer_shape;
  std::string buffer_scope;
  int64_t buffer_elem_bytes = -1;

  stmt_feature.accessed_buffers.insert(buffer_var);
  stmt_feature.num_allocation = stmt_feature.accessed_buffers.size();

  buffer_shape = this->buffer_info_[buffer_var].shape;
  buffer_scope = this->buffer_info_[buffer_var].scope;
  buffer_elem_bytes = this->buffer_info_[buffer_var].dtype.bytes();

  IndexParser parser;
  parser.Parse(index);

  if (access_type & AccessType::kWrite) {
    stmt_feature.output_buffer_size = buffer_shape;
    for (auto item: itervar_stack_)
      item.is_reduce = !parser.pattern_map.count(item.var.get());
  }

  int64_t buffer_nelems =
      std::accumulate(buffer_shape.begin(), buffer_shape.end(), 1, std::multiplies<int64_t>());

  // access type
  feature[buf].access_type = AccessType(feature[buf].access_type | access_type);

  // reuse type
  auto& reuse_type = feature[buf].reuse_type;

  auto& appearances = buffervar_stmt_map[buffer_var];
  appearances.insert({this->current_stmt});
  bool serial_reuse_tag = appearances.size() > 1;

  bool loop_reuse_tag = false;
  int64_t bytes = buffer_elem_bytes;
  int64_t unique_bytes = buffer_nelems * buffer_elem_bytes;
  int64_t reuse_counter = 1;
  int64_t &stride = feature[buf].stride;
  int64_t topdown = 1;

  for (auto item : itervar_stack_) {
    auto x = parser.pattern_map.find(item.var.get());

    auto length = extent[item.var];
    bytes *= length;
    if (!item.is_attr_stmt) topdown *= length;

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
  feature[buf].topdown = topdown;

  if (serial_reuse_tag) {
    for (auto stmt: buffervar_stmt_map[buffer_var]) {
      auto& f = innermost_stmt_map[stmt].buffer_access_feature;
      auto& rt = f[buf].reuse_type;
      rt = ReuseType(rt | ReuseType::kSerialMultipleRead);
    }
  }

  int64_t topdown2 = 1;
  for (auto item : itervar_stack_) {
    auto length = this->extent[item.var];
    topdown2 *= length;
  }

  if (buffer_scope == "global") {
    feature[buf].lines += topdown2;
    const int CACHELINE_SIZE = 128;  // 128 bytes per L1 cache line

    feature[buf].unique_lines = buffer_nelems * buffer_elem_bytes / CACHELINE_SIZE;
  }

  if (loop_reuse_tag) {
    int64_t bottomup = 1;
    for (auto it = itervar_stack_.rbegin(); it != itervar_stack_.rend(); ++it) {
      auto x = parser.pattern_map.find(it->var.get());
      auto length = extent[it->var];
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


void TouchExtractor::EnterAllocateNode_(std::string scope) { this->next_allocation_scope_ = scope; }


void TouchExtractor::ExitAllocateNode_() {}


void TouchExtractor::VisitStmt_(const StoreNode* op) {
  EnterInnermostStmt_(*op);
  EnterMem_(op->buffer_var, op->index, AccessType::kWrite);
  StmtExprVisitor::VisitStmt_(op);
  ExitMem_();
  ExitInnermostStmt_();
}


void TouchExtractor::VisitStmt_(const AllocateNode* op) {
  assert(buffer_info_.count(op->buffer_var) == 0);
  std::vector<int64_t> buffer_shape;
  for (auto x : op->extents) buffer_shape.push_back(x.as<IntImmNode>()->value);
  buffer_info_.insert({op->buffer_var, BufferInfo{this->next_allocation_scope_, buffer_shape, op->dtype}});
  next_allocation_scope_ = "";
  StmtExprVisitor::VisitStmt_(op);
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
        StringImm("_stmt_"),
        StringImm(buffer.str()),
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
          Array<PrimExpr>{
                StringImm(k),
                v.access_type,
                FloatImm(DataType::Float(32), trans(v.bytes)),
                FloatImm(DataType::Float(32), trans(v.unique_bytes)),
                FloatImm(DataType::Float(32), trans(v.lines)),
                FloatImm(DataType::Float(32), trans(v.unique_lines)),
                v.reuse_type,
                FloatImm(DataType::Float(32), trans(v.reuse_distance)),
                FloatImm(DataType::Float(32), trans(v.reuse_counter)),
                FloatImm(DataType::Float(32), trans(v.stride)),
                FloatImm(DataType::Float(32), trans(v.topdown)),
                });
    }

    Array<PrimExpr> int_arithmetic_features{
      StringImm("int_arith_features"),
      FloatImm(DataType::Float(32), trans(fea.int_add_ct)),
      FloatImm(DataType::Float(32), trans(fea.int_sub_ct)),
      FloatImm(DataType::Float(32), trans(fea.int_mul_ct)),
      FloatImm(DataType::Float(32), trans(fea.int_div_ct)),
      FloatImm(DataType::Float(32), trans(fea.int_mod_ct)),
      FloatImm(DataType::Float(32), trans(fea.int_cmp_ct)),
    };

    for (auto k: INTRIN_KEYS) {
      if (fea.int_intrin_ct.count(k))
        int_arithmetic_features.push_back(FloatImm(DataType::Float(32), trans(fea.int_intrin_ct[k])));
      else
        int_arithmetic_features.push_back(FloatImm(DataType::Float(32), trans(0)));
    }
    feature_row.push_back(int_arithmetic_features);

    Array<PrimExpr> float_arithmetic_features{
      StringImm("flt_arith_features"),
      FloatImm(DataType::Float(32), trans(fea.flt_add_ct)),
      FloatImm(DataType::Float(32), trans(fea.flt_sub_ct)),
      FloatImm(DataType::Float(32), trans(fea.flt_mul_ct)),
      FloatImm(DataType::Float(32), trans(fea.flt_div_ct)),
      FloatImm(DataType::Float(32), trans(fea.flt_mod_ct)),
      FloatImm(DataType::Float(32), trans(fea.flt_cmp_ct)),
    };

    for (auto k: INTRIN_KEYS) {
      if (fea.flt_intrin_ct.count(k))
        float_arithmetic_features.push_back(FloatImm(DataType::Float(32), trans(fea.flt_intrin_ct[k])));
      else
        float_arithmetic_features.push_back(FloatImm(DataType::Float(32), trans(0)));
    }
    feature_row.push_back(float_arithmetic_features);

    feature_row.push_back(Array<PrimExpr>{
      StringImm("vectorization_features"),
      FloatImm(DataType::Float(32), trans(fea.vectorize_len_imost)),
      FloatImm(DataType::Float(32), trans(fea.vectorize_len_prod)),
      FloatImm(DataType::Float(32), trans(fea.vectorize_loop_num)),
      fea.vectorize_loop_pos,
    });

    feature_row.push_back(Array<PrimExpr>{
      StringImm("unrolling_features"),
      FloatImm(DataType::Float(32), trans(fea.unroll_len_imost)),
      FloatImm(DataType::Float(32), trans(fea.unroll_len_prod)),
      FloatImm(DataType::Float(32), trans(fea.unroll_loop_num)),
      fea.unroll_loop_pos,
    });

    feature_row.push_back(Array<PrimExpr>{
      StringImm("parallel_features"),
      FloatImm(DataType::Float(32), trans(fea.parallel_len_imost)),
      FloatImm(DataType::Float(32), trans(fea.parallel_len_prod)),
      FloatImm(DataType::Float(32), trans(fea.parallel_loop_num)),
      fea.parallel_loop_pos,
    });

    Array<PrimExpr> thread_bind_len{
      StringImm("thread_binding_features"),
    };
    for (auto k : THREAD_BIND_KEYS) {
      if (fea.thread_bind_len.count(k))
        thread_bind_len.push_back(FloatImm(DataType::Float(32), trans(fea.thread_bind_len[k])));
      else
        thread_bind_len.push_back(FloatImm(DataType::Float(32), trans(1)));
    }
    feature_row.push_back(thread_bind_len);

    Array<PrimExpr> alloc_features{
      StringImm("allocation_features"),
      FloatImm(DataType::Float(32), trans(fea.num_allocation)),
    };
    for (int i = 0; i < std::min(10, int(fea.output_buffer_size.size())); i++)
      alloc_features.push_back(FloatImm(DataType::Float(32), trans(fea.output_buffer_size[i])));
    for (int i = 0; i < 10 - int(fea.output_buffer_size.size()); i++) 
      alloc_features.push_back(FloatImm(DataType::Float(32), trans(0)));
    feature_row.push_back(alloc_features);

    feature_row.push_back(Array<PrimExpr>{
        StringImm("other_features"),
        FloatImm(DataType::Float(32), trans(fea.num_outer_loops)),
        FloatImm(DataType::Float(32), trans(fea.prod_outer_loops)),
        FloatImm(DataType::Float(32), trans(fea.auto_unroll_max_step)),
    });

    ret_feature->push_back(feature_row);
  }
}


void GetInnerStatementFeatureFlatten(
  Stmt stmt, bool take_log, 
  Array<Array<FloatImm>> *ret_feature, 
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
    Array<FloatImm> feature_vec;

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
        feature_vec.push_back(FloatImm(DataType::Float(32), j == v.access_type));
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(v.bytes)));
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(v.unique_bytes)));
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(v.lines)));
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(v.unique_lines)));
      for (auto j = 0; j < 4; j++)  // one-hot encoding
        feature_vec.push_back(FloatImm(DataType::Float(32), j == v.reuse_type));
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(v.reuse_distance)));
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(v.reuse_counter)));
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(v.stride)));
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(v.topdown)));
    }

    for (auto i = 0; i < 5 - int(bufs.size()); i++)
      for (auto j = 0; j < 16; j++)
        feature_vec.push_back(FloatImm(DataType::Float(32), 0));

    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.int_add_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.int_sub_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.int_mul_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.int_div_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.int_mod_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.int_cmp_ct)));

    for (auto k: INTRIN_KEYS) {
      if (fea.int_intrin_ct.count(k))
        feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.int_intrin_ct[k])));
      else
        feature_vec.push_back(FloatImm(DataType::Float(32), trans(0)));
    }

    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.flt_add_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.flt_sub_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.flt_mul_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.flt_div_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.flt_mod_ct)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.flt_cmp_ct)));

    for (auto k: INTRIN_KEYS) {
      if (fea.flt_intrin_ct.count(k))
        feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.flt_intrin_ct[k])));
      else
        feature_vec.push_back(FloatImm(DataType::Float(32), trans(0)));
    }

    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.vectorize_len_imost)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.vectorize_len_prod)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.vectorize_loop_num)));
    for (auto j = 0; j < 8; j++)
        feature_vec.push_back(FloatImm(DataType::Float(32), fea.vectorize_loop_pos == j));

    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.unroll_len_imost)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.unroll_len_prod)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.unroll_loop_num)));
    for (auto j = 0; j < 8; j++)
        feature_vec.push_back(FloatImm(DataType::Float(32), fea.unroll_loop_pos == j));

    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.parallel_len_imost)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.parallel_len_prod)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.parallel_loop_num)));
    for (auto j = 0; j < 8; j++)
        feature_vec.push_back(FloatImm(DataType::Float(32), fea.parallel_loop_pos == j));

    for (auto k : THREAD_BIND_KEYS) {
      if (fea.thread_bind_len.count(k))
        feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.thread_bind_len[k])));
      else
        feature_vec.push_back(FloatImm(DataType::Float(32), trans(1)));
    }

    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.num_allocation)));
    for (int i = 0; i < std::min(10, int(fea.output_buffer_size.size())); i++)
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.output_buffer_size[i])));
    for (int i = 0; i < 10 - int(fea.output_buffer_size.size()); i++) 
      feature_vec.push_back(FloatImm(DataType::Float(32), trans(0)));

    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.num_outer_loops)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.prod_outer_loops)));
    feature_vec.push_back(FloatImm(DataType::Float(32), trans(fea.auto_unroll_max_step)));

    ret_feature->push_back(feature_vec);
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
  Array<Array<FloatImm>> ret_feature;

  GetInnerStatementFeatureFlatten(stmt, take_log, &ret_feature, out_binds);

  // TODO: cast ret_feature into a byte array
  /* TVMByteArray arr;
  arr.size = sizeof(float) * ret_feature.size();
  arr.data = reinterpret_cast<char *>(ret_feature.data()); */
  *ret = ret_feature;  // arr
});

}  // namespace autotvm
}  // namespace tvm
