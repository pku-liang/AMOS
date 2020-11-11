#include "abstract_graph.h"


namespace tvm {

namespace tg {

PrimExpr ExprReMapper::VisitExpr_(const VarNode* op) {
  if (var_map.find(op) != var_map.end()) {
    return var_map[op];
  }
  Var ret = Var(get_new_var_name(), op->dtype);
  var_map[op] = ret;
  return ret;
}

PrimExpr ExprReMapper::VisitExpr_(const SizeVarNode* op) {
  if (size_var_map.find(op) != size_var_map.end()) {
    return size_var_map[op];
  }
  SizeVar ret = SizeVar(get_new_var_name(), op->dtype);
  size_var_map[op] = ret;
  return ret;
}


PrimExpr ExprReMapper::VisitExpr_(const ProducerLoadNode* op) {
  Array<PrimExpr> new_args;
  for (auto v : op->indices) {
    new_args.push_back(VisitExpr(v));
  }
  
  if (call_map.find(op->producer) != call_map.end()) {
    return ProducerLoad(call_map[op->producer], op->indices);
  } else {
    te::Tensor new_tensor = get_new_tensor(Downcast<te::Tensor>(op->producer));
    call_map[op->producer] = new_tensor;
    return ProducerLoad(new_tensor, op->indices);
  }
}


PrimExpr ExprReMapper::VisitExpr_(const ReduceNode* op) {
  CommReducer reducer;
  Array<Var> lhs;
  Array<Var> rhs;
  Array<PrimExpr> results;
  Array<PrimExpr> identities;
  for (Var l : op->combiner->lhs) {
    if (var_map.find(l.get()) != var_map.end()) {
      lhs.push_back(var_map[l.get()]);
    } else {
      VisitExpr(l);
      lhs.push_back(var_map[l.get()]);
    }
  }
  for (auto r : op->combiner->rhs) {
    if (var_map.find(r.get()) != var_map.end()) {
      rhs.push_back(var_map[r.get()]);
    } else {
      VisitExpr(r);
      rhs.push_back(var_map[r.get()]);
    }
  }
  for (auto r : op->combiner->result) {
    results.push_back(VisitExpr(r));
  }
  for (auto i : op->combiner->identity_element) {
    identities.push_back(VisitExpr(i));
  }
  reducer = CommReducer(lhs, rhs, results, identities);

  
  Array<PrimExpr> source;
  for (auto s : op->source) {
    source.push_back(VisitExpr(s));
  }

  Array<IterVar> axis;
  for (auto iv : op->axis) {
    VisitExpr(iv->var);
    axis.push_back(
      IterVar(iv->dom, var_map[iv->var.get()], iv->iter_type, iv->thread_tag));
  }

  PrimExpr condition = this->VisitExpr(op->condition);

  return Reduce(
    reducer, source, axis, condition, op->value_index, op->init);
}


std::string generate_tag_from_body(Array<IterVar> axis, Array<PrimExpr> body) {
  std::ostringstream oss;
  oss.str("");
  if (body.size() == 0U) {
    ERROR << "Unexpected empty body!";
  }

  const ReduceNode* as_reduce = body[0].as<ReduceNode>();

  if (as_reduce != nullptr) {
    CHECK(body.size() == 1U) << "Only support reduce with one body.";
    Array<IterVar> axis_;
    for (auto iv : axis) {
      axis_.push_back(iv);
    }
    for (auto iv : as_reduce->axis) {
      axis_.push_back(iv);
    }
    ExprReMapper remapper(axis_);
    PrimExpr new_reduce = remapper(body[0]);
    const ReduceNode* as_reduce = new_reduce.as<ReduceNode>();
    CHECK(as_reduce != nullptr);

    oss << "R[";
    bool add_colon = false;
    for (auto s : axis) {
      if (add_colon) {
        oss << ", ";
      } else {
        add_colon = true;
      }
      oss << s->dom->extent;
    }
    oss << "] [";
    add_colon = false;
    for (auto iv : as_reduce->axis) {
      if (add_colon) {
        oss << ", ";
      } else {
        add_colon = true;
      }
      oss << iv->dom->extent;
    }
    oss << "] { ";
    oss << as_reduce->combiner;
    oss << " } { ";
    for (size_t i = 0; i < as_reduce->source.size(); ++i) {
      if (i != 0) {
        oss << "; ";
      }
      oss << as_reduce->source[i];
    }
    oss << " }";
  } else {
    // not reduce
    oss << "S[";
    bool add_colon = false;
    for (auto s : axis) {
      if (add_colon) {
        oss << ", ";
      } else {
        add_colon = true;
      }
      oss << s->dom->extent;
    }
    oss << "] [ ] { } { ";
    bool add_semicolon = false;
    for (auto b : body) {
      CHECK(b.as<ReduceNode>() == nullptr) << "Should only contain non-reduce expr.";
      ExprReMapper remapper(axis);
      PrimExpr new_b = remapper(b);
      if (add_semicolon) {
        oss << "; ";
      } else {
        add_semicolon = true;
      }
      oss << new_b;
    }
    oss << " }";
  }

  return oss.str();
}



TVM_REGISTER_GLOBAL("tg.generate_tag_from_body")
.set_body_typed([](Array<IterVar> axis, Array<PrimExpr> body) {
  return generate_tag_from_body(axis, body);
});


}  // namespace tg

}  // namespace tvm