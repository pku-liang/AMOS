#include "arg_util.h"


namespace tvm {

namespace tg {

bool NameGenerator::has_name(std::string &name) {
  return name_map_.count(name) != 0;
}


std::string NameGenerator::unique_name(const std::string hint) {
  std::ostringstream oss;
  oss.str("");
  if (name_map_.find(hint) == name_map_.end()) {
    name_map_[hint] = 0;
  } else {
    name_map_[hint]++;
  }
  
  oss << hint.c_str();
  oss << name_map_[hint];
  std::string ret = oss.str();
  return ret;
}


int SubstituteContext::find_bound(PrimExpr &expr) {
  CheckExprEqual cee;
  int i = 0;
  for (auto kv : expr2var) {
    if (cee.check_equal(expr, kv.first)) {
      return i;
    }
    ++i;
  }
  return -1;
}


std::string SubstituteContext::get_bound_name(PrimExpr &expr) {
  int id = find_bound(expr);
  if (id < 0) {
    return "";
  } else {
    return expr2var[id].second;
  }
}


void SubstituteContext::add(std::string &name, Var var, PrimExpr expr, ExtRange range) {
  CHECK(var2expr.count(name) == 0) << "Internal error: variable of the same name: " << name << "\n";
  if (bound_begin < 0) {
    bound_begin = (int)index_names.size();
  }
  index_names.push_back(name);
  var_map[name] = var;
  range_map[name] = range;
  var2expr[name] = expr;
  expr2var.push_back(std::make_pair(expr, name));
}


PrimExpr EliminateFloorDivAndMod::VisitExpr_(const FloorDivNode* op) {
  PrimExpr new_a = Simplify(VisitExpr(op->a));
  PrimExpr new_b = Simplify(VisitExpr(op->b));
  const VarNode *a_as_var = new_a.as<VarNode>();
  const IntImmNode *b_as_int = new_b.as<IntImmNode>();
  CHECK(b_as_int != nullptr) << "Only support floor_div on type int, but find " << new_b.dtype() << "\n";

  PrimExpr new_div;
  std::string new_name;
  if (a_as_var == nullptr) {
    // bound left expr to unique var
    new_name = context_.get_bound_name(new_a);
    Var new_var = Var(new_name);
    if (new_name == "") {
      // not found
      new_name = name_generator_.unique_name(substitute_name_hint_);
      new_var = Var(new_name);
      // we don't infer the range of expression
      // TODO: infer range for expression
      context_.add(new_name, new_var, new_a, ExtRange());
    }
    new_div = FloorDivNode::make(new_var, new_b);
  } else {
    // left expr is already a var
    new_name = a_as_var->name_hint;
    new_div = FloorDivNode::make(new_a, new_b);
  }
  
  // check if this new div expr is bounded
  // check by structure equal
  std::string bound_this = context_.get_bound_name(new_div);
  if (bound_this == "") {
    // not bound
    std::string new_div_name = name_generator_.unique_name(substitute_name_hint_);
    // we should know the left var
    Var new_var = Var(new_div_name);
    context_.add(new_div_name, new_var,
        new_div, context_.range_map[new_name].floor_div((int)b_as_int->value));
    return std::move(new_var);
  } else {
    // bound
    return context_.var_map[bound_this];
  }
}


PrimExpr EliminateFloorDivAndMod::VisitExpr_(const FloorModNode* op) {
  PrimExpr new_a = Simplify(VisitExpr(op->a));
  PrimExpr new_b = Simplify(VisitExpr(op->b));
  const VarNode *a_as_var = new_a.as<VarNode>();
  const IntImmNode *b_as_int = new_b.as<IntImmNode>();
  CHECK(b_as_int != nullptr) << "Only support floor_mod on type int, but find " << new_b.dtype() << "\n";

  PrimExpr new_mod;
  std::string new_name;
  if (a_as_var == nullptr) {
    // bound left expr to unique var
    new_name = context_.get_bound_name(new_a);
    Var new_var = Var(new_name);
    if (new_name == "") {
      // not found
      new_name = name_generator_.unique_name(substitute_name_hint_);
      new_var = Var(new_name);
      // we don't infer the range of expression
      // TODO: infer range for expression
      context_.add(new_name, new_var, new_a, ExtRange());
    }
    new_mod = FloorModNode::make(new_var, new_b);
  } else {
    // left expr is already a var
    new_mod = FloorModNode::make(new_a, new_b);
  }
  
  // check if this new mod expr is bounded
  // check by structure equal
  std::string bound_this = context_.get_bound_name(new_mod);
  if (bound_this == "") {
    // not bound
    std::string new_mod_name = name_generator_.unique_name(substitute_name_hint_);
    Var new_var = Var(new_mod_name);
    // we should know the left var
    context_.add(new_mod_name, new_var,
        new_mod, context_.range_map[new_name].floor_mod((int)b_as_int->value));
    return std::move(new_var);
  } else {
    // bound
    return context_.var_map[bound_this];
  }
}


void ExtractCoefficient::VisitExpr_(const AddNode* op) {
  std::unordered_map<std::string, int> tmp1;
  scope_.push_back(&tmp1);
  VisitExpr(op->a);

  std::unordered_map<std::string, int> tmp2;
  scope_.push_back(&tmp2);
  VisitExpr(op->b);

  scope_.pop_back();
  scope_.pop_back();

  // merge results from two branches
  for (auto kv : tmp1) {
    if ((*(scope_.back())).count(kv.first)) {
      (*(scope_.back()))[kv.first] += kv.second;
    } else {
      (*(scope_.back()))[kv.first] = kv.second;
    }
  }
  for (auto kv : tmp2) {
    if ((*(scope_.back())).count(kv.first)) {
      (*(scope_.back()))[kv.first] += kv.second;
    } else {
      (*(scope_.back()))[kv.first] = kv.second;
    }
  }
}


void ExtractCoefficient::VisitExpr_(const SubNode* op) {
  std::unordered_map<std::string, int> tmp1;
  scope_.push_back(&tmp1);
  VisitExpr(op->a);

  std::unordered_map<std::string, int> tmp2;
  scope_.push_back(&tmp2);
  VisitExpr(op->b);

  scope_.pop_back();
  scope_.pop_back();

  // merge results from two branches
  for (auto kv : tmp1) {
    if ((*(scope_.back())).count(kv.first)) {
      (*(scope_.back()))[kv.first] += kv.second;
    } else {
      (*(scope_.back()))[kv.first] = kv.second;
    }
  }
  for (auto kv : tmp2) {
    if ((*(scope_.back())).count(kv.first)) {
      (*(scope_.back()))[kv.first] -= kv.second;
    } else {
      (*(scope_.back()))[kv.first] = -kv.second;
    }
  }
}


void ExtractCoefficient::VisitExpr_(const MulNode* op) {
  std::unordered_map<std::string, int> tmp1;
  scope_.push_back(&tmp1);
  VisitExpr(op->a);

  std::unordered_map<std::string, int> tmp2;
  scope_.push_back(&tmp2);
  VisitExpr(op->b);

  scope_.pop_back();
  scope_.pop_back();

  // merge results from two branches
  for (auto kv1 : tmp1) {
    for (auto kv2 : tmp2) {
      if (kv1.first == const_tag_) {
        if ((*(scope_.back())).count(kv2.first)) {
          (*(scope_.back()))[kv2.first] += kv1.second * kv2.second;
        } else {
          (*(scope_.back()))[kv2.first] = kv1.second * kv2.second;
        }
      } else if (kv2.first == const_tag_) {
        if ((*(scope_.back())).count(kv1.first)) {
          (*(scope_.back()))[kv1.first] += kv1.second * kv2.second;
        } else {
          (*(scope_.back()))[kv1.first] = kv1.second * kv2.second;
        }
      } else {
        LOG(FATAL) << "Find index multiply: " << GetRef<PrimExpr>(op); throw;
      }
    }
  }
}


void ExtractCoefficient::VisitExpr_(const IntImmNode* op) {
  (*(scope_.back()))[const_tag_] = (int)op->value;
}


FloorDivModEntry FloorDivModEntry::merge(const FloorDivModEntry &other) const {
  CHECK((*this) == other) << "Can't handle different entry.\n";
  std::string new_first;
  std::string new_second;
  if (first != "" && other.first == "") {
    new_first = first;
  } else if (first == "" && other.first != "") {
    new_first = other.first;
  } else {
    LOG(FATAL) << "Unexpected conflict.\n";
  }

  if (second != "" && other.second == "") {
    new_second = second;
  } else if (second == "" && other.second != "") {
    new_second = other.second;
  } else {
    LOG(FATAL) << "Unexpected conflict.\n";
  }

  return FloorDivModEntry(factor, var_name, new_first, new_second);
}


PrimExpr flatten_axes(Array<PrimExpr> args, Array<PrimExpr> shape) {
  CHECK(args.size() == shape.size()) << "Shape mismatch with args.";
  int num_args = (int)args.size();
  PrimExpr ret = args[num_args-1];
  PrimExpr product = 1;
  for (int i = num_args - 2; i >= 0; --i) {
    product = product * shape[i + 1];
    ret = ret + args[i] * product;
  }
  return Simplify(ret);
}


void solve_floor_div_mod(const SubstituteContext &context,
  std::unordered_set<FloorDivModEntry, FloorDivModEntryHash> &s) {
  for (auto kv : context.var2expr) {
    FloorDivModEntry entry;
    const FloorDivNode *as_div = kv.second.as<FloorDivNode>();
    const FloorModNode *as_mod = kv.second.as<FloorModNode>();
    CHECK(as_div != nullptr || as_mod != nullptr) << "Only can solve floor_div or floor_mod now.\n";
    if (as_div != nullptr) {
      const IntImmNode *as_int = as_div->b.as<IntImmNode>();
      const VarNode *as_var = as_div->a.as<VarNode>();
      CHECK(as_var != nullptr) << "Div must be on variable.\n";
      CHECK(as_int != nullptr) << "Div factor must be int type.\n";
      entry.factor = (int)as_int->value;
      entry.first = kv.first;
      entry.var_name = as_var->name_hint;
    } else {
      const IntImmNode *as_int = as_mod->b.as<IntImmNode>();
      const VarNode *as_var = as_mod->a.as<VarNode>();
      CHECK(as_var != nullptr) << "Mod must be on variable.\n";
      CHECK(as_int != nullptr) << "Mod factor must be int type.\n";
      entry.factor = (int)as_int->value;
      entry.second = kv.first;
      entry.var_name = as_var->name_hint;
    }
    auto it = s.find(entry);
    if (it != s.end()) {
      FloorDivModEntry new_entry = entry.merge(*it);
      s.erase(it);
      s.insert(new_entry);
    } else {
      s.insert(entry);
    }
  }
}


PrimExpr solve_multi_bindings(SubstituteContext &context, std::vector<PrimExpr> &bindings,
    std::unordered_set<std::string> &unused, Array<PrimExpr> &conditions) {
  int num_bindings = (int)bindings.size();
  CHECK(num_bindings > 0) << "Internal error: empty bindings.\n";
  int res_pos = -1;
  PrimExpr res;
  for (int i = 0; i < num_bindings; ++i) {
    const VarNode *as_var = bindings[i].as<VarNode>();
    if (as_var != nullptr) {
      CHECK(context.range_map.count(as_var->name_hint) != 0) << "Internal error: unknown var: "
                                                             << as_var->name_hint << ".\n";
      ExtRange range = context.range_map[as_var->name_hint];
      ExtRangeType range_type = range.range_type();
      if (range_type == ExtRangeType::LCRC) {
        res = bindings[i];
        res_pos = i;
        break;
      }
    } else {
      res = bindings[i];
      res_pos = i;
    }
  }

  // all the bindings are unbounded
  if (res_pos < 0) {
    res = bindings[0];
    res_pos = 0;
  }

  // the second pass
  // merge bindings
  for (int i = 0; i < num_bindings; ++i) {
    // skip res itself
    if (i == res_pos) {
      continue;
    }
    const VarNode *as_var = bindings[i].as<VarNode>();
    if (as_var != nullptr) {
      CHECK(context.range_map.count(as_var->name_hint) != 0) << "Internal error: unknown var: "
                                                             << as_var->name_hint << ".\n";
      ExtRange range = context.range_map[as_var->name_hint];
      ExtRangeType range_type = range.range_type();
      if (range_type == ExtRangeType::LORO) {
        // skip (-inf, +inf)
        unused.insert(as_var->name_hint);
      } else if (range_type == ExtRangeType::LORC) {
        // (-inf, val)
        // this shouldn't be index
        // conditions.push_back(LTNode::make(res, range.right));
        LOG(FATAL) << "Unexpected range : (-inf, " << range.right << ").\n";
        throw;
      } else if (range_type == ExtRangeType::LCRO) {
        // [val, +inf)
        // this shouldn't be index
        // conditions.push_back(GENode::make(res, range.left));
        LOG(FATAL) << "Unexpected range : [" << range.right << ", +inf).\n";
        throw;
      } else {
        // [val1, val2)
        // this should be index
        conditions.push_back(EQNode::make(res, bindings[i]));
      }
    }
  }

  return res;
}


void solve_substitutions(SubstituteContext &context,
  std::unordered_map<std::string, std::vector<PrimExpr>> &bindings,
  std::unordered_set<std::string> &unused,
  Array<PrimExpr> &conditions, std::unordered_map<std::string, PrimExpr> &result) {
  int num_index = (int)context.index_names.size();
  int end = context.bound_begin < 0 ? num_index : context.bound_begin;
  for (int i = num_index - 1; i >= end; --i) {
    std::string sub_var_name = context.index_names[i];
    CHECK(bindings.count(sub_var_name) != 0) << "Internal error: unknown substitution var: "
                                             << sub_var_name << ".\n";
    PrimExpr unique_binding = solve_multi_bindings(context, bindings[sub_var_name], unused, conditions);
    Map<Var, PrimExpr> vmap;
    vmap.Set(context.var_map[sub_var_name], unique_binding);
    // std::cout << "check solve sub var: " << sub_var_name << "\n";
    for (int j = i - 1; j >= 0; --j) {
      std::vector<PrimExpr> new_bindings;
      for (auto expr : bindings[context.index_names[j]]) {
        // std::cout << "target expr: " << expr << "\n";
        new_bindings.push_back(Simplify(Substitute(expr, vmap)));
      }
      // replace bindings
      bindings[context.index_names[j]] = new_bindings;
    }
  }
  // solve indice that are not substitution vars
  int beg = context.bound_begin < 0 ? num_index - 1 : context.bound_begin;
  for (int i = beg; i >= 0; --i) {
    std::string var_name = context.index_names[i];
    CHECK(bindings.count(var_name) != 0) << "Internal error: unknown var: "
                                             << var_name << ".\n";
    PrimExpr unique_binding = solve_multi_bindings(context, bindings[var_name], unused, conditions);
    result[var_name] = unique_binding;
  }
}


bool expr_equal(const PrimExpr &a, const PrimExpr &b) {
  CheckExprEqual cee;
  bool res = cee.check_equal(a, b);
  NameGenerator generator;
  SubstituteContext context;
  std::string subname_hint = "_s";
  EliminateFloorDivAndMod efdam(generator, subname_hint, context);
  PrimExpr new_expr_a = efdam.eliminate(a);
  // std::cout << "check:\n" << new_expr_a << "\n";
  // std::cout << context << "\n";
  PrimExpr new_expr_b = efdam.eliminate(b);
  // std::cout << "check:\n" << new_expr_b << "\n";
  // std::cout << context << "\n";
  // efdam.eliminate(a);
  // std::cout << context << "\n";
  // efdam.eliminate(b);
  // std::cout << context << "\n";

  // std::cout << "check get coefficients:\n";
  std::string const_tag = "const";
  ExtractCoefficient ec1(const_tag);
  ec1.do_extract(new_expr_a);
  // for (auto kv : ec1.coefficient_) {
  //   std::cout << kv.first << " : " << kv.second << "\n";
  // }
  // std::cout << "check get coefficients:\n";
  ExtractCoefficient ec2(const_tag);
  ec2.do_extract(new_expr_b);
  // for (auto kv : ec2.coefficient_) {
  //   std::cout << kv.first << " : " << kv.second << "\n";
  // }


  int M = 4;
  int N = 7;
  Matrix<int> trans(M, N);
  Matrix<int> U(M, M);
  Matrix<int> V(N, N);
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      trans[i][j] = 0;
    }
  }
  trans[0][0] = 1;
  trans[1][4] = 1;
  trans[2][2] = 1;
  trans[2][5] = 1;
  trans[3][3] = 1;
  trans[3][6] = 1;

  // std::cout << "trans before:\n";
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     std::cout << trans[i][j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // int dim = smith_normalize(trans, U, V);

  // std::cout << "trans after:\n";
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     std::cout << trans[i][j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // std::cout << "U after:\n";
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < M; ++j) {
  //     std::cout << U[i][j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // std::cout << "V after:\n";
  // for (int i = 0; i < N; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     std::cout << V[i][j] << " ";
  //   }
  //   std::cout << "\n";
  // }

  // std::cout << "dim= " << dim << "\n";
  
  return res;
}


TVM_REGISTER_GLOBAL("tg.expr_equal")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  // LOG(WARNING) << "te.expr_equal is an experimental feature.";
  *ret = expr_equal(args[0], args[1]);
});


}  // namespace tg

}  // namespace tvm