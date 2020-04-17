#include <tvm/te/myautodiff.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <topi/transform.h>
#include <memory>

#include "arith.h"
#include "arg_util.h"


namespace tvm {
namespace te {

#define NOT_IMPLEMENTED \
  { LOG(FATAL) << "Grad of this expr is not implemented: " << GetRef<PrimExpr>(op); throw; }


class GradOp : public ExprMutator {
 private:
  std::string const_tag_;
  std::string sub_hint_;
  std::string dummy_tag_;
  NameGenerator &generator_;
  SubstituteContext &context_;
  EliminateFloorDivAndMod *eliminator_;
  const Tensor &input_;
  const Tensor &doutput_;
  Array<PrimExpr> &call_args_;
  Array<PrimExpr> &compute_args_;
  std::vector<Map<Var, PrimExpr>> vmap_scope_;
  // bool first_met_;
 public:
  explicit GradOp(NameGenerator &generator, SubstituteContext &context, const Tensor &input,
    const Tensor &doutput, Array<PrimExpr> &call_args, Array<PrimExpr> &compute_args) :
    generator_(generator), context_(context), input_(input), doutput_(doutput),
    call_args_(call_args), compute_args_(compute_args) {
      const_tag_ = generator_.unique_name("_const");
      sub_hint_ = generator_.unique_name("_s");
      dummy_tag_ = generator_.unique_name("_r");
      eliminator_ = new EliminateFloorDivAndMod(generator_, sub_hint_, context_);
      // first_met_ = true;
    }
  
  ~GradOp() {
    if (eliminator_ != nullptr) {
      delete eliminator_;
    }
  }

  PrimExpr grad(PrimExpr e) {
    if (e.dtype().is_int() || e.dtype().is_uint()) {
      LOG(FATAL) << "No support for grad on int type.";
      throw;
    } else {
      return ExprMutator::VisitExpr(e);
    }
  }

  // PrimExpr VisitExpr_(const VarNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LoadNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LetNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const CallNode* op) {
    PrimExpr expr = GetRef<PrimExpr>(op);
    if (op->call_type == CallNode::CallType::Halide) {
      if (input_.get() && op->func.same_as(input_->op) &&
          op->value_index == input_->value_index) {
        // if (!first_met_) {
        //   // this is a quite difficult case
        //   // TVM doesn't allow ReduceNode + ReduceNode
        //   // so handling this case can be very hard
        //   LOG(FATAL) << "The input " << GetRef<ComputeOpNode>(input_->op) << " occurs more than one time"
        //              << " in " << GetRef<PrimExpr>(op) << ", this is not supported.\n";
        //   throw;
        // }
        // mark met
        // first_met_ = false;
        std::vector<std::unordered_map<std::string, int>> coeffs;
        // handle args
        for (const PrimExpr &arg : op->args) {
          // eliminate possible mod & div
          PrimExpr new_arg = eliminator_->eliminate(arg);
          // extract coefficients
          ExtractCoefficient extractor(const_tag_);
          extractor.do_extract(new_arg);
          coeffs.push_back(extractor.coefficient_);
        }

        std::cout << "check context after elimination:\n";
        std::cout << context_ << "\n";

        // assemble coefficents to Matrix
        int cols = (int)context_.index_names.size();
        int rows = (int)coeffs.size();
        Matrix<int> trans(rows, cols);
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
            if (coeffs[i].count(context_.index_names[j]) != 0) {
              // has the coefficent for this index
              trans[i][j] = coeffs[i][context_.index_names[j]];
            } else {
              trans[i][j] = 0;
            }
          }
        }

        std::cout << "check trans before:\n";
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
            std::cout << trans[i][j] << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";

        // compute simith normal form
        Matrix<int> U(rows, rows);
        Matrix<int> V(cols, cols);
        int dims = smith_normalize(trans, U, V);

        std::cout << "check dim=" << dims << "\n";

        std::cout << "check trans after:\n";
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < cols; ++j) {
            std::cout << trans[i][j] << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";

        std::cout << "check U:\n";
        for (int i = 0; i < rows; ++i) {
          for (int j = 0; j < rows; ++j) {
            std::cout << U[i][j] << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";

        std::cout << "check V:\n";
        for (int i = 0; i < cols; ++i) {
          for (int j = 0; j < cols; ++j) {
            std::cout << V[i][j] << " ";
          }
          std::cout << "\n";
        }
        std::cout << "\n";

        // check if is identity
        if (!check_identity(trans, dims)) {
          LOG(FATAL) << "Don't know how to handle non-identity matrix, waiting for more discussion...\n";
          throw;
        }

        // explain the results:
        Array<PrimExpr> Ub = relax_matrix_array_product(U, compute_args_);
        // unbounded bindings
        std::unordered_set<std::string> relaxes;
        // if cols > dims
        for (int i = 0; i < cols - dims; ++i) {
          std::string new_name = generator_.unique_name(dummy_tag_);
          relaxes.insert(new_name);
          Var v(new_name);
          Ub.push_back(v);
          context_.var_map[new_name] = v;
          // these vars are unbounded
          context_.range_map[new_name] = ExtRange();
        }

        std::cout << "check relaxes:\n";
        for (auto it : relaxes) {
          std::cout << it << " ";
        }
        std::cout << "\n\n";

        // bindings, transformation from original index to new index
        // one var may have many bindings
        // for example, i = r0, i = r1 * 4 + s0
        std::unordered_map<std::string, std::vector<PrimExpr>> bindings;
        Array<PrimExpr> VUb = relax_matrix_array_product(V, Ub);
        std::cout << "check VUb:\n";
        for (auto val : VUb) {
          std::cout << Simplify(val) << " ";
        }
        std::cout << "\n\n";
        for (int i = 0; i < cols; ++i) {
          if (bindings.count(context_.index_names[i]) > 0) {
            bindings[context_.index_names[i]].push_back(Simplify(VUb[i]));
          } else {
            bindings[context_.index_names[i]] = std::vector<PrimExpr>({Simplify(VUb[i])});
          }
        }
        Array<PrimExpr> conditions;
        // if rows > dims
        for (int i = dims; i < rows; ++i) {
          // must be zeros
          conditions.push_back(EQNode::make(Ub[i], 0));
        }

        std::cout << "check conditions:\n";
        for (auto it : conditions) {
          std::cout << it << " ";
        }
        std::cout << "\n";

        // solve the floor_div/mod substitution
        // e.g. s = i // 8 -> i = s * 8 + r0, r0: [0, 8)
        std::unordered_set<FloorDivModEntry, FloorDivModEntryHash> sub_set;
        solve_floor_div_mod(context_, sub_set);
        
        for (auto it : sub_set) {
          PrimExpr rhs;
          if (it.first == "") {
            std::string new_name = generator_.unique_name(dummy_tag_);
            Var v(new_name);
            rhs = v;
            context_.var_map[new_name] = v;
            relaxes.insert(new_name);
            CHECK(context_.range_map.count(it.var_name) != 0) << "We should know var: "
                                                               << it.var_name << ".\n";
            context_.range_map[new_name] = context_.range_map[it.var_name].floor_div(it.factor);
          } else {
            rhs = context_.var_map[it.first];
          }
          rhs = MulNode::make(rhs, it.factor);
          if (it.second == "") {
            std::string new_name = generator_.unique_name(dummy_tag_);
            Var v(new_name);
            rhs = AddNode::make(rhs, v);
            relaxes.insert(new_name);
            context_.var_map[new_name] = v;
            CHECK(context_.range_map.count(it.var_name) != 0) << "We should know var: "
                                                               << it.var_name << ".\n";
            context_.range_map[new_name] = context_.range_map[it.var_name].floor_mod(it.factor);
          } else {
            rhs = AddNode::make(rhs, context_.var_map[it.second]);
          }
          if (bindings.count(it.var_name) != 0) {
            bindings[it.var_name].push_back(rhs);
          } else {
            bindings[it.var_name] = std::vector<PrimExpr>({rhs});
          }
        }

        std::cout << "check original bindings:\n";
        for (auto kv : bindings) {
          std::cout << kv.first << " : [ ";
          for (auto val : kv.second) {
            std::cout << val << " "; 
          }
          std::cout << "]\n";
        }
        std::cout << "\n";

        // resolve the bindings
        std::unordered_map<std::string, PrimExpr> results;
        std::unordered_set<std::string> unused;
        solve_substitutions(context_, bindings, unused, conditions, results);
        // eliminate unused vars
        for (auto it : unused) {
          relaxes.erase(it);
        }
        // for the remaining vars, get the bounds
        for (auto kv : results) {
          const VarNode *as_var = kv.second.as<VarNode>();
          if (as_var != nullptr && relaxes.count(as_var->name_hint) != 0) {
            // we should know the lhs
            CHECK(context_.range_map.count(kv.first) != 0) << "Internal error: unknown var: "
                                                           << kv.first << ".\n";
            context_.range_map[as_var->name_hint] = context_.range_map[kv.first];
          }
        }
        for (auto kv : results) {
          // const VarNode *as_var = kv.second.as<VarNode>();
          // if (as_var != nullptr && relaxes.count(as_var->name_hint) != 0) {
          //   // we should know the lhs
          //   CHECK(context_.range_map.count(kv.first) != 0) << "Internal error: unknown var: "
          //                                                  << kv.first << ".\n";
          //   context_.range_map[as_var->name_hint] = context_.range_map[kv.first];
          // }
          RangeInference infer(context_.range_map[kv.first]);
          infer.do_infer(kv.second);
          std::cout << "check range inference:\n";
          for (auto kkv : infer.range_map) {
            std::cout << kkv.first << ": [" << kkv.second.left << ", " << kkv.second.right << ")\n";
            if (context_.range_map.count(kkv.first) == 0 ||
                context_.range_map[kkv.first].range_type() != ExtRangeType::LCRC) {
              if (kkv.second.range_type() == ExtRangeType::LCRC)
                context_.range_map[kkv.first] = kkv.second;
            }
          }

          // Put bound checkers
          // TODO: do not put unnecessary checkers
          if (kv.second.as<VarNode>() == nullptr) {
            CHECK(context_.range_map[kv.first].range_type() == ExtRangeType::LCRC);
            conditions.push_back(AndNode::make(
              GENode::make(kv.second, context_.range_map[kv.first].left),
              LTNode::make(kv.second, context_.range_map[kv.first].right)
            ));
          }
        }

        std::cout << "check bindings:\n";
        for (auto kv : results) {
          std::cout << kv.first << " = " << kv.second << "\n";
        }
        std::cout << "\n";

        // check if any var his no concrete range
        for (auto it : relaxes) {
          CHECK(context_.range_map.count(it) != 0) << "Internal error: fail to infer range for: "
                                                    << it << ".\n";
          CHECK(context_.range_map[it].range_type() == ExtRangeType::LCRC) << "Internel error"
              << ": only infer unbounded range for: " << it << ".\n";
        }

        // form final expr
        PrimExpr result_expr;
        // prepare condition
        PrimExpr result_condition = const_true();
        for (auto val : conditions) {
          result_condition = AndNode::make(result_condition, val);
        }
        // prepare source
        result_expr = CallNode::make(op->dtype,
                            doutput_->op->name,
                            call_args_,
                            CallNode::Halide,
                            doutput_->op,
                            doutput_->value_index);
        Map<Var, PrimExpr> vmap;
        for (auto kv : results) {
          vmap.Set(context_.var_map[kv.first], kv.second);
        }
        // add new vmap
        vmap_scope_.push_back(vmap);
        result_expr = Substitute(result_expr, vmap);
        result_expr = Simplify(SelectNode::make(result_condition, result_expr, make_const(result_expr.dtype(), 0)));
        // no need to produce a reduce
        if ((int)relaxes.size() == 0) {
          return result_expr;
        }

        // prepare axis
        Array<IterVar> new_axis;
        for (auto it : relaxes) {
          ExtRange range = context_.range_map[it];
          IterVar iv = IterVarNode::make(
                  Range(range.left, range.right), context_.var_map[it], kCommReduce);
          new_axis.push_back(iv);
        }

        // form reduce
        result_expr = sum(result_expr, new_axis);
        return result_expr;
      } else {
        return make_zero(op->dtype);
      }
    } else if (op->call_type == CallNode::CallType::PureIntrinsic) {
      static std::unordered_set<std::string> piecewise_const = {"floor", "ceil", "trunc", "round"};
      if (op->name == "exp") {
        return MulNode::make(grad(op->args[0]), expr);
      } else if (op->name == "log") {
        return DivNode::make(grad(op->args[0]), op->args[0]);
      } else if (op->name == "sigmoid") {
        return MulNode::make(grad(op->args[0]),
                             MulNode::make(expr, SubNode::make(FloatImm(expr.dtype(), 1.0), expr)));
      } else if (op->name == "sqrt") {
        return DivNode::make(grad(op->args[0]),
                             MulNode::make(expr, FloatImm(expr.dtype(), 2.0)));
      } else if (op->name == "tanh") {
        return MulNode::make(grad(op->args[0]),
                             SubNode::make(FloatImm(expr.dtype(), 1.0), MulNode::make(expr, expr)));
      } else if (op->name == "pow") {
        auto x = op->args[0], y = op->args[1];
        return expr * (grad(y)*log(x) + grad(x)*y/x);
      } else if (op->name == "fabs") {
        auto type = op->args[0].dtype();
        return MulNode::make(grad(op->args[0]),
                             SelectNode::make(GENode::make(op->args[0], make_zero(type)),
                                              FloatImm(type, 1.0), FloatImm(type, -1.0)));
      } else if (op->name == intrinsic::tvm_if_then_else) {
        Array<PrimExpr> new_args = {op->args[0],
                                    grad(op->args[1]),
                                    grad(op->args[2])};
        return CallNode::make(op->dtype, op->name, new_args,
                              op->call_type, op->func, op->value_index);
      } else if (piecewise_const.count(op->name)) {
        return FloatImm(expr.dtype(), 0.0);
      } else {
        throw dmlc::Error("Derivative of this intrinsic is not implemented: " + op->name);
      }
    }
    NOT_IMPLEMENTED
  }

  PrimExpr VisitExpr_(const AddNode* op) {
    return AddNode::make(grad(op->a), grad(op->b));
  }

  PrimExpr VisitExpr_(const SubNode* op) {
    return SubNode::make(grad(op->a), grad(op->b));
  }

  PrimExpr VisitExpr_(const MulNode* op) {
    vmap_scope_.clear();
    PrimExpr new_a = grad(op->a);
    if (vmap_scope_.size() != 0) {
      new_a = MulNode::make(new_a, Substitute(op->b, vmap_scope_.back()));
    } else {
      new_a = MulNode::make(new_a, op->b);
    }

    vmap_scope_.clear();
    PrimExpr new_b = grad(op->b);
    if (vmap_scope_.size() != 0) {
      new_b = MulNode::make(Substitute(op->a, vmap_scope_.back()), new_b);
    } else {
      new_b = MulNode::make(op->a, new_b);
    }
    return AddNode::make(new_a, new_b);
  }

  PrimExpr VisitExpr_(const DivNode* op) {
    vmap_scope_.clear();
    PrimExpr new_a = grad(op->a);
    PrimExpr sub_b = op->b;
    if (vmap_scope_.size() != 0) {
      sub_b = Substitute(sub_b, vmap_scope_.back());
    }

    vmap_scope_.clear();
    PrimExpr new_b = grad(op->b);
    PrimExpr sub_a = op->a;
    if (vmap_scope_.size() != 0) {
      sub_a = Substitute(sub_a, vmap_scope_.back());
    }

    return DivNode::make(
        SubNode::make(
            MulNode::make(new_a, sub_b),
            MulNode::make(sub_a, new_b)),
        MulNode::make(sub_a, sub_b));
  }

  PrimExpr VisitExpr_(const ModNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const FloorDivNode* op) {
    vmap_scope_.clear();
    PrimExpr new_a = grad(op->a);
    PrimExpr sub_b = op->b;
    if (vmap_scope_.size() != 0) {
      sub_b = Substitute(sub_b, vmap_scope_.back());
    }

    vmap_scope_.clear();
    PrimExpr new_b = grad(op->b);
    PrimExpr sub_a = op->a;
    if (vmap_scope_.size() != 0) {
      sub_a = Substitute(sub_a, vmap_scope_.back());
    }

    return FloorDivNode::make(
        SubNode::make(
            MulNode::make(new_a, sub_b),
            MulNode::make(sub_a, new_b)),
        MulNode::make(sub_a, sub_b));
  }

  PrimExpr VisitExpr_(const FloorModNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const MinNode* op) {
    vmap_scope_.clear();
    PrimExpr new_a = grad(op->a);
    PrimExpr sub_b = op->b;
    if (vmap_scope_.size() != 0) {
      sub_b = Substitute(sub_b, vmap_scope_.back());
    }

    vmap_scope_.clear();
    PrimExpr new_b = grad(op->b);
    PrimExpr sub_a = op->a;
    if (vmap_scope_.size() != 0) {
      sub_a = Substitute(sub_a, vmap_scope_.back());
    }

    return SelectNode::make(LENode::make(sub_a, sub_b),
        new_a, new_b);
  }

  PrimExpr VisitExpr_(const MaxNode* op) {
    vmap_scope_.clear();
    PrimExpr new_a = grad(op->a);
    PrimExpr sub_b = op->b;
    if (vmap_scope_.size() != 0) {
      sub_b = Substitute(sub_b, vmap_scope_.back());
    }

    vmap_scope_.clear();
    PrimExpr new_b = grad(op->b);
    PrimExpr sub_a = op->a;
    if (vmap_scope_.size() != 0) {
      sub_a = Substitute(sub_a, vmap_scope_.back());
    }

    return SelectNode::make(GENode::make(sub_a, sub_b),
        new_a, new_b);
  }

  PrimExpr VisitExpr_(const EQNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const NENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LTNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const LENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const GTNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const GENode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const AndNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const OrNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const ReduceNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const CastNode* op) {
    if (op->dtype.is_float()) {
      return CastNode::make(op->dtype, grad(op->value));
    } else {
      return make_zero(op->dtype);
    }
  }

  PrimExpr VisitExpr_(const NotNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const SelectNode* op) {
    vmap_scope_.clear();
    PrimExpr new_true = grad(op->true_value);
    // TODO: we do not support grad in condition
    PrimExpr sub_cond = op->condition;
    if (vmap_scope_.size() != 0) {
      sub_cond = Substitute(sub_cond, vmap_scope_.back());
    }

    vmap_scope_.clear();
    PrimExpr new_false = grad(op->false_value);
    if (vmap_scope_.size() != 0) {
      sub_cond = Substitute(sub_cond, vmap_scope_.back());
    }

    return SelectNode::make(sub_cond,
        new_true, new_false);
  }

  PrimExpr VisitExpr_(const RampNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const BroadcastNode* op) NOT_IMPLEMENTED
  PrimExpr VisitExpr_(const ShuffleNode* op) NOT_IMPLEMENTED

  PrimExpr VisitExpr_(const IntImmNode* op) {
    return IntImm(op->dtype, 0);
  }

  PrimExpr VisitExpr_(const FloatImmNode* op) {
    return FloatImm(op->dtype, 0);
  }

  PrimExpr VisitExpr_(const StringImmNode* op) NOT_IMPLEMENTED
};


class LiftReduce : public ExprMutator {
 private:
 public:
  PrimExpr lift(const PrimExpr &expr) {
    return VisitExpr(expr);
  }
 protected:
  // list of functions to override.
  // PrimExpr VisitExpr_(const VarNode* op) override;
  // PrimExpr VisitExpr_(const SizeVarNode* op) override;
  // PrimExpr VisitExpr_(const LoadNode* op) override;
  // PrimExpr VisitExpr_(const BufferLoadNode* op) override;
  // PrimExpr VisitExpr_(const LetNode* op) override;
  // PrimExpr VisitExpr_(const CallNode* op) override;

  PrimExpr VisitExpr_(const AddNode* op) override {
    PrimExpr new_a = VisitExpr(op->a);
    PrimExpr new_b = VisitExpr(op->b);
    const ReduceNode *a_as_red = new_a.as<ReduceNode>();
    const ReduceNode *b_as_red = new_b.as<ReduceNode>();
    if (a_as_red != nullptr && b_as_red != nullptr) {
      // see if the reduce axis can be merged
      // only support one source
      // TODO: check combiner
      if (a_as_red->axis.size() == b_as_red->axis.size() &&
        (int)a_as_red->source.size() == 1 && (int)b_as_red->source.size() == 1) {
        // TODO: we still can't identify such equality
        // a: reduce axis r0:[0, 8], r1: [0, r0]
        // b: reduce axis r2 [0, 8], r3: [0, r2]
        CheckExprEqual cee;
        std::unordered_set<size_t> visit;
        std::vector<size_t> pos_map;
        // empty axis, equal
        bool equal = true;
        for (size_t i = 0; i < a_as_red->axis.size(); ++i) {
          // reset to not equal
          equal = false;
          for (size_t j = 0; j < b_as_red->axis.size(); ++j) {
            if (visit.count(j) != 0) {
              continue;
            } else if (cee(a_as_red->axis[i]->dom->min, b_as_red->axis[j]->dom->min) &&
                cee(a_as_red->axis[i]->dom->extent, b_as_red->axis[j]->dom->extent)) {
              visit.insert(j);
              pos_map.push_back(j);
              // only if found the same axis
              equal = true;
              break;
            }
          }
          // no match
          if (!equal) {
            break;
          }
        }
        // only when reduce axes are the same
        if (equal) {
          Map<Var, PrimExpr> vmap;
          size_t i = 0;
          for (auto iv : a_as_red->axis) {
            vmap.Set(b_as_red->axis[pos_map[i]]->var, iv->var);
            ++i;
          }
          Array<PrimExpr> new_source;
          i = 0;
          for (auto expr : a_as_red->source) {
            new_source.push_back(
              Substitute(AddNode::make(expr, b_as_red->source[i]), vmap));
            ++i;
          }
          return ReduceNode::make(
            a_as_red->combiner,
            new_source,
            a_as_red->axis,
            a_as_red->condition,
            a_as_red->value_index
          );
        }
      }
    }
    return Simplify(AddNode::make(new_a, new_b));
  }

  PrimExpr VisitExpr_(const SubNode* op) override {
    PrimExpr new_a = VisitExpr(op->a);
    PrimExpr new_b = VisitExpr(op->b);
    const ReduceNode *a_as_red = new_a.as<ReduceNode>();
    const ReduceNode *b_as_red = new_b.as<ReduceNode>();
    if (a_as_red != nullptr && b_as_red != nullptr) {
      // see if the reduce axis can be merged
      // only support one source
      // TODO: check combiner
      if (a_as_red->axis.size() == b_as_red->axis.size() &&
        (int)a_as_red->source.size() == 1 && (int)b_as_red->source.size() == 1) {
        // TODO: we still can't identify such equality
        // a: reduce axis r0:[0, 8], r1: [0, r0]
        // b: reduce axis r2 [0, 8], r3: [0, r2]
        CheckExprEqual cee;
        std::unordered_set<size_t> visit;
        std::vector<size_t> pos_map;
        // empty axis, equal
        bool equal = true;
        for (size_t i = 0; i < a_as_red->axis.size(); ++i) {
          // reset to not equal
          equal = false;
          for (size_t j = 0; j < b_as_red->axis.size(); ++j) {
            if (visit.count(j) != 0) {
              continue;
            } else if (cee(a_as_red->axis[i]->dom->min, b_as_red->axis[j]->dom->min) &&
                cee(a_as_red->axis[i]->dom->extent, b_as_red->axis[j]->dom->extent)) {
              visit.insert(j);
              pos_map.push_back(j);
              // only if found the same axis
              equal = true;
              break;
            }
          }
          // no match
          if (!equal) {
            break;
          }
        }
        // only when reduce axes are the same
        if (equal) {
          Map<Var, PrimExpr> vmap;
          size_t i = 0;
          for (auto iv : a_as_red->axis) {
            std::cout << "check temp b iv=" << b_as_red->axis[pos_map[i]]->var << ", a iv=" << iv->var << "\n";
            vmap.Set(b_as_red->axis[pos_map[i]]->var, iv->var);
            ++i;
          }
          Array<PrimExpr> new_source;
          i = 0;
          for (auto expr : a_as_red->source) {
            new_source.push_back(
              Substitute(SubNode::make(expr, b_as_red->source[i]), vmap));
            ++i;
          }
          return ReduceNode::make(
            a_as_red->combiner,
            new_source,
            a_as_red->axis,
            a_as_red->condition,
            a_as_red->value_index
          );
        }
      }
    }
    return Simplify(SubNode::make(new_a, new_b));
  }

  PrimExpr VisitExpr_(const MulNode* op) override {
    PrimExpr new_a = VisitExpr(op->a);
    PrimExpr new_b = VisitExpr(op->b);
    const ReduceNode *a_as_red = new_a.as<ReduceNode>();
    const ReduceNode *b_as_red = new_b.as<ReduceNode>();
    if (a_as_red != nullptr && b_as_red == nullptr) {
      Array<PrimExpr> new_source;
      for (auto expr : a_as_red->source) {
        new_source.push_back(
          MulNode::make(expr, new_b));
      }
      return ReduceNode::make(
        a_as_red->combiner,
        new_source,
        a_as_red->axis,
        a_as_red->condition,
        a_as_red->value_index
      );
    } else if (a_as_red == nullptr && b_as_red != nullptr) {
      Array<PrimExpr> new_source;
      for (auto expr : b_as_red->source) {
        new_source.push_back(
          MulNode::make(new_a, expr));
      }
      return ReduceNode::make(
        b_as_red->combiner,
        new_source,
        b_as_red->axis,
        b_as_red->condition,
        b_as_red->value_index
      );
    }
    return Simplify(MulNode::make(new_a, new_b));
  }

  PrimExpr VisitExpr_(const DivNode* op) override {
    PrimExpr new_a = VisitExpr(op->a);
    PrimExpr new_b = VisitExpr(op->b);
    const ReduceNode *a_as_red = new_a.as<ReduceNode>();
    const ReduceNode *b_as_red = new_b.as<ReduceNode>();
    if (a_as_red != nullptr && b_as_red == nullptr) {
      Array<PrimExpr> new_source;
      for (auto expr : a_as_red->source) {
        new_source.push_back(
          DivNode::make(expr, new_b));
      }
      return ReduceNode::make(
        a_as_red->combiner,
        new_source,
        a_as_red->axis,
        a_as_red->condition,
        a_as_red->value_index
      );
    }
    return Simplify(DivNode::make(new_a, new_b));
  }

  // PrimExpr VisitExpr_(const ModNode* op) override;
  // PrimExpr VisitExpr_(const FloorDivNode* op) override;
  // PrimExpr VisitExpr_(const FloorModNode* op) override;
  // PrimExpr VisitExpr_(const MinNode* op) override;
  // PrimExpr VisitExpr_(const MaxNode* op) override;
  // PrimExpr VisitExpr_(const EQNode* op) override;
  // PrimExpr VisitExpr_(const NENode* op) override;
  // PrimExpr VisitExpr_(const LTNode* op) override;
  // PrimExpr VisitExpr_(const LENode* op) override;
  // PrimExpr VisitExpr_(const GTNode* op) override;
  // PrimExpr VisitExpr_(const GENode* op) override;
  // PrimExpr VisitExpr_(const AndNode* op) override;
  // PrimExpr VisitExpr_(const OrNode* op) override;
  // PrimExpr VisitExpr_(const ReduceNode* op) override;

  PrimExpr VisitExpr_(const CastNode* op) override {
    PrimExpr new_val = VisitExpr(op->value);
    const ReduceNode *as_red = new_val.as<ReduceNode>();
    if (as_red != nullptr) {
      Array<PrimExpr> new_source;
      for (auto expr : as_red->source) {
        new_source.push_back(CastNode::make(op->dtype, expr));
      }
      return ReduceNode::make(
        as_red->combiner,
        new_source,
        as_red->axis,
        as_red->condition,
        as_red->value_index
      );
    }
    return Simplify(CastNode::make(op->dtype, op->value));
  }

  // PrimExpr VisitExpr_(const NotNode* op) override;

  PrimExpr VisitExpr_(const SelectNode* op) override {
    PrimExpr new_cond = VisitExpr(op->condition);
    PrimExpr new_true = VisitExpr(op->true_value);
    PrimExpr new_false = VisitExpr(op->false_value);
    const ReduceNode *a_as_red = new_true.as<ReduceNode>();
    const ReduceNode *b_as_red = new_false.as<ReduceNode>();
    if (a_as_red != nullptr && b_as_red != nullptr) {
      // see if the reduce axis can be merged
      // only support one source
      // TODO: check combiner
      if (a_as_red->axis.size() == b_as_red->axis.size() &&
        (int)a_as_red->source.size() == 1 && (int)b_as_red->source.size() == 1) {
        // TODO: we still can't identify such equality
        // a: reduce axis r0:[0, 8], r1: [0, r0]
        // b: reduce axis r2 [0, 8], r3: [0, r2]
        CheckExprEqual cee;
        std::unordered_set<size_t> visit;
        std::vector<size_t> pos_map;
        // empty axis, equal
        bool equal = true;
        for (size_t i = 0; i < a_as_red->axis.size(); ++i) {
          // reset to not equal
          equal = false;
          for (size_t j = 0; j < b_as_red->axis.size(); ++j) {
            if (visit.count(j) != 0) {
              continue;
            } else if (cee(a_as_red->axis[i]->dom->min, b_as_red->axis[j]->dom->min) &&
                cee(a_as_red->axis[i]->dom->extent, b_as_red->axis[j]->dom->extent)) {
              visit.insert(j);
              pos_map.push_back(j);
              // only if found the same axis
              equal = true;
              break;
            }
          }
          // no match
          if (!equal) {
            break;
          }
        }
        // only when reduce axes are the same
        if (equal) {
          Map<Var, PrimExpr> vmap;
          size_t i = 0;
          for (auto iv : a_as_red->axis) {
            vmap.Set(b_as_red->axis[pos_map[i]]->var, iv->var);
            ++i;
          }
          Array<PrimExpr> new_source;
          i = 0;
          for (auto expr : a_as_red->source) {
            new_source.push_back(
              Substitute(SelectNode::make(new_cond, expr, b_as_red->source[i]), vmap));
            ++i;
          }
          return ReduceNode::make(
            a_as_red->combiner,
            new_source,
            a_as_red->axis,
            a_as_red->condition,
            a_as_red->value_index
          );
        }
      }
    }
    return Simplify(SelectNode::make(new_cond, new_true, new_false));
  }

  // PrimExpr VisitExpr_(const RampNode* op) override;
  // PrimExpr VisitExpr_(const BroadcastNode* op) override;
  // PrimExpr VisitExpr_(const ShuffleNode* op) override;
  // PrimExpr VisitExpr_(const IntImmNode* op) override;
  // PrimExpr VisitExpr_(const FloatImmNode* op) override;
  // PrimExpr VisitExpr_(const StringImmNode* op) override;
};


class FormCompute : public ExprVisitor {
 private:
  NameGenerator &generator_;
  const std::string &tensor_name_key_;
  Array<PrimExpr> &shape_;
  Array<Var> &sub_vars_;
 public:
  std::vector<Tensor> tensor_list;
  FormCompute(NameGenerator &generator, const std::string &tensor_name,
    Array<PrimExpr> &shape, Array<Var> &sub_vars) :
    generator_(generator), tensor_name_key_(tensor_name), shape_(shape), sub_vars_(sub_vars) {}

  void form_compute(const PrimExpr &expr) {
    VisitExpr(expr);
  }

 protected:
  void VisitExpr_(const VarNode* op) override {
    auto func =
      [=](const Array<Var> &input_indices) {
        Map<Var, PrimExpr> vmap;
        CHECK(input_indices.size() == sub_vars_.size());
        for (size_t i = 0; i < input_indices.size(); ++i) {
          vmap.Set(sub_vars_[i], input_indices[i]);
        }
        return Substitute(Var(op->name_hint, op->dtype), vmap);
      };
    tensor_list.push_back(te::compute(shape_, func, generator_.unique_name(tensor_name_key_)));
  }

  void VisitExpr_(const SizeVarNode* op) override UNEXPECTED
  void VisitExpr_(const LoadNode* op) override UNEXPECTED
  void VisitExpr_(const BufferLoadNode* op) override UNEXPECTED
  void VisitExpr_(const LetNode* op) override UNEXPECTED

  void VisitExpr_(const CallNode* op) override {
    auto func =
      [=](const Array<Var> &input_indices) {
        Map<Var, PrimExpr> vmap;
        CHECK(input_indices.size() == sub_vars_.size());
        for (size_t i = 0; i < input_indices.size(); ++i) {
          vmap.Set(sub_vars_[i], input_indices[i]);
        }
        return Substitute(CallNode::make(
                op->dtype,
                op->name,
                op->args,
                op->call_type,
                op->func,
                op->value_index), vmap);
      };
    tensor_list.push_back(te::compute(shape_, func, generator_.unique_name(tensor_name_key_)));
  }

  template<typename T>
  void visit_binay_op(const T*op) {
    VisitExpr(op->a);
    VisitExpr(op->b);
    CHECK((int)tensor_list.size() > 1);
    Tensor ta = tensor_list[tensor_list.size() - 2];
    Tensor tb = tensor_list[tensor_list.size() - 1];
    const ComputeOpNode *ca = ta->op.as<ComputeOpNode>();
    const ComputeOpNode *cb = tb->op.as<ComputeOpNode>();
    CHECK(ca != nullptr && cb != nullptr);
    const ReduceNode *a_red = op->a.template as<ReduceNode>();
    const ReduceNode *b_red = op->b.template as<ReduceNode>();
    if (a_red != nullptr && b_red != nullptr) {
      auto func =
        [=](const Array<Var> &input_indices){
          Array<PrimExpr> call_args;
          for (auto v : input_indices) {
            call_args.push_back(v);
          }
          PrimExpr a_body = CallNode::make(
            ta->dtype,
            ta->op->name,
            call_args,
            CallNode::CallType::Halide,
            ta->op,
            ta->value_index
          );
          PrimExpr b_body = CallNode::make(
            tb->dtype,
            tb->op->name,
            call_args,
            CallNode::CallType::Halide,
            tb->op,
            tb->value_index
          );
          Map<Var, PrimExpr> vmap;
          CHECK(sub_vars_.size() == input_indices.size());
          size_t i = 0;
          for (auto v : sub_vars_) {
            vmap.Set(v, input_indices[i]);
            ++i;
          }
          return Substitute(T::make(a_body, b_body), vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.push_back(new_tensor);
    } else if (a_red == nullptr && b_red != nullptr) {
      auto func =
        [=](const Array<Var> &input_indices){
          Array<PrimExpr> call_args;
          for (auto v : input_indices) {
            call_args.push_back(v);
          }
          CHECK((int)ca->body.size() == 1);
          Map<Var, PrimExpr> a_vmap;
          size_t i = 0;
          for (auto iv : ca->axis) {
            a_vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          PrimExpr a_body = Substitute(ca->body[0], a_vmap);
          PrimExpr b_body = CallNode::make(
            tb->dtype,
            tb->op->name,
            call_args,
            CallNode::CallType::Halide,
            tb->op,
            tb->value_index
          );
          Map<Var, PrimExpr> vmap;
          CHECK(sub_vars_.size() == input_indices.size());
          i = 0;
          for (auto v : sub_vars_) {
            vmap.Set(v, input_indices[i]);
            ++i;
          }
          return Substitute(T::make(a_body, b_body), vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.pop_back();
      tensor_list.pop_back();
      tensor_list.push_back(tb);
      tensor_list.push_back(new_tensor);
    } else if (a_red != nullptr && b_red == nullptr) {
      auto func =
        [=](const Array<Var> &input_indices){
          Array<PrimExpr> call_args;
          for (auto v : input_indices) {
            call_args.push_back(v);
          }
          CHECK((int)cb->body.size() == 1);
          Map<Var, PrimExpr> b_vmap;
          size_t i = 0;
          for (auto iv : cb->axis) {
            b_vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          PrimExpr b_body = Substitute(cb->body[0], b_vmap);
          PrimExpr a_body = CallNode::make(
            ta->dtype,
            ta->op->name,
            call_args,
            CallNode::CallType::Halide,
            ta->op,
            ta->value_index
          );
          Map<Var, PrimExpr> vmap;
          CHECK(sub_vars_.size() == input_indices.size());
          i = 0;
          for (auto v : sub_vars_) {
            vmap.Set(v, input_indices[i]);
            ++i;
          }
          return Substitute(T::make(a_body, b_body), vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.pop_back();
      tensor_list.pop_back();
      tensor_list.push_back(ta);
      tensor_list.push_back(new_tensor);
    } else {
      auto func =
        [=](const Array<Var> &input_indices){
          CHECK((int)ca->body.size() == 1);
          Map<Var, PrimExpr> a_vmap;
          size_t i = 0;
          for (auto iv : ca->axis) {
            a_vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          PrimExpr a_body = Substitute(ca->body[0], a_vmap);
          CHECK((int)cb->body.size() == 1);
          Map<Var, PrimExpr> b_vmap;
          i = 0;
          for (auto iv : cb->axis) {
            b_vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          PrimExpr b_body = Substitute(cb->body[0], b_vmap);
          Map<Var, PrimExpr> vmap;
          CHECK(sub_vars_.size() == input_indices.size());
          i = 0;
          for (auto v : sub_vars_) {
            vmap.Set(v, input_indices[i]);
            ++i;
          }
          return Substitute(T::make(a_body, b_body), vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.pop_back();
      tensor_list.pop_back();
      tensor_list.push_back(new_tensor);
    }
  }

  void VisitExpr_(const AddNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const SubNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const MulNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const DivNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const ModNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const FloorDivNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const FloorModNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const MinNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const MaxNode* op) override {
    visit_binay_op(op);
  }

  void VisitExpr_(const EQNode* op) override UNEXPECTED
  void VisitExpr_(const NENode* op) override UNEXPECTED
  void VisitExpr_(const LTNode* op) override UNEXPECTED
  void VisitExpr_(const LENode* op) override UNEXPECTED
  void VisitExpr_(const GTNode* op) override UNEXPECTED
  void VisitExpr_(const GENode* op) override UNEXPECTED
  void VisitExpr_(const AndNode* op) override UNEXPECTED
  void VisitExpr_(const OrNode* op) override UNEXPECTED

  void VisitExpr_(const ReduceNode* op) override {
    auto func =
      [=](const Array<Var> &input_indices) {
        Map<Var, PrimExpr> vmap;
        CHECK(input_indices.size() == sub_vars_.size());
        for (size_t i = 0; i < input_indices.size(); ++i) {
          vmap.Set(sub_vars_[i], input_indices[i]);
        }
        return Substitute(ReduceNode::make(
                op->combiner,
                op->source,
                op->axis,
                op->condition,
                op->value_index), vmap);
      };
    tensor_list.push_back(te::compute(shape_, func, generator_.unique_name(tensor_name_key_)));
  }

  void VisitExpr_(const CastNode* op) override {
    VisitExpr(op->value);
    const ReduceNode *as_red = op->value.as<ReduceNode>();
    Tensor t = tensor_list.back();
    const ComputeOpNode *cop = t->op.as<ComputeOpNode>();
    CHECK(cop != nullptr);
    if (as_red != nullptr) {
      auto func =
        [=](const Array<Var> &input_indices){
          Array<PrimExpr> call_args;
          for (auto v : input_indices) {
            call_args.push_back(v);
          }
          PrimExpr body = CallNode::make(
            t->dtype, t->op->name, call_args, CallNode::CallType::Halide, t->op, t->value_index);
          body = CastNode::make(op->dtype, body);
          return body;
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.push_back(new_tensor);
    } else {
      auto func =
        [=](const Array<Var> &input_indices){
          CHECK(cop->body.size() == 1) << "Only support body size 1.\n";
          PrimExpr body = cop->body[0];
          body = CastNode::make(op->dtype, body);
          CHECK(cop->axis.size() == input_indices.size()) << "Internal error: input indices mismatch.\n";
          Map<Var, PrimExpr> vmap;
          size_t i = 0;
          for (auto iv : cop->axis) {
            vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          return Substitute(body, vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.pop_back();
      tensor_list.push_back(new_tensor);
    }
  }

  void VisitExpr_(const NotNode* op) override UNEXPECTED

  void VisitExpr_(const SelectNode* op) override {
    // TODO: we do not support grad in condition
    VisitExpr(op->true_value);
    VisitExpr(op->false_value);
    CHECK((int)tensor_list.size() > 1);
    Tensor true_tensor = tensor_list[tensor_list.size() - 2];
    Tensor false_tensor = tensor_list[tensor_list.size() - 1];
    const ReduceNode *t_as_red = op->true_value.as<ReduceNode>();
    const ReduceNode *f_as_red = op->false_value.as<ReduceNode>();
    const ComputeOpNode *top = true_tensor->op.as<ComputeOpNode>();
    const ComputeOpNode *fop = false_tensor->op.as<ComputeOpNode>();
    CHECK(top != nullptr && fop != nullptr);
    if (t_as_red != nullptr && f_as_red != nullptr) {
      auto func =
        [=](const Array<Var> &input_indices){
          Array<PrimExpr> call_args;
          for (auto v : input_indices) {
            call_args.push_back(v);
          }
          PrimExpr true_body = CallNode::make(
            true_tensor->dtype,
            true_tensor->op->name,
            call_args,
            CallNode::CallType::Halide,
            true_tensor->op,
            true_tensor->value_index
          );
          PrimExpr false_body = CallNode::make(
            false_tensor->dtype,
            false_tensor->op->name,
            call_args,
            CallNode::CallType::Halide,
            false_tensor->op,
            false_tensor->value_index
          );
          Map<Var, PrimExpr> vmap;
          CHECK(sub_vars_.size() == input_indices.size());
          size_t i = 0;
          for (auto v : sub_vars_) {
            vmap.Set(v, input_indices[i]);
            ++i;
          }
          return Substitute(SelectNode::make(op->condition, true_body, false_body), vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.push_back(new_tensor);
    } else if (t_as_red == nullptr && f_as_red != nullptr) {
      auto func =
        [=](const Array<Var> &input_indices){
          Array<PrimExpr> call_args;
          for (auto v : input_indices) {
            call_args.push_back(v);
          }
          CHECK((int)top->body.size() == 1);
          Map<Var, PrimExpr> true_vmap;
          size_t i = 0;
          for (auto iv : top->axis) {
            true_vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          PrimExpr true_body = Substitute(top->body[0], true_vmap);
          PrimExpr false_body = CallNode::make(
            false_tensor->dtype,
            false_tensor->op->name,
            call_args,
            CallNode::CallType::Halide,
            false_tensor->op,
            false_tensor->value_index
          );
          Map<Var, PrimExpr> vmap;
          CHECK(sub_vars_.size() == input_indices.size());
          i = 0;
          for (auto v : sub_vars_) {
            vmap.Set(v, input_indices[i]);
            ++i;
          }
          return Substitute(SelectNode::make(op->condition, true_body, false_body), vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.pop_back();
      tensor_list.pop_back();
      tensor_list.push_back(false_tensor);
      tensor_list.push_back(new_tensor);
    } else if (t_as_red != nullptr && f_as_red == nullptr) {
      auto func =
        [=](const Array<Var> &input_indices){
          Array<PrimExpr> call_args;
          for (auto v : input_indices) {
            call_args.push_back(v);
          }
          CHECK((int)fop->body.size() == 1);
          Map<Var, PrimExpr> false_vmap;
          size_t i = 0;
          for (auto iv : fop->axis) {
            false_vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          PrimExpr false_body = Substitute(fop->body[0], false_vmap);
          PrimExpr true_body = CallNode::make(
            true_tensor->dtype,
            true_tensor->op->name,
            call_args,
            CallNode::CallType::Halide,
            true_tensor->op,
            true_tensor->value_index
          );
          Map<Var, PrimExpr> vmap;
          CHECK(sub_vars_.size() == input_indices.size());
          i = 0;
          for (auto v : sub_vars_) {
            vmap.Set(v, input_indices[i]);
            ++i;
          }
          return Substitute(SelectNode::make(op->condition, true_body, false_body), vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.pop_back();
      tensor_list.pop_back();
      tensor_list.push_back(true_tensor);
      tensor_list.push_back(new_tensor);
    } else {
      auto func =
        [=](const Array<Var> &input_indices){
          CHECK((int)top->body.size() == 1);
          Map<Var, PrimExpr> true_vmap;
          size_t i = 0;
          for (auto iv : top->axis) {
            true_vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          PrimExpr true_body = Substitute(top->body[0], true_vmap);
          CHECK((int)fop->body.size() == 1);
          Map<Var, PrimExpr> false_vmap;
          i = 0;
          for (auto iv : fop->axis) {
            false_vmap.Set(iv->var, input_indices[i]);
            ++i;
          }
          PrimExpr false_body = Substitute(fop->body[0], false_vmap);
          Map<Var, PrimExpr> vmap;
          CHECK(sub_vars_.size() == input_indices.size());
          i = 0;
          for (auto v : sub_vars_) {
            vmap.Set(v, input_indices[i]);
            ++i;
          }
          return Substitute(SelectNode::make(op->condition, true_body, false_body), vmap);
        };
      Tensor new_tensor = te::compute(shape_, func, generator_.unique_name(tensor_name_key_));
      tensor_list.pop_back();
      tensor_list.pop_back();
      tensor_list.push_back(new_tensor);
    }
  }

  void VisitExpr_(const RampNode* op) override UNEXPECTED
  void VisitExpr_(const BroadcastNode* op) override UNEXPECTED
  void VisitExpr_(const ShuffleNode* op) override UNEXPECTED

  void VisitExpr_(const IntImmNode* op) override {
    auto func =
      [=](const Array<Var> &input_indices) {
        return make_const(op->dtype, op->value);
      };
    tensor_list.push_back(te::compute(shape_, func, generator_.unique_name(tensor_name_key_)));
  }

  void VisitExpr_(const FloatImmNode* op) override {
    auto func =
      [=](const Array<Var> &input_indices) {
        PrimExpr ret = make_const(op->dtype, op->value);
        return ret;
      };
    tensor_list.push_back(te::compute(shape_, func, generator_.unique_name(tensor_name_key_)));
  }

  void VisitExpr_(const StringImmNode* op) override UNEXPECTED
};


// PrimExpr change_iter_var_names(const PrimExpr &body, const Array<IterVar> &ivs, const Array<IterVar> &nivs) {
//   Map<Var, PrimExpr> vmap;
//   for (size_t i = 0; i < ivs.size(); ++i) {
//     vmap.Set(ivs[i]->var, nivs[i]->var);
//   }
//   return Substitute(body, vmap);
// }


PrimExpr ensure_unique_var(const ComputeOpNode *op, SubstituteContext &context,
    NameGenerator &generator, Array<PrimExpr> &call_args, int idx) {
  Map<Var, PrimExpr> vmap;

  PrimExpr body = op->body[idx];
  for (auto iter_var : op->axis) {
    std::string name_hint = iter_var->var->name_hint;
    if (generator.has_name(name_hint)) {
      LOG(FATAL) << "Find repeat axis iter_var name: " << name_hint;
      throw;
    }
    std::string new_name = generator.unique_name(name_hint);
    context.index_names.push_back(new_name);
    Var new_var = Var(new_name);
    context.var_map[new_name] = new_var;
    context.range_map[new_name] = ExtRange(
        iter_var->dom->min, AddNode::make(iter_var->dom->min, iter_var->dom->extent), false, false);
    vmap.Set(iter_var->var, new_var);
    call_args.push_back(new_var);
  }

  // reduce node is always top-level
  if (const ReduceNode *red = body.as<ReduceNode>()) {
    // Array<IterVar> new_iter_vars;
    // name conflicts come from bad front-end definition
    // change repeated names to unique names
    for (auto iv : red->axis) {
      std::string name_hint = iv->var->name_hint;
      std::string new_name = generator.unique_name(name_hint);
      if (name_hint != new_name) {
        LOG(WARNING) << "Find repeat axis iter_var name: " << name_hint << "\n"
                     << "change to new name: " << new_name;
      }
      // new_iter_vars.push_back(IterVarNode::make(iv->dom,
      //   Var(new_name, iv->var->dtype), iv->iter_type, iv->thread_tag));
      context.index_names.push_back(new_name);
      Var new_var = Var(new_name);
      context.var_map[new_name] = new_var;
      context.range_map[new_name] = ExtRange(
          iv->dom->min, AddNode::make(iv->dom->min, iv->dom->extent), false, false);
      vmap.Set(iv->var, new_var);
    }
    // modify the sources
    // Array<PrimExpr> new_source;
    // for (auto src : red->source) {
    //   new_source.push_back(change_iter_var_names(src, red->axis, new_iter_vars));
    // }

    // PrimExpr new_condition = change_iter_var_names(red->condition, red->axis, new_iter_vars);

    // body = ReduceNode::make(red->combiner, new_source, new_iter_vars, new_condition, red->value_index);
  }

  return Simplify(Substitute(body, vmap));
}


Tensor grad_op(const Tensor& input, const Tensor& output, const Tensor& doutput) {
  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  CHECK(op) << "Derivative of this operation is not implemented: " << output->op;
  bool is_input_tensor = false;
  for (const Tensor& child : op->InputTensors()) {
    if (input == child) {
      is_input_tensor = true;
      break;
    }
  }
  CHECK(is_input_tensor) << "grad_op is called on a pair of tensors such that the output "
                         << "does not directly depend on the input.";

  SubstituteContext context;
  NameGenerator generator;
  Array<IterVar> compute_indices;
  Array<PrimExpr> call_args;
  PrimExpr body = ensure_unique_var(op, context, generator, call_args, output->value_index);
  PrimExpr new_body;

  for (size_t i = 0; i < input->shape.size(); ++i) {
    std::string new_name = generator.unique_name("z");
    compute_indices.push_back(
      IterVarNode::make(
        Range(0, input->shape[i]),
        Var(new_name),
        kDataPar)
    );
    context.range_map[new_name] = ExtRange(0, input->shape[i], false, false);
  }

  std::cout << "check initial context:\n";
  std::cout << context << "\n";

  Array<PrimExpr> compute_args;
  for (auto val : compute_indices) {
    compute_args.push_back(val->var);
  }

  const ComputeOpNode* oop = output->op.as<ComputeOpNode>();
  CHECK(oop != nullptr) << "Only support ComputeOpNode.\n";

  // Array<PrimExpr> call_args;
  // for (auto val : oop->axis) {
  //   call_args.push_back(val->var);
  // }

  PrimExpr grad_body;

  const ReduceNode *as_red = body.as<ReduceNode>();
  if (as_red != nullptr) {
    CHECK((int)as_red->source.size() == 1) << "Only support one source now.\n";
    Array<PrimExpr> new_source;
    for (auto expr : as_red->source) {
      SubstituteContext new_context = context.copy();
      GradOp helper(generator, new_context, input, doutput, call_args, compute_args);
      PrimExpr new_expr = helper.grad(expr);
      std::cout << "check source: " << Simplify(new_expr) << "\n";
      new_source.push_back(new_expr);
    }
    // only take away the first one
    grad_body = Simplify(new_source[0]);
  } else {
    SubstituteContext new_context = context.copy();
    GradOp helper(generator, new_context, input, doutput, call_args, compute_args);
    PrimExpr new_expr = helper.grad(body);
    std::cout << "check body: " << Simplify(new_expr)  << "\n";
    grad_body = Simplify(new_expr);
  }

  std::cout << "\nLift ReduceNode:\n";
  LiftReduce lifter;
  grad_body = Simplify(lifter.lift(grad_body));
  std::cout << "Result:\n" << grad_body << "\n";

  // std::vector<Tensor> tensor_list;
  // FormCompute(NameGenerator &generator, const std::string &tensor_name,
  //   Array<PrimExpr> &shape, Array<Var> &sub_vars)
  Array<PrimExpr> shape;
  for (auto it : input->shape) {
    shape.push_back(it);
  }
  Array<Var> sub_vars;
  for (auto iv : compute_indices) {
    sub_vars.push_back(iv->var);
  }
  FormCompute former(generator, generator.unique_name("_tensor"), shape, sub_vars);
  former.form_compute(grad_body);
  
  std::cout << "check form compute:\n";
  for (auto it : former.tensor_list) {
    std::cout << "tensor: " << it << "\n";
    const ComputeOpNode *cop = it->op.as<ComputeOpNode>();
    CHECK(cop != nullptr);
    std::cout << "axis: " << cop->axis << "\n";
    std::cout << "body: " << cop->body << "\n";
  }
  
  return former.tensor_list.back();
}


TVM_REGISTER_GLOBAL("te.grad_op")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  LOG(WARNING) << "te.grad_op is an experimental feature.";
  *ret = grad_op(args[0], args[1], args[2]);
});


}  // namespace te
}  // namespace tvm
