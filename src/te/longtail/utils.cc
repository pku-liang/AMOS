#include <unordered_set>
#include <tvm/tir/op.h>

#include "utils.h"


namespace tvm {

namespace te {


void FindBatchLikeDim::VisitExpr_(const CallNode* op) {
  ExprVisitor::VisitExpr_(op);
  if (op->call_type == CallNode::CallType::Halide) {
    int pos = 0;
    for (auto e : op->args) {
      const VarNode* e_as_var = e.as<VarNode>();
      if (e_as_var != nullptr) {
        int spatial_id = 0;
        for (auto v : this->spatial_indices_) {
          if (e_as_var == v.get()) {
            records[spatial_id].push_back(pos);
          }
          spatial_id += 1;
        }
      }

      pos += 1;
    }
  }
}


void FindAxisPosition::VisitExpr_(const CallNode* op) {
  ExprVisitor::VisitExpr_(op);
  if (op->call_type == CallNode::CallType::Halide &&
        tensor_->op.same_as(op->func)) {
    int pos = 0;
    for (auto e : op->args) {
      const VarNode* e_as_var = e.as<VarNode>();
      if (e_as_var != nullptr) {
        int spatial_id = 0;
        for (auto v : this->spatial_indices_) {
          if (e_as_var == v.get()) {
            records[spatial_id].push_back(pos);
          }
          spatial_id += 1;
        }
      }

      pos += 1;
    }
  }
}


void CountOperationNum::VisitExpr_(const CallNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->call_type == CallNode::CallType::PureIntrinsic) {
    static std::unordered_set<std::string> piecewise_const = {"floor", "ceil", "trunc", "round"};
    if (op->name == "exp") {
      this->num_special += 1;
    } else if (op->name == "log") {
      this->num_special += 1;
    } else if (op->name == "sigmoid") {
      this->num_special += 1;
    } else if (op->name == "sqrt") {
      this->num_special += 1;
    } else if (op->name == "tanh") {
      this->num_special += 1;
    } else if (op->name == "pow") {
      this->num_special += 1;
    } else if (op->name == "fabs") {
      this->num_special += 1;
    } else if (op->name == intrinsic::tvm_if_then_else) {
      this->num_branch += 1;
    } else if (piecewise_const.count(op->name)) {
      this->num_special += 1;
    } else {
      throw dmlc::Error("Derivative of this intrinsic is not implemented: " + op->name);
    }
  }
}


void CountOperationNum::VisitExpr_(const AddNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_add += 1;
}


void CountOperationNum::VisitExpr_(const SubNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_add += 1;
}


void CountOperationNum::VisitExpr_(const MulNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_mul += 1;
}


void CountOperationNum::VisitExpr_(const DivNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_div += 1;
}


void CountOperationNum::VisitExpr_(const ModNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_special += 1;
}


void CountOperationNum::VisitExpr_(const FloorDivNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_div += 1;
}


void CountOperationNum::VisitExpr_(const FloorModNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_special += 1;
}


void CountOperationNum::VisitExpr_(const MinNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_branch += 1;
}


void CountOperationNum::VisitExpr_(const MaxNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_branch += 1;
}


void CountOperationNum::VisitExpr_(const AndNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_logic += 1;
}


void CountOperationNum::VisitExpr_(const OrNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_logic += 1;
}


void CountOperationNum::VisitExpr_(const NotNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_logic += 1;
}


void CountOperationNum::VisitExpr_(const SelectNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_branch += 1;
}


void CountOperationNum::VisitExpr_(const CastNode *op) {
  ExprVisitor::VisitExpr_(op);
  this->num_special += 1;
}


void CountOperationNum::VisitExpr_(const ReduceNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->condition.defined()) {
    this->num_branch += 1;
  }
}


void CountInputOccur::VisitExpr_(const CallNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->call_type == CallNode::CallType::Halide) {
    int i = 0;
    for (auto t : inputs_) {
      if (t->op.same_as(op->func)) {
        count_occur[i] += 1;
      }
      i += 1;
    }
  }
}



Array<IntImm> get_batch_like_dim(const Tensor &output) {
  Array<IntImm> ret;
  std::vector<bool> is_batch;

  const ComputeOpNode* op = output->op.as<ComputeOpNode>();
  CHECK(op != nullptr);

  Array<Var> spatial_indices;
  for (auto iv : op->axis) {
    is_batch.push_back(true);
    spatial_indices.push_back(iv->var);
  }
  
  for (auto b : op->body) {
    FindBatchLikeDim fbld(spatial_indices);
    fbld.VisitExpr(b);
    for (auto kv : fbld.records) {
      if ((int)kv.second.size() == 0) {
        // this is not a batch like dim
        is_batch[kv.first] = false;
      }
    }
  }

  for (int i = 0; i < (int)is_batch.size(); ++i) {
    if (is_batch[i]) {
      ret.push_back(IntImm(DataType::Int(32), i));
    }
  }

  return ret;
}


Array<IntImm> find_axis_in(Array<IterVar> axis, const Tensor &tensor, const Tensor &output) {
  Array<Var> vars;
  for (auto iv : axis) {
    vars.push_back(iv->var);
  }

  const ComputeOpNode *as_compute = output->op.as<ComputeOpNode>();
  CHECK(as_compute != nullptr);

  FindAxisPosition fap(vars, tensor);
  for (auto body : as_compute->body) {
    fap.VisitExpr(body);
  }

  Array<IntImm> ret;
  std::unordered_set<int> has_value;
  for (auto kv : fap.records) {
    for (auto v : kv.second) {
      if (has_value.find(v) == has_value.end()) {
        ret.push_back(IntImm(DataType::Int(32), v));
        has_value.insert(v);
      }
    }
  }

  return ret;
}


Map<PrimExpr, IntImm> count_operation(const Operation& op) {
  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  CHECK(as_compute != nullptr);

  CountOperationNum con;
  Map<PrimExpr, IntImm> ret;

  for (auto body : as_compute->body) {
    con.VisitExpr(body);
  }

  ret.Set(StringImmNode::make("num_add"), IntImm(DataType::Int(32), con.num_add));
  ret.Set(StringImmNode::make("num_div"), IntImm(DataType::Int(32), con.num_div));
  ret.Set(StringImmNode::make("num_mul"), IntImm(DataType::Int(32), con.num_mul));
  ret.Set(StringImmNode::make("num_branch"), IntImm(DataType::Int(32), con.num_branch));
  ret.Set(StringImmNode::make("num_special"), IntImm(DataType::Int(32), con.num_special));
  ret.Set(StringImmNode::make("num_logic"), IntImm(DataType::Int(32), con.num_logic));

  return ret;
}


Array<IntImm> count_input_occur(Array<Tensor> inputs, const Operation& op) {
  const ComputeOpNode *as_compute = op.as<ComputeOpNode>();
  CHECK(as_compute != nullptr);

  CountInputOccur cio(inputs);
  for (auto b : as_compute->body) {
    cio.VisitExpr(b);
  }

  Array<IntImm> ret;
  for (auto v : cio.count_occur) {
    ret.push_back(IntImm(DataType::Int(32), v));
  }
  return ret;
}


TVM_REGISTER_GLOBAL("te.get_batch_like_dim")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = get_batch_like_dim(args[0]);
});


TVM_REGISTER_GLOBAL("te.find_axis_in")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = find_axis_in(args[0], args[1], args[2]);
});


TVM_REGISTER_GLOBAL("te.count_operation")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = count_operation(args[0]);
});


TVM_REGISTER_GLOBAL("te.count_input_occur")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = count_input_occur(args[0], args[1]);
});


}  // namespace tvm

}  // namespace te