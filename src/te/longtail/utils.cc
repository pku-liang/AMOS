#include <unordered_set>
#include <tvm/tir/op.h>

#include "utils.h"


namespace tvm {

namespace te {


void FindBatchLikeDim::VisitExpr_(const CallNode* op) {
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


TVM_REGISTER_GLOBAL("te.get_batch_like_dim")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = get_batch_like_dim(args[0]);
});


TVM_REGISTER_GLOBAL("te.find_axis_in")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = find_axis_in(args[0], args[1], args[2]);
});


}  // namespace tvm

}  // namespace te