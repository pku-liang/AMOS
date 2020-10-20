#include <unordered_set>
#include <tvm/tir/op.h>

#include "utils.h"


namespace tvm {

namespace tg {

void FindBatchLikeDim::VisitExpr_(const ProducerLoadNode* op) {
  ExprVisitor::VisitExpr_(op);
  int pos = 0;
  for (auto e : op->indices) {
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


void FindFusibleDim::VisitExpr_(const ProducerLoadNode *op) {
    ExprVisitor::VisitExpr_(op);
    bool is_weight = false;
    for (auto &w: weights_) {
        if (Downcast<te::Tensor>(op->producer) == w) {
            is_weight = true;
            break;
        }
    }
    int pos = 0;
    for (auto e : op->indices) {
        const VarNode* e_as_var = e.as<VarNode>();
        if (e_as_var != nullptr) {
            int spatial_id = 0;
            for (auto v : this->spatial_indices_) {
                if (e_as_var == v.get()) {
                    if (is_weight) {
                        spatial_indices_in_weight[spatial_id] = std::make_pair(true, pos);
                    }
                    else {
                        spatial_indices_in_input[spatial_id] = true;
                    }
                }
                spatial_id += 1;
            }
        }
        pos += 1;
    }
}


void FindAxisPosition::VisitExpr_(const ProducerLoadNode* op) {
  ExprVisitor::VisitExpr_(op);
  if (tensor_ == Downcast<te::Tensor>(op->producer)) {
    int pos = 0;
    for (auto e : op->indices) {
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
    static std::unordered_set<RelayExpr, ObjectPtrHash, ObjectPtrEqual> piecewise_const = {
        Op::Get("tir.floor"), Op::Get("tir.ceil"), Op::Get("tir.trunc"), Op::Get("tir.round")};
    if (op->op.same_as(Op::Get("tir.exp"))) {
      this->num_special += 1;
    } else if (op->op.same_as(Op::Get("tir.log"))) {
      this->num_special += 1;
    } else if (op->op.same_as(Op::Get("tir.sigmoid"))) {
      this->num_special += 1;
    } else if (op->op.same_as(Op::Get("tir.sqrt"))) {
      this->num_special += 1;
    } else if (op->op.same_as(Op::Get("tir.tanh"))) {
      this->num_special += 1;
    } else if (op->op.same_as(Op::Get("tir.pow"))) {
      this->num_special += 1;
    } else if (op->op.same_as(Op::Get("tir.fabs"))) {
      this->num_special += 1;
    } else if (op->op.same_as(Op::Get("tir.if_then_else"))) {
      this->num_branch += 1;
    } else if (piecewise_const.count(op->op)) {
      this->num_special += 1;
    } else {
      LOG(FATAL) << "Count opteration number of this intrinsic is not implemented: " << op->op;
    }
}


void CountOperationNum::VisitExpr_(const AddNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->dtype.is_float())
    this->num_add += 1;
}


void CountOperationNum::VisitExpr_(const SubNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->dtype.is_float())
    this->num_add += 1;
}


void CountOperationNum::VisitExpr_(const MulNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->dtype.is_float())
    this->num_mul += 1;
}


void CountOperationNum::VisitExpr_(const DivNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->dtype.is_float())
    this->num_div += 1;
}


void CountOperationNum::VisitExpr_(const ModNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->dtype.is_float())
    this->num_special += 1;
}


void CountOperationNum::VisitExpr_(const FloorDivNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->dtype.is_float())
    this->num_div += 1;
}


void CountOperationNum::VisitExpr_(const FloorModNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->dtype.is_float())
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
  // this->num_special += 1;
}


void CountOperationNum::VisitExpr_(const ReduceNode *op) {
  ExprVisitor::VisitExpr_(op);
  if (op->condition.defined()) {
    this->num_branch += 1;
  }
}


void CountInputOccur::VisitExpr_(const ProducerLoadNode *op) {
  ExprVisitor::VisitExpr_(op);
  int i = 0;
  for (auto t : inputs_) {
    if (t == Downcast<te::Tensor>(op->producer)) {
      count_occur[i] += 1;
    }
    i += 1;
  }
}


Array<IntImm> get_batch_like_dim(const Tensor &output) {
  Array<IntImm> ret;
  std::vector<bool> is_batch;

  const ComputeOpNode* op = output->op.as<ComputeOpNode>();  // 不是 PlaceHolder 或 Tensor
  CHECK(op != nullptr);

  Array<Var> spatial_indices;
  for (auto iv : op->axis) {  // op->axis 返回所有 spatial axis
    is_batch.push_back(true);  // 一开始将所有 axis 都标记成
    spatial_indices.push_back(iv->var);
  }
  
  for (auto b : op->body) { // op->body 即用户提供的 lambda 表达式
    FindBatchLikeDim fbld(spatial_indices);
    fbld.VisitExpr(b);
    for (auto kv : fbld.records) {
      if ((int)kv.second.size() == 0) {  // 该 spatial axis 没有单独出现在任何访存中
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


Array<Array<IntImm> > find_fusible_dim(const Tensor &output, Array<Tensor> weights) {

    Array<Array<IntImm> > ret;
    std::vector<std::pair<bool, int> > is_fusible;

    const ComputeOpNode* op = output->op.as<ComputeOpNode>();
    CHECK(op != nullptr);

    Array<Var> spatial_indices;
    for (auto iv : op->axis) {
        is_fusible.push_back(std::make_pair(false, -1));
        spatial_indices.push_back(iv->var);
    }

    CHECK_EQ(op->body.size(), 1);

    for (auto b : op->body) {
        FindFusibleDim ffd(spatial_indices, weights);
        ffd.VisitExpr(b);

        int n = (int)spatial_indices.size();
        for (int i = 0; i < n; ++i) {
            if (ffd.spatial_indices_in_weight[i].first && !ffd.spatial_indices_in_input[i]) {
                is_fusible[i] = std::make_pair(true, ffd.spatial_indices_in_weight[i].second);
            }
        }
    }

    for (int i = 0; i < (int)is_fusible.size(); ++i) {
        if (is_fusible[i].first) {
            ret.push_back(Array<IntImm> {
                IntImm(DataType::Int(32), i),
                IntImm(DataType::Int(32), is_fusible[i].second),
            });
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

  ret.Set(StringImm("num_add"), IntImm(DataType::Int(32), con.num_add));
  ret.Set(StringImm("num_div"), IntImm(DataType::Int(32), con.num_div));
  ret.Set(StringImm("num_mul"), IntImm(DataType::Int(32), con.num_mul));
  ret.Set(StringImm("num_branch"), IntImm(DataType::Int(32), con.num_branch));
  ret.Set(StringImm("num_special"), IntImm(DataType::Int(32), con.num_special));
  ret.Set(StringImm("num_logic"), IntImm(DataType::Int(32), con.num_logic));

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


std::pair<Array<Operation>, Map<Operation, Array<Operation> > >
  serialize_compute_dag(Array<Operation> root_ops, bool output_first) {
  
  std::set<Operation> visited;
  std::vector<Operation> ret;
  std::map<Operation, std::unordered_set<Operation> > down_graph_set;
  std::unordered_map<Operation, Array<Operation> > down_graph;
  // std::deque<Operation> q;

  std::function<void(Operation)> helper;
  helper = [&] (Operation op) {
    if (visited.find(op) != visited.end()) {
      return;
    }
    const ComputeOpNode* as_compute = op.as<ComputeOpNode>();
    if (as_compute == nullptr) {
      return;
    }
    for (auto t : op->InputTensors()) {
      helper(t->op);
      // if (down_graph.find(t->op) == down_graph.end()) {
      //     Array<Operation> tmp;
      //     down_graph[t->op] = tmp;
      // }
      if (down_graph_set.find(t->op) == down_graph_set.end()) {
        down_graph_set[t->op] = std::unordered_set<Operation>();
      }
      if (down_graph_set[t->op].find(op) == down_graph_set[t->op].end()) {
        down_graph_set[t->op].insert(op);
        down_graph[t->op].push_back(op);
      }
      
    }
    ret.push_back(op);
    visited.insert(op);
  };

  for (auto op : root_ops) {
    helper(op);
  }

  // for (auto op : root_ops) {
  //   q.push_back(op);
  //   visited.insert(op);
  // }

  // while (!q.empty()) {
  //   Operation cur = q.front();
  //   q.pop_front();
  //   const ComputeOpNode *as_compute = cur.as<ComputeOpNode>();
  //   if (as_compute != nullptr) {
  //     ret.push_back(cur);
  //     for (auto t : as_compute->InputTensors()) {
  //       if (visited.find(t->op) == visited.end()) {
  //         visited.insert(t->op);
  //         q.push_back(t->op);
  //       }
  //       if (down_graph.find(t->op) == down_graph.end()) {
  //          Array<Operation> tmp;
  //          down_graph[t->op] = tmp;
  //       }
  //       down_graph[t->op].push_back(cur);
  //     }
  //   }
  // }

  if (output_first) {
    std::reverse(std::begin(ret), std::end(ret));
  }
  
  return std::make_pair(Array<Operation>(ret), Map<Operation, Array<Operation> >(down_graph));
}


TVM_REGISTER_GLOBAL("tg.find_fusible_dim")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = find_fusible_dim(args[0], args[1]);
});


TVM_REGISTER_GLOBAL("tg.get_batch_like_dim")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = get_batch_like_dim(args[0]);
});


TVM_REGISTER_GLOBAL("tg.find_axis_in")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = find_axis_in(args[0], args[1], args[2]);
});


TVM_REGISTER_GLOBAL("tg.count_operation")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = count_operation(args[0]);
});


TVM_REGISTER_GLOBAL("tg.count_input_occur")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  *ret = count_input_occur(args[0], args[1]);
});


TVM_REGISTER_GLOBAL("tg.flatten_tir_graph")
.set_body_typed([](Array<Operation> root_ops, bool output_first){
  Array<Operation> op_list;
  Map<Operation, Array<Operation> > down_graph;
  std::tie(op_list, down_graph) = serialize_compute_dag(root_ops, output_first);
  return Array<ObjectRef>{op_list, down_graph};
});


}  // namespace tg

}  // namespace tvm