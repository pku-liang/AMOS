#ifndef TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_
#define TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_

#include <unordered_map>

#include <tvm/te/schedule.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/target/target.h>

#include "utils.h"
#include "interpreter.h"
#include "structure_space.h"
#include "search_tree.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"


namespace tvm {

namespace tg {


class ScheduleResultNode : public Object {
 public:
  te::Schedule schedule;
  Array<te::Tensor> tensors;
  Array<Config> configs;

  static constexpr const char* _type_key = "tg.ScheduleResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleResultNode, Object);
};


class ScheduleResult : public ObjectRef {
 public:
  ScheduleResult(te::Schedule sch, Array<te::Tensor> tensors, Array<Config> configs) {
    auto node = make_object<ScheduleResultNode>();
    node->schedule = sch;
    node->tensors = tensors;
    node->configs = configs;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(ScheduleResult, ObjectRef, ScheduleResultNode);
};


class AutoScheduleContextNode : public Object {
 public:
  Target target;
  IntKey task_id;
  SearchTree search_tree;

  static constexpr const char* _type_key = "tg.AutoScheduleContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(AutoScheduleContextNode, Object);
};


class AutoScheduleContext : public ObjectRef {
 public:
  AutoScheduleContext(Target &target, IntKey &task_id) {
    auto node = make_object<AutoScheduleContextNode>();
    node->target = target;
    node->task_id = task_id;
    node->search_tree = SearchTree();
    data_ = std::move(node);
  }

  SearchTree& get_search_tree() {
    auto self = Self();
    return self->search_tree;
  }

  TVM_DEFINE_OBJECT_REF_METHODS(AutoScheduleContext, ObjectRef, AutoScheduleContextNode);
  TG_DEFINE_OBJECT_SELF_METHOD(AutoScheduleContextNode);
};


// auto_schedule for one subgraph
bool auto_schedule(
    TIRGraph subgraph,
    AutoScheduleContext &context,
    std::vector<ScheduleResult> &results);

}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_