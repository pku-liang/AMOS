#ifndef TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_
#define TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_

#include <unordered_map>
#include <queue>

#include <tvm/te/schedule.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/target/target.h>

// #include "utils.h"
// #include "interpreter.h"
// #include "structure_space.h"
// #include "search_tree.h"
#include "../utils.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"


namespace tvm {

namespace tg {


class ScheduleResultNode : public Object {
 public:
  te::Schedule schedule;
  Array<te::Tensor> tensors;
  // std::shared_ptr<SearchTreeNode> leaf;
  // Array<Config> configs;

  static constexpr const char* _type_key = "tg.ScheduleResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleResultNode, Object);
};


class ScheduleResult : public ObjectRef {
 public:
  ScheduleResult(te::Schedule sch, Array<te::Tensor> tensors/*,
                 std::shared_ptr<SearchTreeNode> leaf, Array<Config> configs*/) {
    auto node = make_object<ScheduleResultNode>();
    node->schedule = sch;
    node->tensors = tensors;
    // node->leaf = leaf;
    // node->configs = configs;
    data_ = std::move(node);
  }

  // std::shared_ptr<SearchTreeNode> get_leaf() {
  //   return Self()->leaf;
  // }

  TVM_DEFINE_OBJECT_REF_METHODS(ScheduleResult, ObjectRef, ScheduleResultNode);
  TG_DEFINE_OBJECT_SELF_METHOD(ScheduleResultNode);
};


class EvaluatedScheduleResultNode : public Object {
 public:
  ScheduleResult schedule_result;
  float evaluation;

  static constexpr const char* _type_key = "tg.EvaluatedScheduleResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleResultNode, Object);
};


class EvaluatedScheduleResult : public ObjectRef {
 public:
  EvaluatedScheduleResult(ScheduleResult result, float evaluation) {
    auto node = make_object<EvaluatedScheduleResultNode>();
    node->schedule_result = result;
    node->evaluation = evaluation;
    data_ = std::move(node);
  }

  bool operator< (const EvaluatedScheduleResult &other) const {
    return (*this)->evaluation > other->evaluation;
  }

  TVM_DEFINE_OBJECT_REF_METHODS(EvaluatedScheduleResult, ObjectRef, EvaluatedScheduleResultNode);
};


class AutoScheduleContextNode : public Object {
 public:
  Target target;
  IntKey task_id;
  // SearchTree search_tree;

  static constexpr const char* _type_key = "tg.AutoScheduleContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(AutoScheduleContextNode, Object);
};


class AutoScheduleContext : public ObjectRef {
 public:
  AutoScheduleContext(Target &target, IntKey &task_id) {
    auto node = make_object<AutoScheduleContextNode>();
    node->target = target;
    node->task_id = task_id;
    // node->search_tree = SearchTree();
    data_ = std::move(node);
  }

  // SearchTree& get_search_tree() {
  //   auto self = Self();
  //   return self->search_tree;
  // }

  TVM_DEFINE_OBJECT_REF_METHODS(AutoScheduleContext, ObjectRef, AutoScheduleContextNode);
  TG_DEFINE_OBJECT_SELF_METHOD(AutoScheduleContextNode);
};


// auto_schedule for one subgraph
bool auto_schedule(
    TIRGraph subgraph,
    AutoScheduleContext &context,
    std::vector<ScheduleResult> &results);


class AutoScheduler {
 private:
  const static int num_topk = 10;
  const static int schedule_trials_for_one = 100;
  ThreadPool *thread_pool = nullptr;

  std::unordered_map<IntKey, AutoScheduleContext> contexts;
  std::unordered_map<IntKey, std::priority_queue<EvaluatedScheduleResult> > topk_schedules;

  ScheduleResult schedule_func(IntKey key, TIRGraph subgraph, Target target);
  tvm::runtime::Module schedule_and_build_func(IntKey key, TIRGraph subgraph, Target target);
 public:
  AutoScheduler() { thread_pool = new ThreadPool(1); }
  ~AutoScheduler() { if (thread_pool != nullptr) delete thread_pool; }
  void reset() { if (thread_pool != nullptr) {delete thread_pool; thread_pool = new ThreadPool(1);} }
  std::shared_future<ScheduleResult> schedule_for(IntKey key, TIRGraph subgraph, Target target, int priority=0);
  std::shared_future<tvm::runtime::Module> schedule_and_build_for(
    IntKey key, TIRGraph subgraph, Target target, int priority=0);
  
  // void feedback_schedule(IntKey key, ScheduleResult schedule_result, float feedback);
  // static AutoScheduler& Global();
};



}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_