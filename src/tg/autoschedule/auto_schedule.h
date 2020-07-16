#ifndef TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_
#define TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <fstream>

#include <tvm/te/schedule.h>
#include <tvm/node/container.h>
#include <tvm/tir/expr.h>
#include <tvm/target/target.h>

#include "schedule_space.h"
#include "feature.h"
#include "../utils.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"


namespace tvm {

namespace tg {


class ScheduleResultNode : public Object {
 public:
  te::Schedule schedule;
  Array<te::Tensor> tensors;
  MultiScheduleEntity schedule_entities;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("schedule", &schedule);
    v->Visit("tensors", &tensors);
    v->Visit("schedule_entities", &schedule_entities);
  }

  static constexpr const char* _type_key = "tg.autoschedule.ScheduleResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleResultNode, Object);
};


class ScheduleResult : public ObjectRef {
 public:
  ScheduleResult(te::Schedule sch, Array<te::Tensor> tensors,
                 MultiScheduleEntity entities) {
    auto node = make_object<ScheduleResultNode>();
    node->schedule = sch;
    node->tensors = tensors;
    node->schedule_entities = entities;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(ScheduleResult, ObjectRef, ScheduleResultNode);
};


class EvaluatedScheduleResultNode : public Object {
 public:
  ScheduleResult schedule_result;
  double evaluation;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("schedule_result", &schedule_result);
    v->Visit("evaluation", &evaluation);
  }

  static constexpr const char* _type_key = "tg.autoschedule.EvaluatedScheduleResult";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleResultNode, Object);
};


class EvaluatedScheduleResult : public ObjectRef {
 public:
  EvaluatedScheduleResult(ScheduleResult result, double evaluation) {
    auto node = make_object<EvaluatedScheduleResultNode>();
    node->schedule_result = result;
    node->evaluation = evaluation;
    data_ = std::move(node);
  }

  bool operator< (const EvaluatedScheduleResult &other) const {
    return (*this)->evaluation < other->evaluation;
  }

  bool operator> (const EvaluatedScheduleResult &other) const {
    return (*this)->evaluation > other->evaluation;
  }

  TVM_DEFINE_OBJECT_REF_METHODS(EvaluatedScheduleResult, ObjectRef, EvaluatedScheduleResultNode);
};


class AutoScheduleContextNode : public Object {
 public:
  IntKey task_id;
  TIRGraph graph;
  Target target;
  MultiScheduleSpace spaces;
  int topk;
  int number_per_trial;
  std::priority_queue<
        EvaluatedScheduleResult,
        std::vector<EvaluatedScheduleResult>,
        std::greater<EvaluatedScheduleResult> > topk_schedules;
  std::unordered_set<MultiScheduleEntity, ObjectHash> known_schedules;
  std::ofstream log_out;
  std::string policy;

  static constexpr const char* _type_key = "tg.autoschedule.AutoScheduleContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(AutoScheduleContextNode, Object);
};


class AutoScheduleContext : public ObjectRef {
 public:
  AutoScheduleContext(IntKey task_id, TIRGraph graph, Target target,
  int topk=20, int number_per_trial=20, std::string log_file_name="autoschedule_log.txt",
  std::string policy="profile") {
    auto node = make_object<AutoScheduleContextNode>();
    node->task_id = task_id;
    node->graph = graph;
    node->target = target;
    node->spaces = MultiScheduleSpace(graph, target);
    node->topk = topk;
    node->number_per_trial = number_per_trial;
    node->log_out.open(log_file_name, std::ios::app);
    node->policy = policy;
    data_ = std::move(node);
  }

  void add_feedback(ScheduleResult schedule_result, double evaluation);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AutoScheduleContext, ObjectRef, AutoScheduleContextNode);
};


// auto_schedule for one subgraph
void auto_schedule(
    TIRGraph subgraph,
    AutoScheduleContext &context,
    ScheduleResult &results);


class AutoScheduler {
 private:
  const static int schedule_trials_for_one = 10;
  ThreadPool *thread_pool = nullptr;

  std::unordered_map<IntKey, AutoScheduleContext> contexts;

  ScheduleResult schedule_func(IntKey key, TIRGraph subgraph, Target target);
 public:
  AutoScheduler() { thread_pool = new ThreadPool(1); }
  ~AutoScheduler() { if (thread_pool != nullptr) delete thread_pool; }
  void reset() { if (thread_pool != nullptr) {delete thread_pool; thread_pool = new ThreadPool(1);} }
  std::shared_future<ScheduleResult> schedule_for(IntKey key, TIRGraph subgraph, Target target, int priority=0);
  void feedback_for(IntKey key, TIRGraph subgraph, ScheduleResult schedule_result, double evaluation);
};



}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_