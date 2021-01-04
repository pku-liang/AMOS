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
#include "measure.h"
#include "../utils.h"
#include "../logging.h"
#include "../thread_pool.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"


namespace tvm {

namespace tg {

class ScheduleTensorsNode : public Object {
 public:
  te::Schedule schedule;
  Array<te::Tensor> tensors;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("schedule", &schedule);
    v->Visit("tensors", &tensors);
  }

  static constexpr const char* _type_key = "tg.ScheduleTensors";
  TVM_DECLARE_FINAL_OBJECT_INFO(ScheduleTensorsNode, Object);
};


class ScheduleTensors : public ObjectRef {
 public:
  TVM_DLL ScheduleTensors(te::Schedule sch, Array<te::Tensor> tensors) {
    auto node = make_object<ScheduleTensorsNode>();
    node->schedule = sch;
    node->tensors = tensors;
    data_ = std::move(node);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(ScheduleTensors, ObjectRef, ScheduleTensorsNode);
};


class ScheduleResultNode : public Object {
 public:
  te::Schedule schedule;
  Array<te::Tensor> tensors;
  MultiScheduleEntity schedule_entities;
  String external_schedule;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("schedule", &schedule);
    v->Visit("tensors", &tensors);
    v->Visit("schedule_entities", &schedule_entities);
    v->Visit("external_schedule", &external_schedule);
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
    node->external_schedule = "";
    data_ = std::move(node);
  }

  ScheduleResult(te::Schedule sch, Array<te::Tensor> tensors,
                 String external_schedule) {
    auto node = make_object<ScheduleResultNode>();
    node->schedule = sch;
    node->tensors = tensors;
    node->external_schedule = external_schedule;
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
  int new_trial;
  std::priority_queue<
        EvaluatedScheduleResult,
        std::vector<EvaluatedScheduleResult>,
        std::greater<EvaluatedScheduleResult> > topk_schedules;
  std::unordered_set<MultiScheduleEntity, ObjectHash> known_schedules;
  std::unordered_set<MultiScheduleEntity, ObjectHash> knowing_schedules;
  std::string policy;
  unsigned long long counts;

  static constexpr const char* _type_key = "tg.autoschedule.AutoScheduleContext";
  TVM_DECLARE_FINAL_OBJECT_INFO(AutoScheduleContextNode, Object);
};


class AutoScheduleContext : public ObjectRef {
 public:
  AutoScheduleContext(IntKey task_id, TIRGraph graph, Target target,
  int topk=20, int new_trial=4, std::string policy="profile") {
    auto node = make_object<AutoScheduleContextNode>();
    node->task_id = task_id;
    node->graph = graph;
    node->target = target;
    node->spaces = MultiScheduleSpace(graph, target);
    node->topk = topk;
    node->new_trial = new_trial;
    node->policy = policy;
    node->counts = 0U;
    data_ = std::move(node);
  }

  void add_feedback(ScheduleResult schedule_result, double evaluation);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AutoScheduleContext, ObjectRef, AutoScheduleContextNode);
};


class AutoScheduler {
 private:
  int topk;
  int new_trial;
  std::string policy;
  int parallel;
  int profile_parallel;
  double timeout;
  double profile_timeout;
  bool report_profile;
  unsigned warm_up_trials = 20;
  bool use_tensor_core = false;

  DLContext ctx;
  ThreadPool *thread_pool = nullptr;
  std::unordered_map<IntKey, AutoScheduleContext> contexts;
  std::ostream& log_out;
  std::ofstream profile_log;
  // Measurer *measurer = nullptr;
 public:
  AutoScheduler(DLContext context, int topk, int new_trial, std::string policy, int parallel,
  int profile_parallel, double timeout, double profile_timeout, bool report_profile=false,
  std::ostream& log_out=std::cerr, std::string log_file_name="autoschedule_log_profile.txt",
  bool use_tensor_core = false)
  : topk(topk), new_trial(new_trial), policy(policy), parallel(parallel),
    profile_parallel(profile_parallel), timeout(timeout),
    profile_timeout(profile_timeout), report_profile(report_profile),
    use_tensor_core(use_tensor_core),
    log_out(log_out) {
    ctx = context;
    thread_pool = new ThreadPool(parallel, (int)(timeout * 1000));
    std::vector<std::string> parts = string_split(".", log_file_name);
    profile_log.open(log_file_name, std::ios::app);
    // measurer = new Measurer(profile_parallel, profile_timeout);
  }
  ~AutoScheduler() {
    if (thread_pool != nullptr) delete thread_pool;
    profile_log.close();
    // if (measurer != nullptr) delete measurer;
  }
  void reset() {
    if (thread_pool != nullptr) {
      delete thread_pool; thread_pool = new ThreadPool(parallel, (int)(timeout * 1000));
    }
    // if (measurer != nullptr) {delete measurer; measurer = new Measurer(profile_parallel, profile_timeout);}
  }
  ScheduleResult schedule_func(IntKey key, TIRGraph subgraph, Target target);
  ScheduleResult schedule_with_entity(TIRGraph subgraph, Target target, MultiScheduleEntity entity);
  ScheduleResult schedule_with_external(TIRGraph subgraph, Target target, String external_schedule);
  std::shared_future<ScheduleResult> schedule_for(IntKey key, TIRGraph subgraph, Target target, int priority=0);
  void feedback_for(IntKey key, TIRGraph subgraph, Target target, ScheduleResult schedule_result, double evaluation);
  std::vector<double> judge_schedule(
    Array<te::Schedule> schedules, Array<te::Tensor> tensors, Target target, std::string policy, double gflop);
  void auto_schedule(TIRGraph subgraph, AutoScheduleContext &context, ScheduleResult &results);
  void clear_schedule_cache_for(IntKey key);
};



}  // namespace tg

}  // namespace tvm

#endif // TVM_TG_AUTOSCHEDULE_AUTO_SCHEDULE_H_