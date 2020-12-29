#ifndef TVM_TG_DRIVER_DRIVER_H_
#define TVM_TG_DRIVER_DRIVER_H_

#include <vector>
#include <unordered_map>
#include <chrono>
#include <queue>

#include <tvm/te/operation.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>

#include "utils.h"
#include "../graph/concrete_graph.h"
#include "../graph/subgraph.h"
#include "../autoschedule/auto_schedule.h"
#include "../build_function/build_function.h"
#include "../utils.h"
#include "../logging.h"
#include "../thread_pool.h"

namespace tvm {

namespace tg {

class SessionOptionNode : public Object {
 public:
  bool report_profile;
  bool report_iteration;
  int report_iteration_period;
  double autoschedule_trial_ratio;
  int autoschedule_topk;
  int autoschedule_new_trial;
  std::string autoschedule_policy;
  int autoschedule_parallel;
  double autoschedule_timeout;
  std::string autoschedule_log_file;
  int profile_parallel;
  double profile_timeout;
  int build_parallel;
  double build_timeout;
  std::string build_log_file;
  std::string evaluate_log_file;
  double execution_explore_probability;
  int execution_parallel;
  double execution_timeout;
  bool synchronize_subgraph;
  std::string execution_log_file;
  bool use_tensor_core;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("report_profile", &report_profile);
    v->Visit("report_iteration", &report_iteration);
    v->Visit("report_iteration_period", &report_iteration_period);
    v->Visit("autoschedule_trial_ratio", &autoschedule_trial_ratio);
    v->Visit("autoschedule_topk", &autoschedule_topk);
    v->Visit("autoschedule_new_trial", &autoschedule_new_trial);
    v->Visit("autoschedule_policy", &autoschedule_policy);
    v->Visit("autoschedule_timeout", &autoschedule_timeout);
    v->Visit("autoschedule_timeout", &autoschedule_timeout);
    v->Visit("autoschedule_log_file", &autoschedule_log_file);
    v->Visit("profile_parallel", &profile_parallel);
    v->Visit("profile_timeout", &profile_timeout);
    v->Visit("build_parallel", &build_parallel);
    v->Visit("build_timeout", &build_timeout);
    v->Visit("build_log_file", &build_log_file);
    v->Visit("evaluate_log_file", &evaluate_log_file);
    v->Visit("execution_explore_probability", &execution_explore_probability);
    v->Visit("execution_parallel", &execution_parallel);
    v->Visit("execution_timeout", &execution_timeout);
    v->Visit("synchronize_subgraph", &synchronize_subgraph);
    v->Visit("execution_log_file", &execution_log_file);
    v->Visit("use_tensor_core", &use_tensor_core);
  }

  static constexpr const char* _type_key = "tg.autoschedule.SessionOption";
  TVM_DECLARE_FINAL_OBJECT_INFO(SessionOptionNode, Object);
};


class SessionOption : public ObjectRef {
 public:
  SessionOption(
    bool report_profile,
    bool report_iteration,
    int report_iteration_period,
    double autoschedule_trial_ratio,
    int autoschedule_topk,
    int autoschedule_new_trial,
    std::string autoschedule_policy,
    int autoschedule_parallel,
    double autoschedule_timeout,
    std::string autoschedule_log_file,
    int profile_parallel,
    double profile_timeout,
    int build_parallel,
    double build_timeout,
    std::string build_log_file,
    std::string evaluate_log_file,
    double execution_explore_probability,
    int execution_parallel,
    double execution_timeout,
    bool synchronize_subgraph,
    std::string execution_log_file,
    bool use_tensor_core);
  
  SessionOption(int dummy);

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(SessionOption, ObjectRef, SessionOptionNode);
};


class Session {
 public:
  Target target;
  DLContext ctx;
  SessionOption sess_option;
  std::ofstream autoschedule_log;
  std::ofstream build_log;
  std::ofstream evaluate_log;
  std::ofstream exe_log;
  AutoScheduler *auto_scheduler = nullptr;
  FunctionBuilder *function_builder = nullptr;
  std::unordered_map<int, std::thread> sch_threads;
  std::unordered_map<int, std::thread> build_threads;
  std::unordered_map<int, std::thread> evaluate_threads;

  std::unordered_map<int, TIRGraph> task_graph;
  std::unordered_map<int, TIRMultiGraph> task_cache;
  std::unordered_map<int, std::vector<IntKey> > static_call_order;
  std::unordered_map<te::Tensor, tvm::runtime::NDArray> persistent_tensors;
  std::unordered_map<te::Tensor, tvm::runtime::NDArray> volatile_tensors;

  std::unordered_map<IntKey, Queue<std::pair<ScheduleResult,
    std::shared_future<tvm::runtime::Module> > > > future_functions;
  
  std::unordered_map<IntKey, Queue<std::tuple<ScheduleResult,
    tvm::runtime::Module, tvm::runtime::PackedFunc> > > built_functions;
  
  std::unordered_map<IntKey, Queue<std::tuple<ScheduleResult,
    tvm::runtime::Module, tvm::runtime::PackedFunc, double, double> > > best_functions;
  Queue<IntKey> emergency_schedule_queue;
  Queue<IntKey> normal_schedule_queue;
  Queue<IntKey> emergency_build_queue;
  Queue<IntKey> normal_build_queue;
  std::unordered_map<int, bool> finish;
  std::mutex finish_mutex;
  int task_count;
  std::unordered_map<int, bool> in_tuning;
  std::unordered_map<int, bool> cached_all_functions;
  // bool use_autoschedule;

 public:
  Session(Target target, int dev_id, SessionOption sess_option);
  ~Session();
  void clear_autoschedule_context();
  // void disable_autoschedule() {
  //   use_autoschedule = false;
  // }
  // void enable_autoschedule() {
  //   use_autoschedule = true;
  // }
  void initialize_weights(TIRGraph graph);
  void initialize_weights(TIRGraph graph, std::vector<tvm::runtime::NDArray> bindings);
  void allocate_output_buffer(TIRMultiGraph multi_graph);
  Array<tvm::runtime::NDArray> get_data(Array<te::Tensor> keys);
  std::string get_func_name(IntKey key);

  void run_autoschedule(
    int task_id, TIRMultiGraph multi_graph);

  void run_build(
    int task_id, TIRMultiGraph multi_graph);

  void run_evaluate(
    int task_id, TIRMultiGraph multi_graph, int advance_number,
    int first_stage_number=100, double second_stage_topk_ratio=0.1);

  void run_functions(
    int task_id,
    TIRMultiGraph multi_graph,
    std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings,
    std::string save_to="saved_schedules.txt",
    int profile_level=0,
    bool no_actual_run=false);
  
  int add_task(TIRGraph graph);
  void begin_tuning(int task_id, int advance_number, std::string reference="",
    int first_stage_number=100, double second_stage_topk_ratio=0.1);
  void end_tuning(int task_id);
  void prepare_for_test(int task_id, std::string reference);
  // int run(TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings);
  void run(
    int task_id,
    std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings,
    std::string save_to="saved_schedules.txt",
    int profile_level=0,
    bool no_actual_run=false);
};


std::shared_ptr<Session> create_or_get_session(
  Target target, int dev_id, SessionOption log_option, int& session_id, bool get_session=false);


int create_session(Target target, int dev_id, SessionOption log_option);

std::shared_ptr<Session> get_session(int session_id);


void initialize_weights(
  int session_id, TIRGraph graph, std::vector<tvm::runtime::NDArray> bindings);


// int run_graph(
//   int session_id, TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings);



}  // namespace tg


}  // namespace tvm


#endif  // TVM_TG_DRIVER_DRIVER_H_