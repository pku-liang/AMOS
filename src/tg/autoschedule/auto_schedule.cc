#include <cmath>

#include "interpreter.h"
#include "auto_schedule.h"


namespace tvm {


namespace tg {

std::pair<te::Schedule, Array<te::Tensor> >
empty_schedule (TIRGraph subgraph) {
  te::Schedule sch = te::create_schedule(subgraph->root_ops);

  return std::make_pair(sch, subgraph->tensors);
}


double calculate_possibility(double x, double best, double upper=0.7) {
  return std::exp(x/best - 1.0) * upper;
}


std::vector<double> AutoScheduler::judge_schedule(
  Array<te::Schedule> schedules, Array<te::Tensor> tensors, Target target, std::string policy, double gflop) {
  const auto* f = runtime::Registry::Get("tg.autoschedule.query_cost_model");
  ASSERT(f != nullptr) << "Can't find tg.autoschedule.query_cost_model";
  std::vector<double> ret;
  Array<FloatImm> tmp = (*f)(schedules, tensors, target, policy);
  for (auto v : tmp) {
    if (v->value <= 0) {
      ret.push_back(0.0);
    } else {
      ret.push_back(gflop / (v->value / 1e3));
    }
  }

  return ret;
}


// auto_schedule for one subgraph
void AutoScheduler::auto_schedule(
    TIRGraph subgraph,
    AutoScheduleContext &context,
    ScheduleResult &results) {
  /* the empty schedule */
  te::Schedule sch;
  Array<te::Tensor> tensors;
  std::tie(sch, tensors) = empty_schedule(subgraph);

  /*
   * Allow external schedule
   * Use string to represent external schedule
   */
  if (context->target->kind->name == "cuda" && this->use_tensor_core) {
    const auto* f = runtime::Registry::Get("tg.autoschedule.auto_tensorize_cuda");
    ASSERT(f != nullptr) << "Can't find tg.autoschedule.auto_tensorize_cuda";
    std::string log_name = subgraph->tag + ".log";
    int trials = 400;
    Map<te::Schedule, Array<te::Tensor>> ret = (*f)(sch, tensors, log_name, trials);
    ASSERT(ret.size() == 1U);
    te::Schedule new_sch;
    Array<te::Tensor> new_tensors;
    for (auto kv : ret) {
      new_sch = kv.first;
      new_tensors = kv.second;
    }
    if (new_tensors.size() > 0U) {
      results = ScheduleResult(new_sch, new_tensors, log_name);
      return;
    }
  }

  /* the schedule logic
   * a schedule is two-level: skeleton + paramter
   * when the topk cache is empty, all random enumerated
   * when the topk cache is not empty, choose skeleton from cache
   * with possibility 'p', and random enumerate paramter
   * according to the chosen skeleton.
   * Otherwise, still all random.
   */
  std::vector<EvaluatedScheduleResult> reverse_sort;
  std::vector<double> p;
  while (!context->topk_schedules.empty()) {
    reverse_sort.push_back(context->topk_schedules.top());
    context->topk_schedules.pop();
  }

  // return these topks, otherwise, they will be lost
  for (auto ele : reverse_sort) {
    context->topk_schedules.push(ele);
  }

  int num_candidates = (int)(reverse_sort.size());
  // calculate possbilities
  for (auto e : reverse_sort) {
    p.push_back(
      calculate_possibility(
        e->evaluation, reverse_sort[num_candidates - 1]->evaluation, 1.0));
  }

  print(4, log_out) << "Moniter schedule context\n";
  if (num_candidates > 0)
    print(4, log_out) << "Best: [" << reverse_sort[num_candidates-1]->evaluation << "]\n";
  else
    print(4, log_out) << "Best: [inf]\n";
  for (int i = 0; i < num_candidates; ++i) {
    print(4, log_out) << "(" << i << ")" << reverse_sort[i]->evaluation << "[" << p[i] << "] ";
  }
  print(4, log_out) << "\n";

  // prepare new candidates
  std::vector<MultiScheduleEntity> new_candidates;
  int must_new = context->new_trial;
  while ((int)new_candidates.size() < context->new_trial) {
    print(4, log_out) << "schedule not full...\n";
    // choose a seed
    bool use_seed = false;
    EvaluatedScheduleResult seed;
    if (randdouble() < 0.8 && context->counts > warm_up_trials) {
      for (int k = 0; k < num_candidates; ++k) {
        int j = randint(k, num_candidates);
        if (randdouble() <= p[j]) {
          use_seed = true;
          seed = reverse_sort[j];
          print(4, log_out) << "choose " << j << "\n";
          break;
        }
      }
    }
    // choose new one
    MultiScheduleEntity new_one;
    if (use_seed) {
      print(4, log_out) << "Seed:\n";
      new_one = context->spaces.choose_one(seed->schedule_result->schedule_entities);
    } else {
      // pure random
      new_one = context->spaces.choose_one();
      print(4, log_out) << "Random:\n";
    }
    // if must_new, then must be new candidate never met before
    if (must_new > 0) {
      if ((context->known_schedules.find(new_one) == context->known_schedules.end())
          && (context->knowing_schedules.find(new_one) == context->knowing_schedules.end())) {
        new_candidates.push_back(new_one);
      } else {
        print(4, log_out) << "Repeat!\n";
      }
    } else {
      new_candidates.push_back(new_one);
    }
    // if (context->knowing_schedules.size() > 2000U) {
    //   context->known_schedules.clear();
    //   context->known_schedules = context->knowing_schedules;
    //   context->knowing_schedules.clear();
    // }
    must_new = -1;  // the next round, just relaxed
  }
  // choose from new candidates
  double best_value = -1;
  int best_ind = -1;
  int num_new_candidates = (int)new_candidates.size();
  Array<te::Schedule> tmp_schedules;
  for (int i = 0; i < num_new_candidates; ++i) {
    te::Schedule tmp_sch = te::create_schedule(subgraph->root_ops);
    interpret(tmp_sch, tensors, subgraph, context->target, new_candidates[i]);
    tmp_schedules.push_back(tmp_sch);
  }

  double gflop = get_gflop(subgraph);
  std::vector<double> tmp_judges = judge_schedule(tmp_schedules, tensors, context->target, context->policy, gflop);
  for (int i = 0; i < num_new_candidates; ++i) {
    // if (context->policy == "profile") {
    //   context.add_feedback(ScheduleResult(tmp_schedules[i], tensors, new_candidates[i]), tmp_judges[i]);
    // }
    if (tmp_judges[i] > best_value) {
      best_ind = i;
      best_value = tmp_judges[i];
    }
  }

  if (report_profile) {
    log_out << "check judge values:\n";
    for (auto v : tmp_judges) {
      log_out << v << " ";
    }
    log_out << "\n";
  }

  MultiScheduleEntity result_entity = new_candidates[best_ind];
  print(4, log_out) << "Check subgraph:\n" << subgraph->tag << "\n";
  print(4, log_out) << "Check schedule entity:\n" << result_entity.to_string() << "\n";
  interpret(sch, tensors, subgraph, context->target, result_entity);
  results = ScheduleResult(sch, tensors, result_entity);
  context->counts += 1;
  context->knowing_schedules.insert(result_entity);
}


void AutoScheduleContext::add_feedback(ScheduleResult schedule_result, double evaluation) {
  if (schedule_result->schedule_entities.defined()) {
    auto self = (*this);
    if (evaluation > 0.0) {
      EvaluatedScheduleResult evaluated = EvaluatedScheduleResult(schedule_result, evaluation);
      if ((int)self->topk_schedules.size() < self->topk) {
        self->topk_schedules.push(evaluated);
      } else {
        if (evaluated < self->topk_schedules.top()) {
          return;
        } else {
          self->topk_schedules.pop();
          self->topk_schedules.push(evaluated);
        }
      }
    }

    self->known_schedules.insert(schedule_result->schedule_entities);
    self->knowing_schedules.erase(schedule_result->schedule_entities);
    if (self->known_schedules.size() > 2000U) {
      int count = 0;
      std::vector<MultiScheduleEntity> to_delete;
      for (auto val : self->known_schedules) {
        to_delete.push_back(val);
        if (count > 1000)
          break;
        count += 1;
      }
      for (auto val : to_delete) {
        self->known_schedules.erase(val);
      }
    }
  }
}


ScheduleResult AutoScheduler::schedule_func(IntKey key, TIRGraph subgraph, Target target) {
  if (contexts.find(key) == contexts.end()) {
    contexts[key] = AutoScheduleContext(key, subgraph, target, topk, new_trial, policy);
  }

  AutoScheduleContext context = contexts[key];
  ScheduleResult results;
  auto_schedule(subgraph, context, results);
  
  return results;
}


ScheduleResult AutoScheduler::schedule_with_entity(
  TIRGraph subgraph, Target target, MultiScheduleEntity entity) {
  // if (contexts.find(key) == contexts.end()) {
  //   contexts[key] = AutoScheduleContext(key, subgraph, target, topk, new_trial, policy);
  // }

  te::Schedule sch;
  Array<te::Tensor> tensors;
  std::tie(sch, tensors) = empty_schedule(subgraph);
  interpret(sch, tensors, subgraph, target, entity);
  return ScheduleResult(sch, tensors, entity);
}


ScheduleResult AutoScheduler::schedule_with_external(
  TIRGraph subgraph, Target target, String external_schedule
) {
  if (target->kind->name == "cuda") {
    const auto* f = runtime::Registry::Get("tg.autoschedule.auto_tensorize_cuda");
    ASSERT(f != nullptr) << "Can't find tg.autoschedule.auto_tensorize_cuda";
    std::string log_name = external_schedule;
    int trials = 0;
    te::Schedule sch;
    Array<te::Tensor> tensors;
    std::tie(sch, tensors) = empty_schedule(subgraph);
    Map<te::Schedule, Array<te::Tensor>> ret = (*f)(sch, tensors, log_name, trials);
    ASSERT(ret.size() == 1U);
    te::Schedule new_sch;
    Array<te::Tensor> new_tensors;
    for (auto kv : ret) {
      new_sch = kv.first;
      new_tensors = kv.second;
    }
    ASSERT(new_tensors.size() > 0U);
    return ScheduleResult(new_sch, new_tensors, log_name);
  } else {
    ERROR << "Do not support schedule with external for target " << target << "\n";
    throw;
  }
}


std::shared_future<ScheduleResult> AutoScheduler::schedule_for(
  IntKey key, TIRGraph subgraph, Target target, int priority) {
  
  if (priority == 0) {
    return thread_pool->push_back(
      [this] (IntKey k, TIRGraph g, Target t) {
        return this->schedule_func(k, g, t);
        }, key, subgraph, target);
  } else if (priority == 1) {
    return thread_pool->push_front(
      [this] (IntKey k, TIRGraph g, Target t) {
        return this->schedule_func(k, g, t);
        }, key, subgraph, target);
  } else {
    ERROR << "Unsupported schedule priority: " << priority << "\n";
    throw;
  }
}


void AutoScheduler::feedback_for(IntKey key, TIRGraph subgraph, Target target, ScheduleResult schedule_result, double evaluation) {
  if (schedule_result->schedule_entities.defined()) {
    // const auto* f = runtime::Registry::Get("tg.autoschedule.store_feedback");
    // ASSERT(f != nullptr) << "Can't find tg.autoschedule.store_feedback";
    if (contexts.find(key) == contexts.end()) {
      contexts[key] = AutoScheduleContext(key, subgraph, target, topk, new_trial, policy);
    }
    contexts[key].add_feedback(schedule_result, evaluation);
    // Array<Feature> feature = get_feature(schedule_result->schedule, schedule_result->tensors, contexts[key]->target);
    // std::ostringstream oss;
    // double gflop = get_gflop(subgraph);

    // oss << "{ ";
    // oss << "\"gflop\": ";
    // oss << gflop << ", ";
    // oss << "\"loop_nests\": ";
    // oss << "[";
    // for (int i = 0; i < (int)feature.size(); ++i) {
    //   if (i != 0)
    //     oss << ", ";
    //   oss << std::pow(2, feature[i]->features[15]->value);
    // }
    // oss << "], ";
    // oss << "\"features\": ";
    // oss << "[";
    // for (int i = 0; i < (int)feature.size(); ++i) {
    //   if (i != 0)
    //     oss << ", ";
    //   oss << feature[i];
    // }
    // oss << "], ";
    // // oss << "\"schedules\": ";
    // // oss << "\"" << schedule_result->schedule_entities.to_string() << "\", ";
    // oss << "\"evaluation\": ";
    // oss << evaluation;
    // oss << " }\n";
    // profile_log << oss.str();
    
    // if (evaluation > 0)
    //   (*f)(oss.str());
  }
}


void AutoScheduler::clear_schedule_cache_for(IntKey key) {
  if (contexts.find(key) != contexts.end()) {
    auto context = contexts[key];
    context->knowing_schedules.clear();
  }
}


ScheduleResult get_schedule_result(
  String name,
  TIRGraph subgraph,
  Target target,
  int dev_id,
  int timeout,
  double perf=0.0,  // gflops
  bool do_feedback=false,
  ScheduleResult result=ScheduleResult()) {
  std::string name_key = std::string(name);
  static std::unordered_map<std::string, AutoScheduler*> scheduler_map;
  DLContext ctx;
  if (target->kind->name == "cuda") {
    ctx = DLContext({kDLGPU, dev_id});
  } else if (target->kind->name == "llvm") {
    ctx = DLContext({kDLCPU, dev_id});
  } else {
    ERROR << "Currently only support CUDA/LLVM but get " << target->kind->name << ".";
  }
  if (!scheduler_map.count(name_key)) {
    scheduler_map[name_key] = new AutoScheduler(
      /* DLContext context, */ ctx,
      /* int topk, */ 10,
      /* int new_trial, */ 10,
      /* std::string policy, */ "random",
      /* int parallel, */ 1,
      /* int profile_parallel, */ 1,
      /* double timeout, */ (double)timeout,
      /* double profile_timeout, */ (double)timeout
      /* bool report_profile=false, */
      /* std::ostream& log_out=std::cerr, */
      /* std::string log_file_name="autoschedule_log_profile.txt", */
      /* bool use_tensor_core = false */
    );
  }
  IntKey dummy_key = 0;
  if (do_feedback && result.defined()) {
    scheduler_map[name_key]->feedback_for(
      dummy_key, subgraph, target, result, perf
    );
    return result;
  }
  ScheduleResult ret_result = scheduler_map[name_key]->schedule_func(
    dummy_key, subgraph, target);
  return ret_result;
}


TVM_REGISTER_NODE_TYPE(ScheduleResultNode);


TVM_REGISTER_GLOBAL("tg.get_schedule_result_with_feedback")
.set_body_typed([](
  String name,
  TIRGraph subgraph,
  Target target,
  int dev_id,
  int timeout,
  double perf,
  bool do_feedback,
  ScheduleResult result
){
  return get_schedule_result(
    name, subgraph, target, dev_id, timeout, perf, do_feedback, result);
});


TVM_REGISTER_GLOBAL("tg.get_schedule_result_without_feedback")
.set_body_typed([](
  String name,
  TIRGraph subgraph,
  Target target,
  int dev_id,
  int timeout
){
  return get_schedule_result(
    name, subgraph, target, dev_id, timeout);
});


TVM_REGISTER_GLOBAL("tg.get_schedule_result_from_entity")
.set_body_typed([](
  String name,
  TIRGraph subgraph,
  Target target,
  MultiScheduleEntity entity
){
  DLContext ctx;
  if (target->kind->name == "cuda") {
    ctx = DLContext({kDLGPU, 0});
  } else if (target->kind->name == "llvm") {
    ctx = DLContext({kDLCPU, 0});
  } else {
    ERROR << "Currently only support CUDA/LLVM but get " << target->kind->name << ".";
  }
  AutoScheduler tmp(
      /* DLContext context, */ ctx,
      /* int topk, */ 10,
      /* int new_trial, */ 10,
      /* std::string policy, */ "random",
      /* int parallel, */ 1,
      /* int profile_parallel, */ 1,
      /* double timeout, */ (double)0,
      /* double profile_timeout, */ (double)0
      /* bool report_profile=false, */
      /* std::ostream& log_out=std::cerr, */
      /* std::string log_file_name="autoschedule_log_profile.txt", */
      /* bool use_tensor_core = false */
    );
  return tmp.schedule_with_entity(subgraph, target, entity);
});



}  // namespace tg


}  // namespace tvm