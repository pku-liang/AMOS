#include <cmath>

#include "interpreter.h"
#include "auto_schedule.h"


namespace tvm {


namespace tg {

std::pair<te::Schedule, Array<te::Tensor> >
empty_schedule (TIRGraph subgraph) {
  te::Schedule sch = te::create_schedule(subgraph->root_ops);
  Array<te::Tensor> tensors;
  for (auto t : subgraph->inputs) {
    tensors.push_back(t);
  }
  for (auto t : subgraph->labels) {
    tensors.push_back(t);
  }
  for (auto t : subgraph->outputs) {
    tensors.push_back(t);
  }
  for (auto t : subgraph->weights) {
    tensors.push_back(t);
  }
  if (subgraph->loss.defined()) {
    tensors.push_back(subgraph->loss);
  }
  for (auto t : subgraph->gradients) {
    tensors.push_back(t);
  }
  if (subgraph->lr.defined()) {
    tensors.push_back(subgraph->lr);
  }
  for (auto t : subgraph->updates) {
    tensors.push_back(t);
  }

  return std::make_pair(sch, tensors);
}


double calculate_possibility(double x, double best, double upper=0.7) {
  return std::exp(x - best) * upper;
}


std::vector<double> AutoScheduler::judge_schedule(
  Array<te::Schedule> schedules, Array<te::Tensor> tensors, Target target, double gflop, std::string policy) {
  const auto* f = runtime::Registry::Get("tg.autoschedule.judge_schedule");
  // CHECK(f) << "Can't find tg.autoschedule.judge_schedule";
  std::vector<double> ret;
  if (f == nullptr) {
    if (policy == "profile") {
      ret = measurer->measure(schedules, tensors, target, ctx, gflop);
    } else if (policy == "random") {
      for (auto sch : schedules) {
        ret.push_back(randdouble());
      }
    } else {
      ERROR << "No support for policy: " << policy << ".";
    }
  } else {
    Array<FloatImm> tmp = (*f)(schedules, tensors, target, gflop, policy);
    for (auto v : tmp) {
      ret.push_back(v->value);
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
  int num_candidates = (int)(reverse_sort.size());
  // calculate possbilities
  for (auto e : reverse_sort) {
    p.push_back(
      calculate_possibility(
        e->evaluation, reverse_sort[num_candidates - 1]->evaluation, 0.7 * num_candidates / context->topk));
  }
  // choose a seed
  bool use_seed = false;
  EvaluatedScheduleResult seed;
  std::vector<ScheduleSkeleton> skeletons;
  for (int j = 0; j < num_candidates; ++j) {
    if (randdouble() < p[j]) {
      use_seed = true;
      seed = reverse_sort[j];
      break;
    }
  }
  if (use_seed) {
    for (auto se : seed->schedule_result->schedule_entities->entities) {
      skeletons.push_back(se->schedule_skeleton);
    }
  }

  // prepare new candidates
  std::vector<MultiScheduleEntity> new_candidates;
  bool must_new = true;
  while (new_candidates.size() == 0U) {
    for (int i = 0; i < context->new_trial; ++i) {
      MultiScheduleEntity new_one;
      if (use_seed) {
        new_one = context->spaces.choose_one(skeletons);
      } else {
        // pure random
        new_one = context->spaces.choose_one();
      }
      // if must_new, then must be new candidate never met before
      if (must_new) {
        if (context->known_schedules.find(new_one) == context->known_schedules.end()) {
          new_candidates.push_back(new_one);
        }
      } else {
        new_candidates.push_back(new_one);
      }
      context->known_schedules.insert(new_one);
    }
    must_new = false;  // the second round, just relaxed
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
  double gflop = 1;
  std::vector<double> tmp_judges = judge_schedule(tmp_schedules, tensors, context->target, gflop, context->policy);
  for (int i = 0; i < num_new_candidates; ++i) {
    if (context->policy == "profile") {
      context.add_feedback(ScheduleResult(tmp_schedules[i], tensors, new_candidates[i]), tmp_judges[i]);
    }
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

  interpret(sch, tensors, subgraph, context->target, result_entity);
  results = ScheduleResult(sch, tensors, new_candidates[best_ind]);
}


void AutoScheduleContext::add_feedback(ScheduleResult schedule_result, double evaluation) {
  EvaluatedScheduleResult evaluated = EvaluatedScheduleResult(schedule_result, evaluation);
  auto self = (*this);
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


ScheduleResult AutoScheduler::schedule_func(IntKey key, TIRGraph subgraph, Target target) {
  if (contexts.find(key) == contexts.end()) {
    contexts[key] = AutoScheduleContext(key, subgraph, target, topk, new_trial, policy);
  }

  AutoScheduleContext context = contexts[key];
  ScheduleResult results;
  auto_schedule(subgraph, context, results);
  
  return results;
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


void AutoScheduler::feedback_for(IntKey key, TIRGraph subgraph, ScheduleResult schedule_result, double evaluation) {
  contexts[key].add_feedback(schedule_result, evaluation);
  Feature feature = get_feature(schedule_result->schedule, schedule_result->tensors, contexts[key]->target);
  profile_log << feature << " : " << evaluation << "\n";
}


}  // namespace tg


}  // namespace tvm