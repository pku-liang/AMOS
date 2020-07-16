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


double judge_schedule(MultiScheduleEntity schedule_entity) {
  const auto* f = runtime::Registry::Get("tg.autoschedule.judge_schedule");
  // CHECK(f) << "Can't find tg.autoschedule.judge_schedule";
  if (f == nullptr) {
    return randdouble();
  } else {
    return (*f)(schedule_entity);
  }
}


// auto_schedule for one subgraph
void auto_schedule(
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
    for (int i = 0; i < context->number_per_trial; ++i) {
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
    }
    must_new = false;  // the second round, just relaxed
  }

  // choose from new candidates
  double best_value = -1;
  int best_ind;
  int count_cand = 0;
  for (auto cand : new_candidates) {
    double tmp = judge_schedule(cand);
    if (tmp > best_value) {
      best_ind = count_cand;
      best_value = tmp;
    }
    count_cand += 1;
  }

  MultiScheduleEntity result_entity = new_candidates[best_ind];

  interpret(sch, subgraph, context->target, result_entity);
  results = ScheduleResult(sch, tensors, new_candidates[best_ind]);
}


ScheduleResult AutoScheduler::schedule_func(IntKey key, TIRGraph subgraph, Target target) {
  if (contexts.find(key) == contexts.end()) {
    contexts[key] = AutoScheduleContext(key, subgraph, target);
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
    LOG(FATAL) << "Unsupported schedule priority: " << priority << "\n";
    throw;
  }
}


}  // namespace tg


}  // namespace tvm