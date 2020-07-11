#include "auto_schedule.h"


namespace tvm {


namespace tg {

// auto_schedule for one subgraph
bool auto_schedule(
    TIRGraph subgraph,
    AutoScheduleContext &context,
    std::vector<ScheduleResult> &results) {
  
  // one structure space for one operation
  // contains the structured schedule possibilities
  // mainly considers inline and use allreduce
  Array<StructureSpace> spaces = get_structure_spaces(subgraph, context->target);

  // all the structure space together define a search tree
  // the nodes of the search tree may contain parameters to tune
  // called partial configs
  // each time, try one path in the search tree
  std::shared_ptr<SearchTreeNode> leaf = expand_tree(
    subgraph, spaces, context.get_search_tree());

  int num_proposals = 10;
  
  ProposeOption option(context->target, num_proposals);
  RandomLeafProposer proposer(option);
  std::vector<std::vector<Config> > configs;
  leaf->random_leaf_propose(proposer, configs);

  int num_real_proposals = (int)configs.size();
  if (num_real_proposals == 0) {
    // no proposals in this round
    return false;
  }

  for (int i = 0; i < num_real_proposals; ++i) {
    te::Schedule sch;
    Array<te::Tensor> tensors;
    std::tie(sch, tensors) = interpret(subgraph, configs[i]);
    results.push_back(ScheduleResult(sch, tensors, leaf, configs[i]));
  }
  
  return true;
}


ScheduleResult AutoScheduler::schedule_func(IntKey key, TIRGraph subgraph, Target target) {
  if (contexts.find(key) == contexts.end()) {
    contexts[key] = AutoScheduleContext(target, key);
  }

  AutoScheduleContext context = contexts[key];
  std::vector<ScheduleResult> results;
  int num_trial = 0;
  while(!auto_schedule(subgraph, context, results)) {
    num_trial += 1;
    if (num_trial >= schedule_trials_for_one) {
      // no more schedule
      if (topk_schedules.find(key) != topk_schedules.end()) {
        // use the best
        return topk_schedules[key].top()->schedule_result;
      } else {
        throw std::runtime_error("Can't get schedule");
      }
    }
  }
  
  return results[0];
}


tvm::runtime::Module AutoScheduler::schedule_and_build_func(IntKey key, TIRGraph subgraph, Target target) {
  if (contexts.find(key) == contexts.end()) {
    contexts[key] = AutoScheduleContext(target, key);
  }

  AutoScheduleContext context = contexts[key];
  std::vector<ScheduleResult> results;
  int num_trial = 0;
  while(!auto_schedule(subgraph, context, results)) {
    num_trial += 1;
    if (num_trial >= schedule_trials_for_one) {
      // no more schedule
      if (topk_schedules.find(key) != topk_schedules.end()) {
        // use the best
        std::vector<EvaluatedScheduleResult> tmp;
        for (int i = 0; i < num_topk; ++i) {
          tmp.push_back(topk_schedules[key].top());
          topk_schedules[key].pop();
        }
        results.push_back(tmp.back()->schedule_result);
        for (auto v : tmp) {
          topk_schedules[key].push(v);
        }
      } else {
        throw std::runtime_error("Can't get schedule");
      }
    }
  }
  
  std::string name = "subgraph_" + std::to_string(key->value);
  std::unordered_map<te::Tensor, tir::Buffer> binds;
  Target target_host = Target::Create("llvm");
  tvm::BuildConfig config = tvm::BuildConfig::Create();

  auto func = tvm::build(
    tvm::lower(results[0]->schedule, results[0]->tensors, name, binds, config),
    target,
    target_host,
    config
  );

  return func;
}


std::future<ScheduleResult> AutoScheduler::schedule_for(
  IntKey key, TIRGraph subgraph, Target target, int priority) {
  
  if (priority == 0) {
    return ThreadPool::Global().push_back(
      [this] (IntKey k, TIRGraph g, Target t) {
        return this->schedule_func(k, g, t);
        }, key, subgraph, target);
  } else if (priority == 1) {
    return ThreadPool::Global().push_front(
      [this] (IntKey k, TIRGraph g, Target t) {
        return this->schedule_func(k, g, t);
        }, key, subgraph, target);
  } else {
    LOG(FATAL) << "Unsupported schedule priority: " << priority << "\n";
    throw;
  }
}


std::future<tvm::runtime::Module> AutoScheduler::schedule_and_build_for(
    IntKey key, TIRGraph subgraph, Target target, int priority) {

  if (priority == 0) {
    return ThreadPool::Global().push_back(
      [this] (IntKey k, TIRGraph g, Target t) {
        return this->schedule_and_build_func(k, g, t);
        }, key, subgraph, target);
  } else if (priority == 1) {
    return ThreadPool::Global().push_front(
      [this] (IntKey k, TIRGraph g, Target t) {
        return this->schedule_and_build_func(k, g, t);
        }, key, subgraph, target);
  } else {
    LOG(FATAL) << "Unsupported schedule priority: " << priority << "\n";
    throw;
  }
}


void AutoScheduler::feedback_schedule(IntKey key, ScheduleResult schedule_result, float feedback) {
  schedule_result.get_leaf()->update_reward(schedule_result->configs, feedback);

  if (topk_schedules.find(key) == topk_schedules.end()) {
    topk_schedules[key] = std::priority_queue<EvaluatedScheduleResult>();
  }
  if ((int)topk_schedules[key].size() < AutoScheduler::num_topk) {
    topk_schedules[key].push(EvaluatedScheduleResult(schedule_result, feedback));
  } else {
    if (feedback > topk_schedules[key].top()->evaluation) {
      // better feedback
      topk_schedules[key].pop();
      topk_schedules[key].push(EvaluatedScheduleResult(schedule_result, feedback));
    }
  }
}


AutoScheduler& AutoScheduler::Global() {
  static AutoScheduler* auto_scheduler = new AutoScheduler();
  return *auto_scheduler;
}


}  // namespace tg


}  // namespace tvm