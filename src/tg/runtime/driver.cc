#include "driver.h"
#include <unistd.h>
#include <cmath>
#include <set>
#include <map>


#include "../graph/concrete_graph.h"


namespace tvm {


namespace tg {

TVM_REGISTER_NODE_TYPE(SessionOptionNode);

SessionOption::SessionOption(
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
  std::string execution_log_file) {
  auto node = make_object<SessionOptionNode>();
  node->report_profile = report_profile;
  node->report_iteration = report_iteration;
  node->report_iteration_period = report_iteration_period;
  node->autoschedule_trial_ratio = autoschedule_trial_ratio;
  node->autoschedule_topk = autoschedule_topk;
  node->autoschedule_new_trial = autoschedule_new_trial;
  node->autoschedule_policy = autoschedule_policy;
  node->autoschedule_parallel = autoschedule_parallel;
  node->autoschedule_timeout = autoschedule_timeout;
  node->autoschedule_log_file = autoschedule_log_file;
  node->profile_parallel = profile_parallel;
  node->profile_timeout = profile_timeout;
  node->build_parallel = build_parallel;
  node->build_timeout = build_timeout;
  node->build_log_file = build_log_file;
  node->evaluate_log_file = evaluate_log_file;
  node->execution_explore_probability = execution_explore_probability;
  node->execution_parallel = execution_parallel;
  node->execution_timeout = execution_timeout;
  node->synchronize_subgraph = synchronize_subgraph;
  node->execution_log_file = execution_log_file;
  data_ = std::move(node);
}

SessionOption::SessionOption(int dummy) {
  auto node = make_object<SessionOptionNode>();
  data_ = std::move(node);
}


Session::Session(Target target, int dev_id, SessionOption sess_option)
: target(target), sess_option(sess_option) {
  if (target->target_name == "cuda") {
    ctx = DLContext({kDLGPU, dev_id});
  } else if (target->target_name == "llvm") {
    ctx = DLContext({kDLCPU, dev_id});
  } else {
    ERROR << "Currently only support CUDA/LLVM but get " << target->target_name << ".";
  }

  autoschedule_log.open(sess_option->autoschedule_log_file, std::ios::app);
  build_log.open(sess_option->build_log_file, std::ios::app);
  evaluate_log.open(sess_option->evaluate_log_file, std::ios::app);
  exe_log.open(sess_option->execution_log_file, std::ios::app);
  std::string profile_log_name = string_split(".", sess_option->autoschedule_log_file)[0] + "_profile.txt";
  auto_scheduler = new AutoScheduler(ctx, sess_option->autoschedule_topk, sess_option->autoschedule_new_trial,
    sess_option->autoschedule_policy, sess_option->autoschedule_parallel, sess_option->profile_parallel,
    sess_option->autoschedule_timeout, sess_option->profile_timeout, sess_option->report_profile,
    autoschedule_log, profile_log_name);
  function_builder = new FunctionBuilder(
    sess_option->build_parallel, sess_option->build_timeout, build_log);
  task_count = 0;
}


Session::~Session() {
  autoschedule_log.close();
  build_log.close();
  evaluate_log.close();
  exe_log.close();
  task_cache.clear();
  persistent_tensors.clear();
  volatile_tensors.clear();
  future_functions.clear();
  built_functions.clear();
  best_functions.clear();

  for (auto& th : sch_threads) {
    if (th.second.joinable()) {
      th.second.join();
    }
  }
  sch_threads.clear();

  for (auto& th : build_threads) {
    if (th.second.joinable()) {
      th.second.join();
    }
  }
  build_threads.clear();

  for (auto& th : evaluate_threads) {
    if (th.second.joinable()) {
      th.second.join();
    }
  }
  evaluate_threads.clear();

  if (auto_scheduler != nullptr) {
    delete auto_scheduler;
  }

  if (function_builder != nullptr) {
    delete function_builder;
  }
}


void Session::clear_autoschedule_context() {
  future_functions.clear();
  built_functions.clear();
}


void Session::initialize_weights(TIRGraph graph, std::vector<tvm::runtime::NDArray> bindings) {
  // static bindings
  // initialize for weights
  int num_weights = (int)graph->weights.size();
  ASSERT(num_weights == (int)bindings.size()) << "Initialize weights size mismatch.";
  for (int i = 0; i < num_weights; ++i) {
    persistent_tensors[graph->weights[i]] = bindings[i];
  }
  
  for (auto t : graph->gradients) {
    std::vector<int64_t> shape;
    for (auto p : t->shape) {
      shape.push_back(get_const_int(p));
    }
    // for gradients
    persistent_tensors[t] = tvm::runtime::NDArray::Empty(shape, t->dtype, ctx);
  }

  int i = 0;
  for (auto t : graph->updates) {
    ASSERT(persistent_tensors.find(graph->weights[i]) != persistent_tensors.end())
    << "Should initialize for weight " << graph->weights[i];
    // share buffer with weight
    persistent_tensors[graph->updates[i]] = persistent_tensors[graph->weights[i]];
    // std::vector<int64_t> shape;
    // for (auto p : t->shape) {
    //   shape.push_back(get_const_int(p));
    // }
    // persistent_tensors[graph->updates[i]] = tvm::runtime::NDArray::Empty(shape, t->dtype, ctx);
    i += 1;
  }

  // loss
  if (graph->loss.defined()) {
    te::Tensor t = graph->loss;
    if (persistent_tensors.find(t) == persistent_tensors.end()) {
      std::vector<int64_t> shape;
      for (auto p : t->shape) {
        shape.push_back(get_const_int(p));
      }
      persistent_tensors[t] = tvm::runtime::NDArray::Empty(shape, t->dtype, ctx);
    }
  }
}


void Session::allocate_output_buffer(TIRMultiGraph multi_graph) {
  for (auto kv : multi_graph->graphs) {
    // outputs
    for (auto t : kv.second->outputs) {
      te::Tensor old_t = multi_graph.Self()->tensor_index[t];
      if (volatile_tensors.find(old_t) == volatile_tensors.end()) {
        std::vector<int64_t> shape;
        for (auto p : old_t->shape) {
          shape.push_back(get_const_int(p));
        }
        volatile_tensors[old_t] = tvm::runtime::NDArray::Empty(shape, old_t->dtype, ctx);
      }
    }
    
  }
}


Array<tvm::runtime::NDArray> Session::get_data(Array<te::Tensor> keys) {
  Array<tvm::runtime::NDArray> ret;
  for (auto k : keys) {
    if (persistent_tensors.find(k) != persistent_tensors.end()) {
      ret.push_back(persistent_tensors[k]);
    } else if (volatile_tensors.find(k) != volatile_tensors.end()) {
      ret.push_back(volatile_tensors[k]);
    } else {
      ERROR << "Can't find the array for tensor " << k << ".\n";
    }
  }
  return ret;
}


std::string Session::get_func_name(IntKey key) {
  return "subgraph_" + std::to_string(key->value);
}

/* 
 *    autoschedule
 *     |       ^
 *     v       |
 *       build
 *     |       ^
 *     v       |
 *      evaluate <= head
 */


void Session::run_autoschedule(int task_id, TIRMultiGraph multi_graph) {
  std::unordered_map<IntKey, unsigned long long> counts;
  std::unordered_map<IntKey, unsigned long long> count_token;
  std::unordered_map<IntKey, unsigned long long> count_emergency;
  while (true) {
    // see if not finished
    bool peek_finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    peek_finish = this->finish[task_id];
    lock.unlock();
    if (peek_finish) {
      // execution done, no need to schedule
      return;
    }

    std::function<bool(IntKey key, bool emergency)> schedule_helper;
    schedule_helper = [&] (IntKey key, bool emergency) {
      TIRGraph subgraph = multi_graph.Self()->graphs[key];
      
      bool succ = false;
      /*
       * make a schedule
       */
      print(4, autoschedule_log) << "schedule for " << key->value << "\n";
      print(4, autoschedule_log) << "tag: " << subgraph->tag << "\n";
      for (auto op : subgraph->operation_list) {
        print(4, autoschedule_log) << "body: " << op.as<ComputeOpNode>()->body << "\n";
      }
      std::shared_future<ScheduleResult> schedule_result = auto_scheduler->schedule_for(
        key, subgraph, target, 0);

      try {
        print(4, autoschedule_log) << "Waiting for schedule for " << key->value << "...\n";
        ScheduleResult result = schedule_result.get();

        if (counts.find(key) == counts.end()) {
          counts[key] = 0U;
        }
        counts[key] += 1;

        print(4, autoschedule_log) << "Get schedule for " << key->value << " " << counts[key] <<  " times!\n";
        
        // get future func
        std::pair<ScheduleResult, std::shared_future<tvm::runtime::Module> > sch_func = \
        function_builder->build_for(
          result,
          target,
          Target::Create("llvm"),
          get_func_name(key),
          std::unordered_map<te::Tensor, tir::Buffer>(),
          tvm::BuildConfig::Create()
        );

        future_functions[key].push(sch_func);
        succ = true;
      } catch (const std::exception& e) {
        print(2, autoschedule_log) << "Can't get schedule: " << e.what() << "\n";
      }

      return succ;
    };

    if (!emergency_schedule_queue.empty()) {
      auto key = emergency_schedule_queue.front();
      emergency_schedule_queue.pop();
      if (count_emergency.find(key) == count_emergency.end()) {
        count_emergency[key] = 0U;
      }
      count_emergency[key] += 1;
      print(4, autoschedule_log) << "Emergency count for " << key->value << " is " << count_emergency[key] << "\n";
      while (!schedule_helper(key, true)) {
        // spin here
        // see if not finished
        bool peek_finish = false;
        std::unique_lock<std::mutex> lock(this->finish_mutex);
        peek_finish = this->finish[task_id];
        lock.unlock();
        if (peek_finish) {
          // execution done, no need to schedule
          return;
        }
      }
    }

    if (!normal_schedule_queue.empty()) {
      auto key = normal_schedule_queue.front();
      normal_schedule_queue.pop();
      if (count_token.find(key) == count_token.end()) {
        count_token[key] = 0U;
      }
      count_token[key] += 1;
      print(4, autoschedule_log) << "Token count for " << key->value << " is " << count_token[key] << "\n";
      while (!schedule_helper(key, true)) {
        // spin here
        // see if not finished
        bool peek_finish = false;
        std::unique_lock<std::mutex> lock(this->finish_mutex);
        peek_finish = this->finish[task_id];
        lock.unlock();
        if (peek_finish) {
          // execution done, no need to schedule
          return;
        }
      }
    }
  }  // while true
}


void Session::run_build(int task_id, TIRMultiGraph multi_graph) {
  std::unordered_map<IntKey, unsigned long long> counts;
  while (true) {
    // see if not done
    bool peek_finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    peek_finish = this->finish[task_id];
    lock.unlock();
    if (peek_finish)
      return;

    std::function<bool(IntKey key, bool emergency)> build_helper;
    build_helper = [&] (IntKey key, bool emergency) {
      TIRGraph subgraph = multi_graph.Self()->graphs[key];
      bool succ = false;
      bool taken = false;
      /*
        * make a build
        */
      if (!future_functions[key].empty()) {
        if (counts.find(key) == counts.end()) {
          counts[key] = 0U;
        }
        counts[key] += 1;
        print(4, build_log) << "build for " << key->value << " " << counts[key] << " times\n";
        auto sch_and_mod = future_functions[key].front();
        ScheduleResult sch = sch_and_mod.first;
        auto future_mod = sch_and_mod.second;
        future_functions[key].pop();
        taken = true;
        try {
          print(4, build_log) << "Waiting for build for " << key->value << "...\n";
          tvm::runtime::Module mod = future_mod.get();
          tvm::runtime::PackedFunc func = mod->GetFunction(get_func_name(key));
          print(4, build_log) << "Get build for " << key->value << "!\n";

          built_functions[key].push(std::make_tuple(sch, mod, func));

          succ = true;
        } catch (const std::exception &e) {
          print(2, build_log) << "Can't get build for: " << e.what() << "\n";
          auto_scheduler->feedback_for(key, subgraph, target, sch, 0.0);
        }  // try catch
      }  // if (!future_functions[key].empty())

      if (!succ && taken) {
        if (emergency) {
          emergency_schedule_queue.push(key);
        } else {
          normal_schedule_queue.push(key);
        }
      }

      return succ;
    };

    if (!emergency_build_queue.empty()) {
      // send a token to autoscheduler
      auto key = emergency_build_queue.front();
      emergency_build_queue.pop();
      emergency_schedule_queue.push(key);
      while (!build_helper(key, true)) {
        // spin here
        // see if not finished
        bool peek_finish = false;
        std::unique_lock<std::mutex> lock(this->finish_mutex);
        peek_finish = this->finish[task_id];
        lock.unlock();
        if (peek_finish) {
          // execution done, no need to schedule
          return;
        }
      }
    }

    if (!normal_build_queue.empty()) {
      auto key = normal_build_queue.front();
      normal_build_queue.pop();
      normal_schedule_queue.push(key);
      while (!build_helper(key, false)) {
        // spin here
        // see if not finished
        bool peek_finish = false;
        std::unique_lock<std::mutex> lock(this->finish_mutex);
        peek_finish = this->finish[task_id];
        lock.unlock();
        if (peek_finish) {
          // execution done, no need to schedule
          return;
        }
      }
    }

  }  // while true
}


void Session::run_evaluate(
  int task_id, TIRMultiGraph multi_graph, int advance_number, int first_stage_number, double second_stage_topk_ratio) {
  // int second_stage_topk = std::ceil((int)(multi_graph->graphs.size()) * second_stage_topk_ratio);
  // prepare the evaluate_performance
  const auto* evaluate_performance = runtime::Registry::Get("tg.runtime.evaluate_performance");
  ASSERT(evaluate_performance != nullptr) << "Should prepare tg.runtime.evaluate_performance function.";
  std::unordered_map<IntKey, unsigned long long> counts;
  // std::unordered_map<IntKey, double> performance;

  // first of all, push some tokens
  // and keep scheduler busy
  // for (auto key : static_call_order[task_id]) {
  //   normal_build_queue.push(key, sess_option->execution_parallel);
  // }
  print(1, evaluate_log) << "To evaluate " << advance_number << " iterations.\n";
  std::unordered_map<std::string, int> next_round_tokens;
  for (auto k : static_call_order[task_id]) {
    auto subgraph = multi_graph.Self()->graphs[k];
    next_round_tokens[subgraph->tag] = sess_option->autoschedule_parallel;
  }

  for (int ad = 0; ad < advance_number; ++ad) {
    print(1, evaluate_log) << "Iteration " << ad << "\n";
    // decide if evaluate the whole graph
    // if in first stage, then evaluate the whole graph
    // bool in_first_stage = (ad < first_stage_number)
    //     || (cached_all_functions.find(task_id) == cached_all_functions.end())
    //     || (!cached_all_functions[task_id])
    //     || (randdouble() < 0.1);
    // check if finish
    bool finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    finish = this->finish[task_id];
    lock.unlock();
    if (finish) {
      return;
    }
    // initialize cache
    // this cache is used to aovid repeat work
    std::unordered_map<std::string, IntKey> evaluate_cache;

    std::unordered_map<IntKey, int> evaluate_order;
    std::unordered_set<IntKey> free_set;
    std::unordered_map<IntKey, double> topk_list;
    for (auto kv : multi_graph->graph_attrs) {
      evaluate_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.first);
      }
    }
    // not in first stage, only concentrate on the slow subgraphs
    // if (!in_first_stage) {
    //   std::priority_queue<KeyAndTime> max_heap;
    //   free_set.clear();
    //   for (auto& kv : this->best_functions) {
    //     if (kv.second.empty()) {
    //       continue;
    //     }
    //     auto front = kv.second.front();
    //     auto time = std::get<4>(front);
    //     max_heap.push(KeyAndTime(kv.first, time));
    //   }
    //   for (int i = 0; i < second_stage_topk; ++i) {
    //     if (max_heap.empty()) {
    //       break;
    //     }
    //     auto top = max_heap.top();
    //     topk_list[top.key] = top.time;
    //     free_set.insert(top.key);
    //     max_heap.pop();
    //   }
    //   print(4, evaluate_log) << "importance tuning:\n";
    //   for (auto k : free_set) {
    //     print(4, evaluate_log) << k->value << " ";
    //   }
    //   print(4, evaluate_log) << "\n";
    // }

    /* the evaluate helper
     * handles one subgraph at a time
     * fill the delete set and update set
     */
    std::function<void(IntKey key,
    std::unordered_set<IntKey> &delete_set, std::unordered_set<IntKey> &update_set)> evaluate_helper;
    evaluate_helper = [&]
      (IntKey key, std::unordered_set<IntKey> &delete_set, std::unordered_set<IntKey> &update_set) {

      // the mark that indicates this subgraph is done
      bool succ = false;

      TIRGraph subgraph = multi_graph.Self()->graphs[key];
      
      int taken = 0;  // record taken a function

      // try to find repeated subgraph
      if (!succ) {
        if (evaluate_cache.find(subgraph->tag) != evaluate_cache.end()) {
          print(4, evaluate_log) << "Find repeated function, skip evaluation" << subgraph->tag << ".\n";
          IntKey repeat_key = evaluate_cache[subgraph->tag];
          if (!best_functions[repeat_key].empty()) {
            auto mod_func_perf = (best_functions[repeat_key].front());
            if (best_functions[key].empty()) {
              best_functions[key].push(mod_func_perf);
            } else {
              best_functions[key].push(mod_func_perf);
              best_functions[key].pop();
            }
            
            print(4, evaluate_log) << "Push cache function.\n";
            succ = true;
          }
        }
      }  // end try use repeated function
      
      // built functions not empty
      // evaluate this wanted function
      if (!succ && !this->built_functions[key].empty()) {
        Array<tvm::runtime::Module> modules;
        Array<ScheduleResult> schedules;
        Array<te::Tensor> tensors;
        std::vector<tvm::runtime::PackedFunc> functions;
        for (int i = 0; i < sess_option->execution_parallel; ++i) {
          if (this->built_functions[key].empty())
            break;
          auto sch_mod_func = this->built_functions[key].front();
          // take away this one
          this->built_functions[key].pop();
          taken += 1;   // we will take one function
          // check if is ready
          auto schedule_result = std::get<0>(sch_mod_func);
          schedules.push_back(schedule_result);
          tensors = schedule_result->tensors;
          auto mod = std::get<1>(sch_mod_func);
          modules.push_back(mod);
          auto func = std::get<2>(sch_mod_func);
          ASSERT(func != nullptr) << "Get null function, don't know how to deal with it.";
          functions.push_back(func);
        }
        
        /* 
          * run the function in another process
          * to get performance, if return -1
          * then timeout or fail in execution
          */
        Array<FloatImm> elapsed_times = (*evaluate_performance)(modules, get_func_name(key), tensors);
        if (counts.find(key) == counts.end()) {
          counts[key] = 0;
        }
        counts[key] += taken;
        print(4, evaluate_log) << "Evaluate for key: " << key->value << " " << counts[key] << " times\n";

        int best_id = 0;
        double best_perf = 0.0;
        double best_time = 0.0;
        for (int i = 0; i < taken; ++i) {
          double elapsed_time = elapsed_times[i]->value;
          auto mod = modules[i];
          auto schedule_result = schedules[i];
          auto func = functions[i];
          print(4, evaluate_log) << "evaluate result for " << key->value << " is " << elapsed_time << " ms.\n";
          if (elapsed_time > 0) {
            // feedback
            double gflops = get_gflop(subgraph) / (elapsed_time / 1e3 + 1e-8);
            auto_scheduler->feedback_for(key, subgraph, target, schedule_result, gflops);

            if (gflops > best_perf) {
              best_id = i;
              best_perf = gflops;
              best_time = elapsed_time;
            }
            
          } else {
            // can't run the function
            print(2, evaluate_log) << "Can't evaluate function: " << "\n";
            // auto sub_mods = mod->imports();
            // if (sub_mods.size() > 0U) {
            //   runtime::Module sub_mod = (mod->imports().at(0));
            //   print(4, evaluate_log) << "Check source:\n" << sub_mod->GetSource() << "\n";
            // }
            // feedback
            auto_scheduler->feedback_for(key, subgraph, target, schedule_result, 0.0);
          }
        }

        if (best_perf > 0) {
          // store function
          int to_put = 1;
          if (best_functions[key].empty()) {
            print(4, evaluate_log) << "set best function for " << key->value << ": " << best_perf << " GFLOPS.\n";
            best_functions[key].push(std::make_tuple(
              schedules[best_id], modules[best_id], functions[best_id], best_perf, best_time));
          } else {
            auto best = best_functions[key].front();
            double previous_perf = std::get<3>(best);
            if (best_perf > previous_perf) {
              print(4, evaluate_log) << "replace best function for "
                                    << key->value << ": " << best_perf << " GFLOPS."
                                    << "(original " << previous_perf << " GFLOPS)\n";
              best_functions[key].push(
                std::make_tuple(schedules[best_id], modules[best_id], functions[best_id], best_perf, best_time));
              best_functions[key].pop();
              double ratio = std::min((best_perf - previous_perf) / previous_perf, 1.0);
              int additional_tokens = std::ceil(ratio * sess_option->autoschedule_parallel);
              to_put += additional_tokens;
              // normal_build_queue.push(key, additional_tokens);
            }
          }

          // success
          succ = true;
          evaluate_cache[subgraph->tag] = key;
          // normal_build_queue.push(key, to_put);
          // if (performance.find(key) == performance.end()) {
          //   performance[key] = best_time;
          // } else {
          //   if (best_time < performance[key]) {
          //     performance[key] = best_time;
          //   }
          // }
        }
      }// end try new function

      // must check taken so that we won't overuse builder
      if (!succ && taken > 0) {
        emergency_build_queue.push(key, taken);
      }

      if (succ) {
        // update free set
        delete_set.insert(key);
        for (auto v : multi_graph.Self()->graph_attrs[key]->successors) {
          evaluate_order[v] -= 1;
          if (evaluate_order[v] == 0) {
            update_set.insert(v);
          }
        }
      }
    };  // end evaluate helper
    
    std::unordered_set<std::string> sent_token;
    for (auto k : static_call_order[task_id]) {
      TIRGraph subgraph = multi_graph.Self()->graphs[k];
      if (sent_token.find(subgraph->tag) == sent_token.end()) {
        // send a token to build thread
        // so that build thread will produce one function
        // for this key
        // if (ad < first_stage_number) {
        //   normal_build_queue.push(k, sess_option->execution_parallel);
        // } else {
        //   normal_build_queue.push(k, 1);
        // }
        int to_put = sess_option->autoschedule_parallel;
        normal_build_queue.push(k, to_put);
        sent_token.insert(subgraph->tag);
      }
    }

    while (!free_set.empty()) {
      // check if finish
      bool finish = false;
      std::unique_lock<std::mutex> lock(this->finish_mutex);
      finish = this->finish[task_id];
      lock.unlock();
      if (finish) {
        return;
      }
      std::unordered_set<IntKey> update_set, delete_set;
      for (auto k : free_set) {
        evaluate_helper(k, delete_set, update_set);
      }
      for (auto k : delete_set) {
        free_set.erase(k);
      }
      for (auto k : update_set) {
        free_set.insert(k);
      }

      // if (!in_first_stage) {
      //   free_set.clear();
      // }
    }  // while (!free_set.empty())
    cached_all_functions[task_id] = true;
  }  // for ad
  print(1, evaluate_log) << "Stop evaluation.\n";
}


void Session::run_functions(
  int task_id,
  TIRMultiGraph multi_graph,
  std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings,
  std::string save_to,
  int profile_level,
  bool no_actual_run) {
  ASSERT(static_call_order.find(task_id) != static_call_order.end()) << "Can't find task " << task_id
      << "\nDid you forget to add task first?\n";

  if (!no_actual_run) {
    auto* call_unpack = new CallFunc<tvm::runtime::PackedFunc, tvm::runtime::NDArray>();

    int advance_number = (int)bindings.size();
    ProgressBar progress_bar;

    std::vector<std::unordered_map<IntKey, std::vector<tvm::runtime::NDArray> > > ad_arrays;

    for (int ad = 0; ad < advance_number; ++ad) {
      std::unordered_map<IntKey, std::vector<tvm::runtime::NDArray> > array_map;
      for (auto key : static_call_order[task_id]) {
        TIRGraph subgraph = multi_graph.Self()->graphs[key];
        /* get the runtime array
          * the order is the same as build
          * TODO: handle the order by some other
          * independent logic
          */
        std::vector<tvm::runtime::NDArray> arrays;
        for (auto tt : subgraph->tensors) {
          te::Tensor t = multi_graph.Self()->tensor_index[tt];
          if (bindings[ad].find(t) != bindings[ad].end()) {
            arrays.push_back(bindings[ad][t]);
          } else if (this->volatile_tensors.find(t) != this->volatile_tensors.end()) {
            arrays.push_back(this->volatile_tensors[t]);
          } else if (this->persistent_tensors.find(t) != this->persistent_tensors.end()) {
            arrays.push_back(this->persistent_tensors[t]);
          } else {
            ERROR << "Can't find array for tensor " << t;
          }
        }
        // // get the inputs
        // for (auto tt : subgraph->inputs) {
        //   te::Tensor t = multi_graph.Self()->tensor_index[tt];
        //   if (bindings[ad].find(t) != bindings[ad].end()) {
        //     arrays.push_back(bindings[ad][t]);
        //   } else if (this->volatile_tensors.find(t) != this->volatile_tensors.end()) {
        //     arrays.push_back(this->volatile_tensors[t]);
        //   } else {
        //     ERROR << "Can't find input " << t;
        //   }
        // }

        // // get the labels
        // for (auto tt : subgraph->labels) {
        //   te::Tensor t = multi_graph.Self()->tensor_index[tt];
        //   if (bindings[ad].find(t) == bindings[ad].end()) {
        //     ERROR << "Can't find label " << t;
        //   }
        //   arrays.push_back(bindings[ad][t]);
        // }

        // // get the outputs
        // for (auto tt : subgraph->outputs) {
        //   te::Tensor t = multi_graph.Self()->tensor_index[tt];
        //   if (volatile_tensors.find(t) == volatile_tensors.end()) {
        //     ERROR << "Can't find output " << t;
        //   }
        //   arrays.push_back(this->volatile_tensors[t]);
        // }

        // // get the weights
        // for (auto tt : subgraph->weights) {
        //   te::Tensor t = multi_graph.Self()->tensor_index[tt];
        //   if (persistent_tensors.find(t) == persistent_tensors.end()) {
        //     ERROR << "Can't find weight " << t;
        //   }
        //   arrays.push_back(this->persistent_tensors[t]);
        // }

        // // get the loss
        // if (subgraph->loss.defined()) {
        //   te::Tensor t = multi_graph.Self()->tensor_index[subgraph->loss];
        //   if (persistent_tensors.find(t) == persistent_tensors.end()) {
        //     ERROR << "Can't find loss " << t;
        //   }
        //   arrays.push_back(this->persistent_tensors[t]);
        // }
        
        // // get the gradients
        // for (auto tt : subgraph->gradients) {
        //   te::Tensor t = multi_graph.Self()->tensor_index[tt];
        //   if (persistent_tensors.find(t) == persistent_tensors.end()) {
        //     ERROR << "Can't find gradient " << t;
        //   }
        //   arrays.push_back(this->persistent_tensors[t]);
        // }
        
        // // get the lr
        // if (subgraph->lr.defined()) {
        //   te::Tensor t = multi_graph.Self()->tensor_index[subgraph->lr];
        //   if (bindings[ad].find(t) == bindings[ad].end()) {
        //     ERROR << "Can't find lr " << t;
        //   }
        //   arrays.push_back(bindings[ad][t]);
        // }
        
        // // get the updates
        // for (auto tt : subgraph->updates) {
        //   te::Tensor t = multi_graph.Self()->tensor_index[tt];
        //   if (persistent_tensors.find(t) == persistent_tensors.end()) {
        //     ERROR << "Can't find update " << t;
        //   }
        //   arrays.push_back(this->persistent_tensors[t]);
        // }

        array_map[key] = arrays;
      }

      ad_arrays.push_back(array_map);
    }

    std::priority_queue<double> time_queue;
    for (int ad = 0; ad < advance_number; ++ad) {
      if (sess_option->report_iteration) {
        exe_log << "Iteration: " << ad << "\n";
      }
      progress_bar.draw(((double)(ad + 1) / advance_number));
      if (ad == advance_number - 1) {
        progress_bar.end();
      }

      /* the run helper
      * handles one subgraph at a time
      * fill the delete set and update set
      */
      std::function<bool(IntKey key)> run_helper;
      run_helper = [&] (IntKey key) {
        // print(4, exe_log) << "do " << key->value << "\n";

        // the mark that indicates this subgraph is done
        bool succ = false;
        /* get the runtime array
        * the order is the same as build
        * TODO: handle the order by some other
        * independent logic
        */
        std::vector<tvm::runtime::NDArray> arrays = ad_arrays[ad][key];

        if (!this->best_functions[key].empty()) {
          auto mod_func = this->best_functions[key].front();
          auto sch = std::get<0>(mod_func);
          auto mod = std::get<1>(mod_func);
          auto func = std::get<2>(mod_func);
          ASSERT(func != nullptr) << "Get null function, don't know how to deal with it.";

          if (profile_level >= 2) {
            TIRGraph subgraph = multi_graph.Self()->graphs[key];
            // print(4, exe_log) << sch->schedule_entities.to_string() << "\n";
            // for (auto op : subgraph->operation_list) {
            //   print(4, exe_log) << op.as<ComputeOpNode>()->axis << " " <<  op.as<ComputeOpNode>()->body << "\n";
            // }
            
            // for (auto t : subgraph->tensors) {
            //   print(4, exe_log) << t << " ";
            // }
            // print(4, exe_log) << "\n";
            // for (auto t : arrays) {
            //   Array<PrimExpr> t_shape;
            //   for (auto s : t.Shape()) {
            //     t_shape.push_back((int)s);
            //   }
            //   print(4, exe_log) << t_shape << " ";
            // }
            // print(4, exe_log) << "\n";
            // print(4, exe_log) << subgraph->tag << "\n";
            // auto lowered = tvm::lower(sch->schedule, sch->tensors, get_func_name(key),
            //   std::unordered_map<te::Tensor, tir::Buffer>(),
            //   tvm::BuildConfig::Create());
            // print(4, exe_log) << lowered << "\n";
            // auto sub_mods = mod->imports();
            // if (sub_mods.size() > 0U) {
            //   runtime::Module sub_mod = (mod->imports().at(0));
            //   print(4, exe_log) << "Check source:\n" << sub_mod->GetSource() << "\n";
            // }

            // tvm::runtime::Module another_mod = tvm::build(
            //   lowered,
            //   target,
            //   Target::Create("llvm"),
            //   tvm::BuildConfig::Create()
            // );
            // auto another_sub_mods = another_mod->imports();
            // if (another_sub_mods.size() > 0U) {
            //   runtime::Module another_sub_mod = (another_mod->imports().at(0));
            //   print(4, exe_log) << "Check another source:\n" << another_sub_mod->GetSource() << "\n";
            // }

            auto beg = std::chrono::steady_clock::now();
            (*call_unpack)(func, arrays);
            runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
            auto end = std::chrono::steady_clock::now();
            double execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() / 1e3;
            print(1, exe_log) << "Subgraph: " << key->value << "\n"
                              << "-------------------------------------------------\n";
            for (auto op : subgraph->operation_list) {
              print(1, exe_log) << op.as<ComputeOpNode>()->body << "\n";
            }
            print(1, exe_log) << "Time cost: " << execution_time << " ms.\n";
          } else {
            (*call_unpack)(func, arrays);
          }

          // success
          succ = true;
        }  // end try new function

        // print(4, exe_log) << "end " << key->value << "\n";
        return succ;

      };  // end run helper
      
      auto beg = std::chrono::steady_clock::now();
      for (auto k : static_call_order[task_id]) {
        while (!run_helper(k)) {
        }
      }

      if (profile_level >= 1) {
        // synchronize the stream for this run task
        runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
        auto end = std::chrono::steady_clock::now();
        double execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() / 1e3;
        if ((advance_number > 1 && ad > 0) || (advance_number == 1))
          time_queue.push(execution_time);

        print(1, exe_log) << "time cost: " << execution_time << " ms.\n";
      }
    }  // for ad
    
    if (profile_level >= 1) {
      double max_time = time_queue.top();
      double median_time, min_time;
      size_t total_num = time_queue.size();
      double total_time = 0.0;
      for (size_t i = 0; i <= total_num / 2; ++i) {
        median_time = time_queue.top();
        total_time += median_time;
        min_time = median_time;
        time_queue.pop();
      }
      while (time_queue.size() >= 1) {
        min_time = time_queue.top();
        total_time += min_time;
        time_queue.pop();
      }
      print(1, exe_log) << "Time report: min=[" << min_time
                        << " ms], med=[" << median_time
                        << " ms], max=[" << max_time
                        << " ms], avg=[" << total_time / total_num << " ms]\n\n\n";
    }
  }
  // save the functions
  if (save_to != "") {
    std::unordered_map<std::string, std::tuple<std::string, double, double> > stored;
    std::fstream fs(save_to, std::ios::in);
    std::string line;
    if (fs) {
      while (std::getline(fs, line)) {
        std::vector<std::string> parts = string_split("|", line);
        double perf = std::stod(parts[2]);
        double time = std::stod(parts[3]);
        stored[parts[0]] = std::make_tuple(parts[1], perf, time);
      }
      fs.close();
    }

    for (auto& kv : best_functions) {
      if (!kv.second.empty()) {
        auto sch_mod_func_perf = kv.second.front();
        std::string tag = multi_graph.Self()->graphs[kv.first]->tag;
        std::string entity_string = std::get<0>(sch_mod_func_perf)->schedule_entities.to_string();
        double perf = std::get<3>(sch_mod_func_perf);
        double time = std::get<4>(sch_mod_func_perf);
        if (stored.find(tag) == stored.end()) {
          // std::string out_line = 
          //   tag + "|" + std::get<0>(sch_mod_func_perf)->schedule_entities.to_string() 
          //   + "|" + std::to_string(std::get<3>(sch_mod_func_perf)) + "|" + std::to_string(std::get<4>(sch_mod_func_perf));
          stored[tag] = std::make_tuple(entity_string, perf, time);
        } else {
          if (perf > std::get<1>(stored[tag])) {
            stored[tag] = std::make_tuple(entity_string, perf, time);
          }
        }
      }
    }
    std::fstream fout(save_to, std::ios::out);
    for (auto kv : stored) {
      fout << string_join("|",
        {kv.first,
         std::get<0>(kv.second),
         std::to_string(std::get<1>(kv.second)),
         std::to_string(std::get<2>(kv.second))}) << "\n" << std::flush;
    }
    fout.close();
  }

  // synchronize the stream for this run task
  runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
  // notify done
  // std::unique_lock<std::mutex> lock(this->finish_mutex);
  // this->finish = true;
  // lock.unlock();
}


int Session::add_task(TIRGraph graph) {
  SubGraphPartitionEngine partition_engine;
  TIRMultiGraph multi_graph(graph, partition_engine);

  // allocate output/loss/gradients/updates buffer
  // the weight buffers should be initialized before
  allocate_output_buffer(multi_graph);

  int task_id = task_count++;
  task_cache[task_id] = multi_graph;
  int num_subgraphs = (int)multi_graph->graphs.size();
  print(3) << "Num subgraphs: " << num_subgraphs << "\n";

  std::unordered_set<std::string> tags;
  for (auto subgraph : multi_graph->graphs) {
    tags.insert(subgraph.second->tag);
  }
  print(3) << "Num unique subgraphs: " << tags.size() << "\n";

  std::vector<IntKey> order;

  std::unordered_map<IntKey, int> call_order;
  std::set<IntKey> free_set;
  for (auto kv : multi_graph->graph_attrs) {
    call_order[kv.first] = kv.second->num_predecessor;
    if (kv.second->num_predecessor == 0) {
      free_set.insert(kv.first);
    }
  }

  while (!free_set.empty()) {
    std::unordered_set<IntKey> update_set;
    for (auto k : free_set) {
      order.push_back(k);
      for (auto v : multi_graph.Self()->graph_attrs[k]->successors) {
        call_order[v] -= 1;
        if (call_order[v] == 0) {
          update_set.insert(v);
        }
      }
    }
    free_set.clear();
    for (auto k : update_set) {
      free_set.insert(k);
    }
  }

  static_call_order[task_id] = order;
  return task_id;
}


void Session::prepare_for_test(int task_id, std::string reference) {
  // load reference
  // this will add additional schedule results
  std::unordered_map<std::string, IntKey> need_functions;
  std::unordered_map<
    std::string,
    std::shared_future<
      std::tuple<ScheduleResult, runtime::Module, runtime::PackedFunc, double, double> > > target_subgraph_functions;
  /* get subgraph */
  ASSERT(task_cache.find(task_id) != task_cache.end()) << "No such task " << task_id << "\n";
  TIRMultiGraph multi_graph = task_cache[task_id];
  for (auto kv : multi_graph.Self()->graphs) {
    need_functions[kv.second->tag] = kv.first;
  }

  /* helper for schedule and build */
  ThreadPool pool;
  auto schedule_and_build = [&] (IntKey key, std::vector<std::string> parts) {
    MultiScheduleEntity entity = multi_schedule_entity_from_string(parts[1]);
    double perf = std::stod(parts[2]);
    double time = std::stod(parts[3]);
    ScheduleResult schedule_result = auto_scheduler->schedule_with_entity(
      multi_graph.Self()->graphs[key], target, entity);
      std::string name = get_func_name(key);
      
      auto module = function_builder->build_func(
        schedule_result->schedule, schedule_result->tensors, target, Target::Create("llvm"),
        name, std::unordered_map<te::Tensor, tir::Buffer>(), tvm::BuildConfig::Create());

      auto func = module->GetFunction(name);
      return std::make_tuple(schedule_result, module, func, perf, time);
  };
  // std::unordered_map<
  //   IntKey,
  //   std::shared_future<std::tuple<ScheduleResult, runtime::Module, runtime::PackedFunc, double, double> > > future_map;

  /* load the lib */
  std::unordered_set<std::string> unique_check;
  std::ifstream fin(reference);
  if (fin) {
    int count_line = 0;
    std::string line;
    while (std::getline(fin, line)) {
      std::vector<std::string> parts = string_split("|", line);
      ASSERT(parts.size() >= 4U) << "Bad line: " << line << ".\n";
      // IntKey key(std::stoi(parts[0]));
      if (unique_check.find(parts[0]) != unique_check.end()) {
        ERROR << "The library has repeated item with tag:\n" << parts[0] << "\n"
              << "at line " << count_line << "\n";
      }
      unique_check.insert(parts[0]);
      if (need_functions.find(parts[0]) != need_functions.end()) {
        auto future_sch_mod_func_perf_time = pool.push_back(schedule_and_build, need_functions[parts[0]], parts);
        target_subgraph_functions[parts[0]] = future_sch_mod_func_perf_time;
      }
      count_line += 1;
      // ScheduleResult schedule_result = auto_scheduler->schedule_with_entity(
      //   key, multi_graph.Self()->graphs[key], target, entity);

      // std::string name = get_func_name(key);
      
      // auto module = function_builder->build_func(
      //   schedule_result->schedule, schedule_result->tensors, target, Target::Create("llvm"),
      //   name, std::unordered_map<te::Tensor, tir::Buffer>(), tvm::BuildConfig::Create());

      // auto func = module->GetFunction(name);
      // auto future_sch_mod_func_perf_time = pool.push_back(schedule_and_build, key, parts);
      // auto sch_mod_func = schedule_and_build(key, entity);
      // auto schedule_result = std::get<0>(sch_mod_func);
      // auto module = std::get<1>(sch_mod_func);
      // auto func = std::get<2>(sch_mod_func);

      // built_functions[key].push(std::make_tuple(schedule_result, module, func));
      // best_functions[key].push(std::make_tuple(schedule_result, module, func, perf, time));
      // future_map[key] = future_sch_mod_func_perf_time;

      // TIRGraph subgraph = multi_graph.Self()->graphs[key];
      // if (cache.find(subgraph->tag) == cache.end()) {
      //   cache[subgraph->tag] = key;
      // }
    }
    fin.close();

    // for (auto& kv : future_map) {
    //   auto sch_mod_func_perf_time = kv.second.get();
    //   best_functions[kv.first].push(sch_mod_func_perf_time);
    // }
  } else {
    ERROR << "Can't open schedule reference file " << reference << ".\n";
  }

  // bool have_miss = false;
  // for (auto kv : multi_graph.Self()->graphs) {
  //   if ((best_functions.find(kv.first) == best_functions.end()) || (best_functions[kv.first].empty())) {
  //     if (cache.find(kv.second->tag) != cache.end()) {
  //       print(1) << "Can't find the function for subgraph " << kv.second->tag << "\n";
  //       have_miss = true;
  //     }
  //     else
  //       best_functions[kv.first].push(best_functions[cache[kv.second->tag]].front());
  //   }
  // }

  // cached_all_functions[task_id] = !have_miss;

  /* load the fuctions */
  bool has_miss = false;
  for (auto kv : multi_graph.Self()->graphs) {
    if (target_subgraph_functions.find(kv.second->tag) != target_subgraph_functions.end()) {
      auto sch_mod_func_perf_time = target_subgraph_functions[kv.second->tag].get();
      best_functions[kv.first].push(sch_mod_func_perf_time);
    } else {
      print(0) << "Can't find the function for subgraph with tag:\n" << kv.second->tag << "\n"
               << "Only can tune from scratch for it...\n"; 
      has_miss = true;
    }
  }
  cached_all_functions[task_id] = !has_miss;
}


void Session::begin_tuning(int task_id, int advance_number, std::string reference,
  int first_stage_number, double second_stage_topk_ratio) {
  ASSERT(task_cache.find(task_id) != task_cache.end()) << "No such task " << task_id << "\n";
  TIRMultiGraph multi_graph = task_cache[task_id];

  // begin
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish[task_id] = false;
  lock.unlock();

  autoschedule_log << "[time= " << current_time().count() << "] " << "New autoschedule task.\n"
                   << "######################################################################\n" << std::flush;
  build_log << "[time= " << current_time().count() << "] " << "New build task.\n"
            << "######################################################################\n" << std::flush;
  evaluate_log << "[time= " << current_time().count() << "] " << "New evaluate task.\n"
            << "######################################################################\n" << std::flush;
  exe_log << "[time= " << current_time().count() << "] " << "New execution task.\n"
          << "######################################################################\n" << std::flush;

  // load reference
  // touch all the keys
  for (auto& kv : multi_graph->graphs) {
    if (future_functions[kv.first].empty()) {
      // pass
    }
    if (built_functions[kv.first].empty()) {
      // pass
    }
    if (best_functions[kv.first].empty()) {
      // pass
    }
  }

  if (reference != "") {
    prepare_for_test(task_id, reference);
    // add feedback to schedule context
    for (auto& kv : best_functions) {
      auto key = kv.first;
      auto subgraph = multi_graph.Self()->graphs[key];
      if (!kv.second.empty()) {
        auto sch_mod_func_perf_time = kv.second.front();
        auto schedule_result = std::get<0>(sch_mod_func_perf_time);
        auto gflops = std::get<3>(sch_mod_func_perf_time);
        auto_scheduler->feedback_for(key, subgraph, target, schedule_result, gflops);
      }
    }
  }

  /*
   * launch the run_autoschedule thread
   */
  if (this->sch_threads.find(task_id) == this->sch_threads.end()) {
    this->sch_threads[task_id] = std::thread(
      [this](int id, TIRMultiGraph g) {
        run_autoschedule(id, g);
      }, task_id, multi_graph);
  }

  /*
   * launch the run_build thread
   */
  if (this->build_threads.find(task_id) == this->build_threads.end()) {
    this->build_threads[task_id] = std::thread(
      [this](int id, TIRMultiGraph g) {
        run_build(id, g);
      }, task_id, multi_graph);
  }

  /*
   * launch the run_evaluate thread
   */
  if (this->evaluate_threads.find(task_id) == this->evaluate_threads.end()) {
    this->evaluate_threads[task_id] = std::thread(
      [this](int id, TIRMultiGraph g, int b, int f, double s) {
        run_evaluate(id, g, b, f, s);
      }, task_id, multi_graph, advance_number, first_stage_number, second_stage_topk_ratio);
  }

  in_tuning[task_id] = true;
}


void Session::end_tuning(int task_id) {
  // wait until cached
  while (true) {
    if(cached_all_functions.find(task_id) != cached_all_functions.end() && cached_all_functions[task_id]) {
      break;
    }
  }
  // end
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish[task_id] = true;
  lock.unlock();

  in_tuning[task_id] = false;
  in_tuning.erase(task_id);

  /*
   * end the run_autoschedule thread
   */
  if (this->sch_threads.find(task_id) == this->sch_threads.end()) {
    this->sch_threads[task_id].join();
    this->sch_threads.erase(task_id);
  }

  /*
   * end the run_build thread
   */
  if (this->build_threads.find(task_id) == this->build_threads.end()) {
    this->build_threads[task_id].join();
    this->build_threads.erase(task_id);
  }

  /*
   * launch the run_evaluate thread
   */
  if (this->evaluate_threads.find(task_id) == this->evaluate_threads.end()) {
    this->evaluate_threads[task_id].join();
    this->evaluate_threads.erase(task_id);
  }
}


// int Session::run(TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
//   int task_id = add_task(graph);
//   if (!use_autoschedule && cached_all_functions) {
//     print(1) << "No autoschedule overhead, pure execution!\n";
//     exe_log << "[time= " << current_time().count() << "] " << "New execution task.\n";
//     clear_autoschedule_context();
//     run_functions(task_cache[task_id], bindings);
//   } else {
//     autoschedule_log << "[time= " << current_time().count() << "] " << "New autoschedule task.\n";
//     build_log << "[time= " << current_time().count() << "] " << "New build task.\n";
//     exe_log << "[time= " << current_time().count() << "] " << "New execution task.\n";
//     run(task_cache[task_id], bindings);
//   }
  
//   return task_id;
// }


void Session::run(
  int task_id,
  std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings,
  std::string save_to,
  int profile_level,
  bool no_actual_run) {
  ASSERT(task_cache.find(task_id) != task_cache.end()) << "Can't find the task: " << task_id << ".\n";
  if (cached_all_functions.find(task_id) == cached_all_functions.end() || !cached_all_functions[task_id]) {
    if (in_tuning.find(task_id) == in_tuning.end() || !in_tuning[task_id]) {
      ERROR << "Functions of task " << task_id << " are not ready, but the tuning is stopped!\n";
    }
  }

  int advance_number = (int)(bindings.size());
  print(1) << "Advancing " << advance_number << " iterations.\n";

  TIRMultiGraph multi_graph = task_cache[task_id];
  run_functions(task_id, multi_graph, bindings, save_to, profile_level, no_actual_run);
}


std::shared_ptr<Session> create_or_get_session(
  Target target, int dev_id, SessionOption sess_option, int& session_id, bool get_session, bool clear_session) {
  static std::unordered_map<int, std::shared_ptr<Session> > sessions;
  static int global_count = 0;
  if (get_session) {
    ASSERT(sessions.find(session_id) != sessions.end()) << "Can't find the session " << session_id << ".";
    if (clear_session) {
      sessions.erase(session_id);
      return nullptr;
    } else {
      return sessions[session_id];
    }
  } else {
    // create session
    sessions[global_count] = std::make_shared<Session>(target, dev_id, sess_option);
    session_id = global_count;  // record the session id
    global_count += 1;
    return sessions[global_count];
  }
}


int create_session(Target target, int dev_id, SessionOption sess_option) {
  int ret = -1;
  create_or_get_session(target, dev_id, sess_option, ret, false, false);
  ASSERT(ret >= 0) << "Invalid session id when creating session: " << ret << ".";
  return ret;
}


std::shared_ptr<Session> get_session(int session_id) {
  // pass dummy target info
  return create_or_get_session(target::llvm(), 0, SessionOption(0), session_id, true, false);
}


void delete_session(int session_id) {
  // pass dummy target info
  create_or_get_session(target::llvm(), 0, SessionOption(0), session_id, true, true);
}


// void disable_autoschedule(int session_id) {
//   auto sess = get_session(session_id);
//   sess->disable_autoschedule();
// }


// void enable_autoschedule(int session_id) {
//   auto sess = get_session(session_id);
//   sess->enable_autoschedule();
// }


void initialize_weights(
  int session_id, TIRGraph graph, std::vector<tvm::runtime::NDArray> bindings) {

  auto sess = get_session(session_id);
  sess->initialize_weights(graph, bindings);
}


int add_task(int session_id, TIRGraph graph) {
  auto sess = get_session(session_id);
  return sess->add_task(graph);
}


// int run_graph(
//   int session_id, TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  
//   auto sess = get_session(session_id);
//   return sess->run(graph, bindings);
// }


void run_task(
  int session_id, int task_id,
  std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings, std::string save_to,
  int profile_level,
  bool no_actual_run) {
  
  auto sess = get_session(session_id);
  sess->run(task_id, bindings, save_to, profile_level, no_actual_run);
}


TVM_REGISTER_GLOBAL("tg.create_session_option")
.set_body_typed([](
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
  std::string execution_log_file
) {
  SessionOption ret = SessionOption(
    report_profile,
    report_iteration,
    report_iteration_period,
    autoschedule_trial_ratio,
    autoschedule_topk,
    autoschedule_new_trial,
    autoschedule_policy,
    autoschedule_parallel,
    autoschedule_timeout,
    autoschedule_log_file,
    profile_parallel,
    profile_timeout,
    build_parallel,
    build_timeout,
    build_log_file,
    evaluate_log_file,
    execution_explore_probability,
    execution_parallel,
    execution_timeout,
    synchronize_subgraph,
    execution_log_file);
  return ret;
});


TVM_REGISTER_GLOBAL("tg.create_session")
.set_body_typed([](Target target, int dev_id, SessionOption sess_option){
  return create_session(target, dev_id, sess_option);
});


TVM_REGISTER_GLOBAL("tg.delete_session")
.set_body_typed([](int session_id){
  delete_session(session_id);
});


TVM_REGISTER_GLOBAL("tg.get_context_from_session")
.set_body_typed([](int session_id){
  auto sess = get_session(session_id);
  return sess->ctx;
});


TVM_REGISTER_GLOBAL("tg.get_data_from_session")
.set_body_typed([](int session_id, Array<te::Tensor> keys){
  auto sess = get_session(session_id);
  return sess->get_data(keys);
});


// TVM_REGISTER_GLOBAL("tg.disable_autoschedule")
// .set_body_typed([](int session_id){
//   disable_autoschedule(session_id);
// });


// TVM_REGISTER_GLOBAL("tg.enable_autoschedule")
// .set_body_typed([](int session_id){
//   enable_autoschedule(session_id);
// });


TVM_REGISTER_GLOBAL("tg.begin_tuning")
.set_body_typed([](int session_id, int task_id, int advance_number,
  std::string reference, int first_stage_number, double second_stage_topk_ratio){
  auto sess = get_session(session_id);
  sess->begin_tuning(task_id, advance_number, reference, first_stage_number, second_stage_topk_ratio);
});


TVM_REGISTER_GLOBAL("tg.end_tuning")
.set_body_typed([](int session_id, int task_id){
  auto sess = get_session(session_id);
  sess->end_tuning(task_id);
});


TVM_REGISTER_GLOBAL("tg.test_schedule_reference")
.set_body_typed([](int session_id, int task_id, std::string reference){
  auto sess = get_session(session_id);
  sess->prepare_for_test(task_id, reference);
});


TVM_REGISTER_GLOBAL("tg.initialize_weights")
.set_body_typed([](int session_id, TIRGraph graph, Array<tvm::runtime::NDArray> bindings){
  std::vector<tvm::runtime::NDArray> _bindings;
  for (auto v : bindings) {
    _bindings.push_back(v);
  }
  initialize_weights(session_id, graph, _bindings);
});


TVM_REGISTER_GLOBAL("tg.add_task")
.set_body_typed([](
  int session_id, TIRGraph graph){
  return add_task(session_id, graph);
});


// TVM_REGISTER_GLOBAL("tg.run_graph")
// .set_body_typed([](
//   int session_id, TIRGraph graph, Array<Map<te::Tensor, tvm::runtime::NDArray> > bindings){
//   std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > _bindings;
//   for (auto mp : bindings) {
//     std::unordered_map<te::Tensor, tvm::runtime::NDArray> tmp;
//     for (auto kv : mp) {
//       tmp[kv.first] = kv.second;
//     }
//     _bindings.push_back(tmp);
//   }
//   return run_graph(session_id, graph, _bindings);
// });


TVM_REGISTER_GLOBAL("tg.run_task")
.set_body_typed([](
  int session_id, int task_id,
  Array<Map<te::Tensor, tvm::runtime::NDArray> > bindings, std::string save_to, int profile_level, bool no_actual_run){
  std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > _bindings;
  for (auto mp : bindings) {
    std::unordered_map<te::Tensor, tvm::runtime::NDArray> tmp;
    for (auto kv : mp) {
      tmp[kv.first] = kv.second;
    }
    _bindings.push_back(tmp);
  }
  run_task(session_id, task_id, _bindings, save_to, profile_level, no_actual_run);
});


TVM_REGISTER_GLOBAL("tg.print_subgraphs")
.set_body_typed([](
  int session_id, int task_id){
  auto sess = get_session(session_id);
  ASSERT(sess->task_cache.find(task_id) != sess->task_cache.end());
  TIRMultiGraph multi_graph = sess->task_cache[task_id];
  std::map<std::string, TIRGraph> tag_to_graph;
  for (auto kv : multi_graph.Self()->graphs) {
    tag_to_graph[kv.second->tag] = kv.second;
  }
  // for (auto kv : tag_to_graph) {
  //   print(0) << "=======================================================\n";
  //   print(0) << "Tag:\n" << kv.first << "\n" << "Subgraph:\n";
  //   for (auto op : kv.second->operation_list) {
  //     print(0) << "-------------------------------------------------------\n";
  //     print(0) << op << "\n";
  //     if (op.as<ComputeOpNode>()) {
  //       print(0) << op.as<ComputeOpNode>()->body << "\n";
  //     }
  //   }
  // }
  std::map<std::string, TIRGraph> graphs;
  for (auto kv : multi_graph.Self()->graphs) {
    graphs[kv.second->tag] = kv.second;
  }
  for (auto kv : graphs) {
    print(0) << "Tag: " << kv.first << "\n";
    for (auto op : kv.second->operation_list) {
      print(0) << "-------------------------------------------------------\n";
      print(0) << op << "\n";
      if (op.as<ComputeOpNode>()) {
        print(0) << op.as<ComputeOpNode>()->body << "\n";
      }
    }
  }
});


}  // namespace tg


}  // namespace tvm