#include "driver.h"
#include <unistd.h>
#include <cmath>


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
  
  int i = 0;
  for (auto t : graph->gradients) {
    ASSERT(persistent_tensors.find(graph->weights[i]) != persistent_tensors.end())
    << "Should initialize for weight " << graph->weights[i];
    std::vector<int64_t> shape;
    for (auto p : t->shape) {
      shape.push_back(get_const_int(p));
    }
    // for gradients
    persistent_tensors[t] = tvm::runtime::NDArray::Empty(shape, t->dtype, ctx);
    // share buffer with weight
    persistent_tensors[graph->updates[i]] = persistent_tensors[graph->weights[i]];
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


std::string Session::get_func_name(IntKey key) {
  return "subgraph_" + std::to_string(key->value);
}


void Session::run_autoschedule(int task_id, TIRMultiGraph multi_graph, int advance_number) {
  // forward the compilation by multiple iterations
  int schedule_trials = advance_number;  // std::ceil((advance_number * sess_option->autoschedule_trial_ratio));
  for (int ad = 0; ad < schedule_trials; ++ad) {
    // initialize cache
    std::unordered_set<std::string> scheduled;
    // initialize call order
    std::unordered_map<IntKey, int> schedule_order;
    std::unordered_set<IntKey> free_set;
    for (auto kv : multi_graph->graph_attrs) {
      schedule_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.first);
      }
    }

    // schedule and build for subgraphs
    int schedule_count = 0;
    int num_subgraphs = (int)multi_graph->graphs.size();
    while (!free_set.empty()) {
      std::unordered_set<IntKey> update_set;
      std::unordered_set<IntKey> delete_set;

      for (auto cand : free_set) {
        // first, check if there is need to add a new schedule
        // see if not finished
        bool peek_finish = false;
        std::unique_lock<std::mutex> lock(this->finish_mutex);
        peek_finish = this->finish[task_id];
        lock.unlock();
        if (peek_finish) {
          // execution done, no need to schedule
          return;
        }

        // then, check emergency queue
        if (!this->emergency_schedule_queue.empty()) {
          auto& key = this->emergency_schedule_queue.front();

          // the following are repeated
          // TODO: isolate the logic
          // handle emergency
          // get future schedule
          std::shared_future<ScheduleResult> schedule_result = auto_scheduler->schedule_for(
            key, multi_graph.Self()->graphs[key], target, 1);  // priority 1

          try {
            print(4, autoschedule_log) << "Waiting for emergency schedule for " << key->value << "...\n";
            ScheduleResult result = schedule_result.get();
            this->emergency_schedule_queue.pop();
            print(4, autoschedule_log) << "Get emergency schedule for " << key->value << "!\n";
            
            // get future func
            std::pair<ScheduleResult, std::shared_future<tvm::runtime::Module> > sch_func = \
            function_builder->build_for(
              result,
              target,
              Target::Create("llvm"),
              get_func_name(key),
              std::unordered_map<te::Tensor, tir::Buffer>(),
              tvm::BuildConfig::Create(),
              1  // priority 1
            );

            future_functions[key].push(sch_func);
            this->emergency_build_queue.push(key);
          } catch (const std::exception& e) {
            print(2, autoschedule_log) << "Can't get schedule for emergency: " << e.what() << "\n";
          }
        }  // if (!this->emergency_queue.empty())

        // then, check if need to schedule for it
        ASSERT(multi_graph.Self()->graphs.find(cand) != multi_graph.Self()->graphs.end())
          << "Can't find the subgraph " << cand->value << ".";
        TIRGraph subgraph = multi_graph.Self()->graphs[cand];

        // no need to re-schedule the same subgraph
        if (scheduled.find(subgraph->tag) != scheduled.end()) {
          print(4, autoschedule_log) << "Find repteated function " << subgraph->tag << ".\n";
          // update delete_set
          delete_set.insert(cand);

          // this check can be removed when the runtime is mature
          ASSERT(multi_graph.Self()->graph_attrs.find(cand) != multi_graph.Self()->graph_attrs.end())
            << "Can't find subgraph " << cand->value << "'s attributes.";
          for (auto succ : multi_graph.Self()->graph_attrs[cand]->successors) {
            schedule_order[succ] -= 1;
            if (schedule_order[succ] == 0) {
              update_set.insert(succ);
            }
          }
          
          // this subgraph is done
          schedule_count += 1;
          continue;
        }

        // at last, proceed
        // this check can be removed when the runtime is mature
        /*
         * make a schedule
         */
        print(4, autoschedule_log) << "schedule for " << cand->value << "\n";
        std::shared_future<ScheduleResult> schedule_result = auto_scheduler->schedule_for(
          cand, subgraph, target, 0);

        try {
          print(4, autoschedule_log) << "Waiting for schedule for " << cand->value << "...\n";
          ScheduleResult result = schedule_result.get();
          print(4, autoschedule_log) << "Get schedule for " << cand->value << "!\n";
          
          // get future func
          std::pair<ScheduleResult, std::shared_future<tvm::runtime::Module> > sch_func = \
          function_builder->build_for(
            result,
            target,
            Target::Create("llvm"),
            get_func_name(cand),
            std::unordered_map<te::Tensor, tir::Buffer>(),
            tvm::BuildConfig::Create()
          );

          future_functions[cand].push(sch_func);

          // update delete_set
          delete_set.insert(cand);

          // this check can be removed when the runtime is mature
          ASSERT(multi_graph.Self()->graph_attrs.find(cand) != multi_graph.Self()->graph_attrs.end())
            << "Can't find subgraph " << cand->value << "'s attributes.";
          for (auto succ : multi_graph.Self()->graph_attrs[cand]->successors) {
            schedule_order[succ] -= 1;
            if (schedule_order[succ] == 0) {
              update_set.insert(succ);
            }
          }
          
          // this subgraph is done
          schedule_count += 1;
          scheduled.insert(subgraph->tag);
        } catch (const std::exception& e) {
          /*
           * should tell the thread to stop
           * TODO: stop the thread
           */
          print(2, autoschedule_log) << "Can't get schedule: " << e.what() << "\n";
          continue;
        }
      }  // for cand

      for (auto deleted : delete_set) {
        free_set.erase(deleted);
      }
      for (auto new_cand : update_set) {
        free_set.insert(new_cand);
      }
    }  // end while (!free_set.empty())
    
    // make sure that every subgraph is handled
    // double check
    // this can be removed when the runtime is mature
    if (schedule_count != num_subgraphs) {
      throw std::runtime_error(
        "Schedule graph number mismatch "
        + std::to_string(schedule_count)
        + " vs. " + std::to_string(num_subgraphs));
    }

    // print(4) << "before auto_scheduler reset...\n";
    // auto_scheduler->reset();
    // print(4) << "after auto_scheduler reset!\n";
  }  // for ad

  // wait until finished
  while (1) {
    // see if not done
    bool peek_finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    peek_finish = this->finish[task_id];
    lock.unlock();
    if (!peek_finish) {
      if (!this->emergency_schedule_queue.empty()) {
        auto& key = this->emergency_schedule_queue.front();

        // the following are repeated
        // TODO: isolate the logic
        // handle emergency
        // get future schedule
        std::shared_future<ScheduleResult> schedule_result = auto_scheduler->schedule_for(
          key, multi_graph.Self()->graphs[key], target, 1);  // priority 1

        try {
          print(4, autoschedule_log) << "Waiting for emergency schedule for " << key->value << "...\n";
          ScheduleResult result = schedule_result.get();
          this->emergency_schedule_queue.pop();
          print(4, autoschedule_log) << "Get emergency schedule for " << key->value << "!\n";
          
          // get future func
          std::pair<ScheduleResult, std::shared_future<tvm::runtime::Module> > sch_func = \
          function_builder->build_for(
            result,
            target,
            Target::Create("llvm"),
            get_func_name(key),
            std::unordered_map<te::Tensor, tir::Buffer>(),
            tvm::BuildConfig::Create(),
            1  // priority 1
          );

          future_functions[key].push(sch_func);
          this->emergency_build_queue.push(key);
        } catch (const std::exception& e) {
          print(2, autoschedule_log) << "Can't get schedule for emergency: " << e.what() << "\n";
          continue;
        }
      }
    } else {
      break;
    }
  }  // while 1
}


void Session::run_build(int task_id, TIRMultiGraph multi_graph, int advance_number) {
  // forward the compilation by multiple iterations
  int build_trials = advance_number;  // std::ceil((advance_number * sess_option->autoschedule_trial_ratio));
  for (int ad = 0; ad < build_trials; ++ad) {
    // initialize cache
    std::unordered_set<std::string> built;
    // initialize call order
    std::unordered_map<IntKey, int> build_order;
    std::unordered_set<IntKey> free_set;
    for (auto kv : multi_graph->graph_attrs) {
      build_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.first);
      }
    }

    // build for subgraphs
    int build_count = 0;
    int num_subgraphs = (int)multi_graph->graphs.size();
    while (!free_set.empty()) {
      std::unordered_set<IntKey> update_set;
      std::unordered_set<IntKey> delete_set;

      for (auto cand : free_set) {
        // first, check if there is need to add a new schedule
        // see if not finished
        bool peek_finish = false;
        std::unique_lock<std::mutex> lock(this->finish_mutex);
        peek_finish = this->finish[task_id];
        lock.unlock();
        if (peek_finish) {
          // execution done, no need to schedule
          return;
        }

        // then, check emergency queue
        if (!this->emergency_build_queue.empty()) {
          auto& key = this->emergency_build_queue.front();

          // the following are repeated
          // TODO: isolate the logic
          // handle emergency
          // get future schedule
          if (!future_functions[key].empty()) {
            auto sch_and_mod = future_functions[key].front();
            ScheduleResult sch = sch_and_mod.first;
            auto future_mod = sch_and_mod.second;

            try {
              print(4, build_log) << "Waiting for emergency build for " << key->value << "...\n";
              tvm::runtime::Module mod = future_mod.get();
              this->emergency_build_queue.pop();
              tvm::runtime::PackedFunc func = mod->GetFunction(get_func_name(key));
              print(4, build_log) << "Get emergency build for " << key->value << "!\n";

              built_functions[key].push(std::make_tuple(sch, mod, func));
            } catch (const std::exception &e) {
              print(2, build_log) << "Can't get build for emergency: " << e.what() << "\n";
            }
          }
        }  // if (!this->emergency_queue.empty())

        // then, check if need to build
        TIRGraph subgraph = multi_graph.Self()->graphs[cand];
        if (built.find(subgraph->tag) != built.end()) {
          print(4, build_log) << "Find repeated function " << subgraph->tag << ".\n";
          // update delete_set
          delete_set.insert(cand);

          // this check can be removed when the runtime is mature
          ASSERT(multi_graph.Self()->graph_attrs.find(cand) != multi_graph.Self()->graph_attrs.end())
            << "Can't find subgraph " << cand->value << "'s attributes.";
          for (auto succ : multi_graph.Self()->graph_attrs[cand]->successors) {
            build_order[succ] -= 1;
            if (build_order[succ] == 0) {
              update_set.insert(succ);
            }
          }
          build_count += 1;
          continue;
        }

        // at last, proceed
        /*
         * make a build
         */
        print(4, build_log) << "build for " << cand->value << "\n";
        if (!future_functions[cand].empty()) {
          auto sch_and_mod = future_functions[cand].front();
          ScheduleResult sch = sch_and_mod.first;
          auto future_mod = sch_and_mod.second;
          future_functions[cand].pop();

          try {
            print(4, build_log) << "Waiting for build for " << cand->value << "...\n";
            tvm::runtime::Module mod = future_mod.get();
            tvm::runtime::PackedFunc func = mod->GetFunction(get_func_name(cand));
            print(4, build_log) << "Get build for " << cand->value << "!\n";

            built_functions[cand].push(std::make_tuple(sch, mod, func));

            // update delete_set
            delete_set.insert(cand);

            // this check can be removed when the runtime is mature
            ASSERT(multi_graph.Self()->graph_attrs.find(cand) != multi_graph.Self()->graph_attrs.end())
              << "Can't find subgraph " << cand->value << "'s attributes.";
            for (auto succ : multi_graph.Self()->graph_attrs[cand]->successors) {
              build_order[succ] -= 1;
              if (build_order[succ] == 0) {
                update_set.insert(succ);
              }
            }
            build_count += 1;
            built.insert(subgraph->tag);
          } catch (const std::exception &e) {
            print(2, build_log) << "Can't get build for: " << e.what() << "\n";
          }
        }

      }  // for cand

      for (auto deleted : delete_set) {
        free_set.erase(deleted);
      }
      for (auto new_cand : update_set) {
        free_set.insert(new_cand);
      }
    }  // end while (!free_set.empty())
    
    // make sure that every subgraph is handled
    // double check
    // this can be removed when the runtime is mature
    if (build_count != num_subgraphs) {
      throw std::runtime_error(
        "Build graph number mismatch "
        + std::to_string(build_count)
        + " vs. " + std::to_string(num_subgraphs));
    }

  }  // for ad

  // wait until finished
  while (1) {
    // see if not done
    bool peek_finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    peek_finish = this->finish[task_id];
    lock.unlock();
    if (!peek_finish) {
      if (!this->emergency_build_queue.empty()) {
        auto& key = this->emergency_build_queue.front();

        // the following are repeated
        // TODO: isolate the logic
        // handle emergency
        // get future schedule
        if (!future_functions[key].empty()) {
          auto sch_and_mod = future_functions[key].front();
          ScheduleResult sch = sch_and_mod.first;
          auto future_mod = sch_and_mod.second;

          try {
            print(4, build_log) << "Waiting for emergency build for " << key->value << "...\n";
            tvm::runtime::Module mod = future_mod.get();
            this->emergency_build_queue.pop();
            tvm::runtime::PackedFunc func = mod->GetFunction(get_func_name(key));
            print(4, build_log) << "Get emergency build for " << key->value << "!\n";

            built_functions[key].push(std::make_tuple(sch, mod, func));
          } catch (const std::exception &e) {
            print(2, build_log) << "Can't get build for emergency: " << e.what() << "\n";
          }
        }
      }  // if (!this->emergency_build_queue.empty())
    } else {
      break;
    }
  }  // while 1
}


void Session::run_evaluate(
  int task_id, TIRMultiGraph multi_graph, int advance_number) {

  // prepare the evaluate_performance
  const auto* evaluate_performance = runtime::Registry::Get("tg.runtime.evaluate_performance");
  ASSERT(evaluate_performance != nullptr) << "Should prepare tg.runtime.evaluate_performance function.";

  while (true) {
    // check if finish
    bool finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    finish = this->finish[task_id];
    lock.unlock();
    if (finish) {
      break;
    }
    // initialize cache
    std::unordered_map<std::string, IntKey> evaluate_cache;

    std::unordered_map<IntKey, int> evaluate_order;
    std::unordered_set<IntKey> free_set;
    for (auto kv : multi_graph->graph_attrs) {
      evaluate_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.first);
      }
    }

    /* the evaluate helper
     * handles one subgraph at a time
     * fill the delete set and update set
     */
    std::function<void(
      IntKey key,
      std::unordered_set<IntKey>& update_set,
      std::unordered_set<IntKey>& delete_set)> evaluate_helper;
    evaluate_helper = [this, &multi_graph, &evaluate_cache, &evaluate_order, &evaluate_performance]
      (IntKey key,
      std::unordered_set<IntKey>& update_set,
      std::unordered_set<IntKey>& delete_set) {

      // the mark that indicates this subgraph is done
      bool succ = false;

      TIRGraph subgraph = multi_graph.Self()->graphs[key];
      
      bool taken = false;  // record taken a function
      
      // first, try to get new function
      if (!succ && !this->built_functions[key].empty()) {
        auto sch_mod_func = this->built_functions[key].front();
        // take away this one
        this->built_functions[key].pop();
        taken = true;   // we will take one function
        // check if is ready
        auto schedule_result = std::get<0>(sch_mod_func);
        auto mod = std::get<1>(sch_mod_func);
        auto func = std::get<2>(sch_mod_func);
        ASSERT(func != nullptr) << "Get null function, don't know how to deal with it.";

        /* 
          * run the function in another process
          * to get performance, if return -1
          * then timeout or fail in execution
          */
        double elapsed_time = (*evaluate_performance)(mod, get_func_name(key), schedule_result->tensors);
        print(4, evaluate_log) << "evaluate result for " << key->value << " is " << elapsed_time << "ms.\n";

        if (elapsed_time > 0) {
          // feedback
          double gflops = get_gflop(subgraph) / (elapsed_time / 1e3 + 1e-8);
          auto_scheduler->feedback_for(key, subgraph, schedule_result, gflops);

          // store function
          if (best_functions[key].empty()) {
            best_functions[key].push(std::make_tuple(mod, func, gflops));
          } else {
            auto best = best_functions[key].front();
            if (gflops > std::get<2>(best)) {
              best_functions[key].push(std::make_tuple(mod, func, gflops));
              best_functions[key].pop();
            }
          }

          // success
          succ = true;
          evaluate_cache[subgraph->tag] = key;
        } else {
          // can't run the function
          print(2, evaluate_log) << "Can't evaluate function: " << "\n";
          auto sub_mods = mod->imports();
          if (sub_mods.size() > 0U) {
            runtime::Module sub_mod = (mod->imports().at(0));
            print(4, evaluate_log) << "Check source:\n" << sub_mod->GetSource() << "\n";
          }
          // feedback
          auto_scheduler->feedback_for(key, subgraph, schedule_result, 0.0);
        }  // end try run function
      }  // end try new function

      // then, try to find repeated subgraph
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

      // must check taken because chance is that
      // the scheduler is not ready
      if (!succ && this->best_functions[key].empty() && taken) {
        // there is no way to run this subgraph
        // report error
        this->emergency_schedule_queue.push(key);
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

    while (!free_set.empty()) {
      // check if finish
      bool finish = false;
      std::unique_lock<std::mutex> lock(this->finish_mutex);
      finish = this->finish[task_id];
      lock.unlock();
      if (finish) {
        break;
      }
      std::unordered_set<IntKey> update_set, delete_set;

      for (auto k : free_set) {
        evaluate_helper(k, update_set, delete_set);
      }
      for (auto k : delete_set) {
        free_set.erase(k);
      }
      for (auto k : update_set) {
        free_set.insert(k);
      }
    }  // while (!free_set.empty())
    cached_all_functions[task_id] = true;
  }  // while (true)
}


void Session::run_functions(
  TIRMultiGraph multi_graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {

  auto* call_unpack = new CallFunc<tvm::runtime::PackedFunc, tvm::runtime::NDArray>();

  int advance_number = (int)bindings.size();
  ProgressBar progress_bar;

  auto beg = std::chrono::steady_clock::now();
  for (int ad = 0; ad < advance_number; ++ad) {
    if (sess_option->report_iteration) {
      exe_log << "Iteration: " << ad << "\n";
    }
    progress_bar.draw(((double)(ad + 1) / advance_number));
    if (ad == advance_number - 1) {
      progress_bar.end();
    }
    std::unordered_map<IntKey, int> call_order;
    std::unordered_set<IntKey> free_set;
    for (auto kv : multi_graph->graph_attrs) {
      call_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.first);
      }
    }

    /* the run helper
     * handles one subgraph at a time
     * fill the delete set and update set
     */
    std::function<void(
      IntKey key,
      std::unordered_set<IntKey>& update_set,
      std::unordered_set<IntKey>& delete_set)> run_helper;
    run_helper = [this, ad, &multi_graph, &call_order, &bindings, &call_unpack]
      (IntKey key,
      std::unordered_set<IntKey>& update_set,
      std::unordered_set<IntKey>& delete_set) {

      // the mark that indicates this subgraph is done
      bool succ = false;

      TIRGraph subgraph = multi_graph.Self()->graphs[key];

      /* get the runtime array
       * the order is the same as build
       * TODO: handle the order by some other
       * independent logic
       */
      std::vector<tvm::runtime::NDArray> arrays;
      // get the inputs
      for (auto tt : subgraph->inputs) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (bindings[ad].find(t) != bindings[ad].end()) {
          arrays.push_back(bindings[ad][t]);
        } else if (this->volatile_tensors.find(t) != this->volatile_tensors.end()) {
          arrays.push_back(this->volatile_tensors[t]);
        } else {
          ERROR << "Can't find input " << t;
        }
      }

      // get the labels
      for (auto tt : subgraph->labels) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (bindings[ad].find(t) == bindings[ad].end()) {
          ERROR << "Can't find label " << t;
        }
        arrays.push_back(bindings[ad][t]);
      }

      // get the outputs
      for (auto tt : subgraph->outputs) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (volatile_tensors.find(t) == volatile_tensors.end()) {
          ERROR << "Can't find output " << t;
        }
        arrays.push_back(this->volatile_tensors[t]);
      }

      // get the weights
      for (auto tt : subgraph->weights) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (persistent_tensors.find(t) == persistent_tensors.end()) {
          ERROR << "Can't find weight " << t;
        }
        arrays.push_back(this->persistent_tensors[t]);
      }

      // get the loss
      if (subgraph->loss.defined()) {
        te::Tensor t = multi_graph.Self()->tensor_index[subgraph->loss];
        if (persistent_tensors.find(t) == persistent_tensors.end()) {
          ERROR << "Can't find loss " << t;
        }
        arrays.push_back(this->persistent_tensors[t]);
      }
      
      // get the gradients
      for (auto tt : subgraph->gradients) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (persistent_tensors.find(t) == persistent_tensors.end()) {
          ERROR << "Can't find gradient " << t;
        }
        arrays.push_back(this->persistent_tensors[t]);
      }
      
      // get the lr
      if (subgraph->lr.defined()) {
        te::Tensor t = multi_graph.Self()->tensor_index[subgraph->lr];
        if (bindings[ad].find(t) == bindings[ad].end()) {
          ERROR << "Can't find lr " << t;
        }
        arrays.push_back(bindings[ad][t]);
      }
      
      // get the updates
      for (auto tt : subgraph->updates) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (persistent_tensors.find(t) == persistent_tensors.end()) {
          ERROR << "Can't find update " << t;
        }
        arrays.push_back(this->persistent_tensors[t]);
      }
      
      /* loop until this subgraph is handled
       * or break if there is no way to execute this subgraph
       * in such case, emit an error in emergency queue
       * to tell the scheduler to re-schedule.
       */  
      if (!succ && !this->best_functions[key].empty()) {
        auto mod_func = this->best_functions[key].front();

        auto mod = std::get<0>(mod_func);
        auto func = std::get<1>(mod_func);
        ASSERT(func != nullptr) << "Get null function, don't know how to deal with it.";

        (*call_unpack)(func, arrays);

        // success
        succ = true;
      }  // end try new function

      if (succ) {
        // update free set
        delete_set.insert(key);
        for (auto v : multi_graph.Self()->graph_attrs[key]->successors) {
          call_order[v] -= 1;
          if (call_order[v] == 0) {
            update_set.insert(v);
          }
        }
      }

    };  // end run helper

    while (!free_set.empty()) {
      std::unordered_set<IntKey> update_set, delete_set;

      for (auto k : free_set) {
        run_helper(k, update_set, delete_set);
      }
      for (auto k : delete_set) {
        free_set.erase(k);
      }
      for (auto k : update_set) {
        free_set.insert(k);
      }
    }

    
    if (ad == 0)  // skip the first iteration
      beg = std::chrono::steady_clock::now();
  }  // for ad
  auto end = std::chrono::steady_clock::now();
  double execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() / 1e3;

  if (advance_number > 1)
    print(1, exe_log) << "Average time cost for each iteration: " << execution_time / (advance_number-1) << " ms.\n";

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
  return task_id;
}


void Session::begin_tuning(int task_id, int advance_number) {
  ASSERT(task_cache.find(task_id) != task_cache.end()) << "No such task " << task_id << "\n";
  TIRMultiGraph multi_graph = task_cache[task_id];

  // begin
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish[task_id] = false;
  lock.unlock();

  autoschedule_log << "[time= " << current_time().count() << "] " << "New autoschedule task.\n"
                   << "######################################################################\n";
  build_log << "[time= " << current_time().count() << "] " << "New build task.\n"
            << "######################################################################\n";
  exe_log << "[time= " << current_time().count() << "] " << "New execution task.\n"
          << "######################################################################\n";

  /*
   * launch the run_autoschedule thread
   */
  if (this->sch_threads.find(task_id) == this->sch_threads.end()) {
    this->sch_threads[task_id] = std::thread(
      [this](int id, TIRMultiGraph g, int b) {
        run_autoschedule(id, g, b);
      }, task_id, multi_graph, advance_number);
  }

  /*
   * launch the run_build thread
   */
  if (this->build_threads.find(task_id) == this->build_threads.end()) {
    this->build_threads[task_id] = std::thread(
      [this](int id, TIRMultiGraph g, int b) {
        run_build(id, g, b);
      }, task_id, multi_graph, advance_number);
  }

  /*
   * launch the run_evaluate thread
   */
  if (this->evaluate_threads.find(task_id) == this->evaluate_threads.end()) {
    this->evaluate_threads[task_id] = std::thread(
      [this](int id, TIRMultiGraph g, int b) {
        run_evaluate(id, g, b);
      }, task_id, multi_graph, advance_number);
  }

  in_tuning[task_id] = true;
}


void Session::end_tuning(int task_id) {
  // wait until cached
  while(cached_all_functions.find(task_id) == cached_all_functions.end() || !cached_all_functions[task_id]) {

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


void Session::run(int task_id, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  ASSERT(task_cache.find(task_id) != task_cache.end()) << "Can't find the task: " << task_id << ".\n";
  if (cached_all_functions.find(task_id) == cached_all_functions.end() || !cached_all_functions[task_id]) {
    if (in_tuning.find(task_id) == in_tuning.end() || !in_tuning[task_id]) {
      ERROR << "Functions of task " << task_id << " are not ready, but the tuning is stopped!\n";
    }
  }
  int advance_number = (int)(bindings.size());
  print(1) << "Advancing " << advance_number << " iterations.\n";

  TIRMultiGraph multi_graph = task_cache[task_id];
  run_functions(multi_graph, bindings);
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
  int session_id, int task_id, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  
  auto sess = get_session(session_id);
  sess->run(task_id, bindings);
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


// TVM_REGISTER_GLOBAL("tg.disable_autoschedule")
// .set_body_typed([](int session_id){
//   disable_autoschedule(session_id);
// });


// TVM_REGISTER_GLOBAL("tg.enable_autoschedule")
// .set_body_typed([](int session_id){
//   enable_autoschedule(session_id);
// });


TVM_REGISTER_GLOBAL("tg.begin_tuning")
.set_body_typed([](int session_id, int task_id, int advance_number){
  auto sess = get_session(session_id);
  sess->begin_tuning(task_id, advance_number);
});


TVM_REGISTER_GLOBAL("tg.end_tuning")
.set_body_typed([](int session_id, int task_id){
  auto sess = get_session(session_id);
  sess->end_tuning(task_id);
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
  int session_id, int task_id, Array<Map<te::Tensor, tvm::runtime::NDArray> > bindings){
  std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > _bindings;
  for (auto mp : bindings) {
    std::unordered_map<te::Tensor, tvm::runtime::NDArray> tmp;
    for (auto kv : mp) {
      tmp[kv.first] = kv.second;
    }
    _bindings.push_back(tmp);
  }
  run_task(session_id, task_id, _bindings);
});


}  // namespace tg


}  // namespace tvm