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
  exe_log.open(sess_option->execution_log_file, std::ios::app);
  std::string profile_log_name = string_split(".", sess_option->autoschedule_log_file)[0] + "_profile.txt";
  auto_scheduler = new AutoScheduler(ctx, sess_option->autoschedule_topk, sess_option->autoschedule_new_trial,
    sess_option->autoschedule_policy, sess_option->autoschedule_parallel, sess_option->profile_parallel,
    sess_option->autoschedule_timeout, sess_option->profile_timeout, sess_option->report_profile,
    autoschedule_log, profile_log_name);
  function_builder = new FunctionBuilder(
    sess_option->build_parallel, sess_option->build_timeout, build_log);
  thread_pool = new ThreadPool(sess_option->execution_parallel, (int)(sess_option->execution_timeout * 1000));

  task_count = 0;
  cached_all_functions = false;
  use_autoschedule = true;
}


Session::~Session() {
  autoschedule_log.close();
  build_log.close();
  exe_log.close();
  task_cache.clear();
  persistent_tensors.clear();
  volatile_tensors.clear();
  future_functions.clear();
  built_functions.clear();
  best_functions.clear();

  if (auto_scheduler != nullptr) {
    delete auto_scheduler;
  }

  if (function_builder != nullptr) {
    delete function_builder;
  }

  if (thread_pool != nullptr) {
    delete thread_pool;
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


void Session::run_autoschedule(TIRMultiGraph multi_graph, int advance_number) {
  // forward the compilation by multiple iterations
  int schedule_trials = std::ceil((advance_number * sess_option->autoschedule_trial_ratio));
  for (int ad = 0; ad < schedule_trials; ++ad) {
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
        peek_finish = this->finish;
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

        // at last, proceed
        // this check can be removed when the runtime is mature
        ASSERT(multi_graph.Self()->graphs.find(cand) != multi_graph.Self()->graphs.end())
          << "Can't find the subgraph " << cand->value << ".";
        TIRGraph subgraph = multi_graph.Self()->graphs[cand];

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
    peek_finish = this->finish;
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


void Session::run_build(TIRMultiGraph multi_graph, int advance_number) {
  // forward the compilation by multiple iterations
  int build_trials = std::ceil((advance_number * sess_option->autoschedule_trial_ratio));
  for (int ad = 0; ad < build_trials; ++ad) {
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
        peek_finish = this->finish;
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

        // at last, proceed
        /*
         * make a build
         */
        print(4, build_log) << "build for " << cand->value << "\n";
        if (!future_functions[cand].empty()) {
          auto sch_and_mod = future_functions[cand].front();
          ScheduleResult sch = sch_and_mod.first;
          auto future_mod = sch_and_mod.second;

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
    peek_finish = this->finish;
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


void Session::run_functions(
  TIRMultiGraph multi_graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {

  // prepare the call_unpack
  // const auto* call_unpack = runtime::Registry::Get("tg.runtime.call_unpack");
  // ASSERT(call_unpack != nullptr) << "Should prepare call_unpack function.";

  auto* call_unpack = new CallFunc<tvm::runtime::PackedFunc, tvm::runtime::NDArray>();

  int advance_number = (int)bindings.size();
  double total_execution_time = 0.0;
  double best_execution_time = 1e10;
  ProgressBar progress_bar;

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
      
      auto t0 = std::chrono::steady_clock::now();

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
      auto t1 = std::chrono::steady_clock::now();
      auto t1_t0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e3;
      print(4, exe_log) << "Array preparation uses " << t1_t0 << " ms.\n";
      
      /* loop until this subgraph is handled
       * or break if there is no way to execute this subgraph
       * in such case, emit an error in emergency queue
       * to tell the scheduler to re-schedule.
       */
      while (!succ) {
        bool taken = false;  // record taken a function

        double explore = randdouble();
        // print(4) << "explore random value: "
        //          << explore << " vs "
        //          << sess_option->execution_explore_probability
        //          << "\n";
        
        // first, try to get new function
        if ((explore < sess_option->execution_explore_probability) && !this->built_functions[key].empty()) {
          auto sch_mod_func = this->built_functions[key].front();
          // check if is ready
          auto schedule_result = std::get<0>(sch_mod_func);
          auto mod = std::get<1>(sch_mod_func);
          auto func = std::get<2>(sch_mod_func);
          ASSERT(func != nullptr) << "Get null function, don't know how to deal with it.";

          // take away this one
          this->built_functions[key].pop();
          taken = true;   // we will take one function       

          if (sess_option->synchronize_subgraph) {
            /* run the function in another thread
            */
            auto t5 = std::chrono::steady_clock::now();
            auto future = thread_pool->push_back(
              [&]() {
                auto start = std::chrono::steady_clock::now();
                (*call_unpack)(func, arrays);
                runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
                auto end = std::chrono::steady_clock::now();
                float elapsed_time = (float)(
                  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1e3;
                return elapsed_time;
              });

            auto t6 = std::chrono::steady_clock::now();
            auto t6_t5 = std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() / 1e3;
            print(4, exe_log) << "Push back execution task uses " << t6_t5 << " ms.\n";
            // run this function
            try {
              print(4, exe_log) << "Waiting for execution for " << key->value << "...\n";
              float elapsed_time = future.get();
              print(4, exe_log) << "Get execution for " << key->value << "!\n";
              print(4, exe_log) << "Execution uses " << elapsed_time << " ms.\n";
              auto t7 = std::chrono::steady_clock::now();
              // feedback
              float gflops = get_gflop(subgraph) / (elapsed_time / 1e3 + 1e-8);
              auto_scheduler->feedback_for(key, subgraph, schedule_result, gflops);
              auto t8 = std::chrono::steady_clock::now();
              auto t8_t7 = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count() / 1e3;
              print(4, exe_log) << "Feedback uses " << t8_t7 << " ms.\n";

              // store function
              if (best_functions.find(key) == best_functions.end()) {
                best_functions[key] = std::make_tuple(mod, func, gflops);
              } else {
                if (gflops > std::get<2>(best_functions[key])) {
                  best_functions[key] = std::make_tuple(mod, func, gflops);
                }
              }

              // success
              succ = true;
            } catch (const std::exception &e) {
              // can't run the function
              /*
              * should kill the thread
              * TODO: add kill
              */
              print(2, exe_log) << "Can't run function: " << e.what() << "\n";
            }  // end try run function
          } else {
            // run this function
            try {
              print(4, exe_log) << "Waiting for execution for " << key->value << "...\n";
              auto start_exe = std::chrono::steady_clock::now();
              (*call_unpack)(func, arrays);
              auto end_exe = std::chrono::steady_clock::now();
              /*
               * TODO: We need to add cuda events for timing for GPU
               * */
              float elapsed_time = (float)(
                std::chrono::duration_cast<std::chrono::microseconds>(end_exe - start_exe).count()) / 1e3;
              print(4, exe_log) << "Get execution for " << key->value << "!\n";
              print(4, exe_log) << "Execution uses " << elapsed_time << " ms.\n";
              auto t7 = std::chrono::steady_clock::now();
              // feedback
              float gflops = get_gflop(subgraph) / (elapsed_time / 1e3 + 1e-8);
              // auto_scheduler->feedback_for(key, subgraph, schedule_result, gflops);
              auto t8 = std::chrono::steady_clock::now();
              auto t8_t7 = std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count() / 1e3;
              print(4, exe_log) << "Feedback uses " << t8_t7 << " ms.\n";

              // store function
              if (best_functions.find(key) == best_functions.end()) {
                best_functions[key] = std::make_tuple(mod, func, gflops);
              } else {
                if (gflops > std::get<2>(best_functions[key])) {
                  best_functions[key] = std::make_tuple(mod, func, gflops);
                }
              }

              // success
              succ = true;
            } catch (const std::exception &e) {
              // can't run the function
              /*
              * should kill the thread
              * TODO: add kill
              */
              print(2, exe_log) << "Can't run function: " << e.what() << "\n";
            }  // end try run function
          }  // if sess_option->synchronize_subgraph
        }  // end try new function

        // then, try to use old function
        if (!succ) {
          if (best_functions.find(key) != best_functions.end()) {
            auto func = std::get<1>(best_functions[key]);
            auto run_beg = std::chrono::steady_clock::now();
            (*call_unpack)(func, arrays);
            if (sess_option->synchronize_subgraph) {
              runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
            } else {
              /*
               * TODO: add cuda events for timing for GPU
               */
            }
            auto run_end = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(run_end - run_beg).count() / 1e3;
            print(4, exe_log) << "Run cached function uses " << duration << " ms.\n";
            succ = true;
          }
        }  // end try old function

        // must check taken because chance is that
        // the scheduler is not ready
        if (!succ && this->built_functions[key].empty() && taken) {
          // there is no way to run this subgraph
          // report error
          this->emergency_schedule_queue.push(key);
          break;
        }

      }  // end while (1)

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

    auto beg = std::chrono::steady_clock::now();
    while (!free_set.empty()) {
      std::unordered_set<IntKey> update_set, delete_set;

      for (auto k : free_set) {
        auto start = std::chrono::steady_clock::now();
        run_helper(k, update_set, delete_set);
        auto stop = std::chrono::steady_clock::now();
        double run_helper_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1e3;
        print(4, exe_log) << "run helper uses " << run_helper_time << " ms.\n";
      }
      for (auto k : delete_set) {
        free_set.erase(k);
      }
      for (auto k : update_set) {
        free_set.insert(k);
      }
    }
    auto end = std::chrono::steady_clock::now();
    double execution_time = std::chrono::duration_cast<std::chrono::microseconds>(end - beg).count() / 1e3;
    if (execution_time < best_execution_time)
      best_execution_time = execution_time;
    if (sess_option->report_iteration && ((ad + 1) % sess_option->report_iteration_period == 0)) {
      exe_log << "Time cost: " << execution_time << " ms.\n";
    }
    if (ad > 0)
      total_execution_time += execution_time;
    cached_all_functions = true;
  }  // for ad

  if (advance_number > 1)
    print(1, exe_log) << "Average time cost for each iteration: " << total_execution_time / (advance_number-1) << " ms.\n";
  print(1, exe_log) << "Best iteration takes: " << best_execution_time << " ms.\n";

  // synchronize the stream for this run task
  runtime::DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
  // notify done
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish = true;
  lock.unlock();
}


int Session::add_task(TIRGraph graph) {
  SubGraphPartitionEngine partition_engine;
  TIRMultiGraph multi_graph(graph, partition_engine);
  int task_id = task_count++;
  task_cache[task_id] = multi_graph;
  return task_id;
}


void Session::run(TIRMultiGraph multi_graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  int advance_number = (int)(bindings.size());
  print(1) << "Advancing " << advance_number << " iterations.\n";

  // begin
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish = false;
  lock.unlock();

  // valide tensor index
  // this can be removed when the runtime is mature
  for (auto kv : multi_graph->graphs) {
    for (auto t : kv.second->weights) {
      ASSERT(multi_graph->tensor_index.find(t) != multi_graph->tensor_index.end()) << "Can't find " << t << ".";
    }
  }

  // allocate output/loss/gradients/updates buffer
  // the weight buffers should be initialized before
  allocate_output_buffer(multi_graph);

  /*
   * launch the run_autoschedule thread
   */
  std::thread sch_thread(
    [this](TIRMultiGraph g, int b) {
      run_autoschedule(g, b);
    }, multi_graph, advance_number);

  /*
   * launch the run_build thread
   */
  std::thread build_thread(
    [this](TIRMultiGraph g, int b) {
      run_build(g, b);
    }, multi_graph, advance_number);

  /*
   * launch the run_function thread
   */
  std::thread exe_thread(
    [this](TIRMultiGraph g, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > b) {
      run_functions(g, b);
    }, multi_graph, bindings);
  
  // wait until finished
  while (1) {
    // see if not done
    bool peek_finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    peek_finish = this->finish;
    lock.unlock();

    if (peek_finish) {
      break;
    }
  }  // while 1
  // wait for execution
  sch_thread.join();
  build_thread.join();
  exe_thread.join();
  print(1) << "end run.\n";
}


int Session::run(TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  int task_id = add_task(graph);
  if (!use_autoschedule && cached_all_functions) {
    print(1) << "No autoschedule overhead, pure execution!\n";
    exe_log << "[time= " << current_time().count() << "] " << "New execution task.\n";
    clear_autoschedule_context();
    run_functions(task_cache[task_id], bindings);
  } else {
    autoschedule_log << "[time= " << current_time().count() << "] " << "New autoschedule task.\n";
    build_log << "[time= " << current_time().count() << "] " << "New build task.\n";
    exe_log << "[time= " << current_time().count() << "] " << "New execution task.\n";
    run(task_cache[task_id], bindings);
  }
  
  return task_id;
}


void Session::run(int task_id, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  ASSERT(task_cache.find(task_id) != task_cache.end()) << "Can't find the task: " << task_id << ".\n";
  TIRMultiGraph multi_graph = task_cache[task_id];
  if (!use_autoschedule && cached_all_functions) {
    print(1) << "No autoschedule overhead, pure execution!\n";
    exe_log << "[time= " << current_time().count() << "] " << "New execution task.\n";
    clear_autoschedule_context();
    run_functions(multi_graph, bindings);
  } else {
    autoschedule_log << "[time= " << current_time().count() << "] " << "New autoschedule task.\n";
    build_log << "[time= " << current_time().count() << "] " << "New build task.\n";
    exe_log << "[time= " << current_time().count() << "] " << "New execution task.\n";
    run(multi_graph, bindings);
  }
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


void disable_autoschedule(int session_id) {
  auto sess = get_session(session_id);
  sess->disable_autoschedule();
}


void enable_autoschedule(int session_id) {
  auto sess = get_session(session_id);
  sess->enable_autoschedule();
}


void initialize_weights(
  int session_id, TIRGraph graph, std::vector<tvm::runtime::NDArray> bindings) {

  auto sess = get_session(session_id);
  sess->initialize_weights(graph, bindings);
}


int add_task(int session_id, TIRGraph graph) {
  auto sess = get_session(session_id);
  return sess->add_task(graph);
}


int run_graph(
  int session_id, TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  
  auto sess = get_session(session_id);
  return sess->run(graph, bindings);
}


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


TVM_REGISTER_GLOBAL("tg.disable_autoschedule")
.set_body_typed([](int session_id){
  disable_autoschedule(session_id);
});


TVM_REGISTER_GLOBAL("tg.enable_autoschedule")
.set_body_typed([](int session_id){
  enable_autoschedule(session_id);
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


TVM_REGISTER_GLOBAL("tg.run_graph")
.set_body_typed([](
  int session_id, TIRGraph graph, Array<Map<te::Tensor, tvm::runtime::NDArray> > bindings){
  std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > _bindings;
  for (auto mp : bindings) {
    std::unordered_map<te::Tensor, tvm::runtime::NDArray> tmp;
    for (auto kv : mp) {
      tmp[kv.first] = kv.second;
    }
    _bindings.push_back(tmp);
  }
  return run_graph(session_id, graph, _bindings);
});


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