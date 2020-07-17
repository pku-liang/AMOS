#include "driver.h"
#include <unistd.h>


#include "../graph/concrete_graph.h"


namespace tvm {


namespace tg {

TVM_REGISTER_NODE_TYPE(SessionOptionNode);

SessionOption::SessionOption(
  bool report_profile,
  bool report_iteration,
  int report_iteration_period,
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
  int execution_parallel,
  double execution_timeout) {
  auto node = make_object<SessionOptionNode>();
  node->report_profile = report_profile;
  node->report_iteration = report_iteration;
  node->report_iteration_period = report_iteration_period;
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
  node->execution_parallel = execution_parallel;
  node->execution_timeout = execution_timeout;
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

  auto_scheduler = new AutoScheduler(ctx, sess_option->autoschedule_topk, sess_option->autoschedule_new_trial,
    sess_option->autoschedule_policy, sess_option->autoschedule_parallel, sess_option->profile_parallel,
    sess_option->autoschedule_timeout, sess_option->profile_timeout, sess_option->autoschedule_log_file);
  function_builder = new FunctionBuilder(sess_option->build_parallel, sess_option->build_timeout);
  thread_pool = new ThreadPool(sess_option->execution_parallel, (int)(sess_option->execution_timeout * 1000));
}


Session::~Session() {
  persistent_tensors.clear();
  volatile_tensors.clear();
  functions.clear();
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


void Session::run_functions(
  TIRMultiGraph multi_graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {

  // prepare the call_unpack
  const auto* call_unpack = runtime::Registry::Get("tg.runtime.call_unpack");
  ASSERT(call_unpack != nullptr) << "Should prepare call_unpack function.";

  int advance_number = (int)bindings.size();
  double total_execution_time = 0.0;
  double best_execution_time = 1e10;

  for (int ad = 0; ad < advance_number; ++ad) {
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
    run_helper = [this, ad, &multi_graph, &call_order, &bindings, call_unpack]
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
      Array<tvm::runtime::NDArray> arrays;
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
          throw;
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
      while (!succ) {
        bool taken = false;  // record taken a function
        
        // first, try to get new function
        if (!this->functions[key].empty()) {
          auto sch_func = this->functions[key].front();
          // check if is ready
          auto schedule_result = sch_func.first;
          auto mod = sch_func.second;
          auto status = mod.wait_for(std::chrono::milliseconds(0));
          if (status == std::future_status::ready) {
            // std::cout << "take new function for " << key->value << "\n";
            // take away this one
            this->functions[key].pop();
            taken = true;   // we will take one function
   
            try {
              tvm::runtime::Module module = mod.get();
              auto mod_func = module->GetFunction(get_func_name(key));
              ASSERT(mod_func != nullptr) << "Get null function, don't know how to deal with it.";

              /* run the function in another thread 
              * TODO: wait this thread to the end
              */
              auto future = thread_pool->push_back(
                [call_unpack](tvm::runtime::PackedFunc& f, Array<tvm::runtime::NDArray>& v) {
                  auto start = std::chrono::steady_clock::now();
                  (*call_unpack)(f, v);
                  auto end = std::chrono::steady_clock::now();
                  float elapsed_time = (float)(
                    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) / 1e3;
                  return elapsed_time;
                }, mod_func, arrays);

              // run this function
              try {
                float elapsed_time = future.get();
                // std::cout << "elapsed time: " << elapsed_time << " ms\n";
                // feedback
                float gflops = get_gflop(subgraph) / (elapsed_time / 1e3 + 1e-8);
                auto_scheduler->feedback_for(key, subgraph, schedule_result, gflops);
                /*
                * feedback is added here
                * TODO: do feedback here
                */
                // sch_func.first.Self()->leaf->update_reward(sch_func.first->configs, gflops);
                // store function
                if (best_functions.find(key) == best_functions.end()) {
                  best_functions[key] = std::make_pair(module, gflops);
                } else {
                  if (gflops > best_functions[key].second) {
                    best_functions[key] = std::make_pair(module, gflops);
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
               print(2) << "Can't run function: " << e.what() << "\n";
              }  // end try run function

            } catch (const std::exception& e) {
              // can't get module
              /*
              * the thread that produces module
              * should stop
              * TODO: stop the thread
              */
             print(2) << "Can't get schedule module: " << e.what() << "\n";
            }  // end try get mod
          }  // end status == ready
        }  // end try new function

        // then, try to use old function
        if (!succ) {
          // std::cout << "try old function for " << key->value << "\n";
          if (best_functions.find(key) != best_functions.end()) {
            // std::cout << "got old function for " << key->value << "\n";
            auto func = best_functions[key].first->GetFunction(get_func_name(key));
            (*call_unpack)(func, arrays);
            succ = true;
          }
        }  // end try old function

        // must check taken because chance is that
        // the scheduler is not ready
        if (!succ && this->functions[key].empty() && taken) {
          // there is no way to run this subgraph
          // report error
          this->emergency_queue.push(key);
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
        run_helper(k, update_set, delete_set);
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
      print(2) << "Time cost: " << execution_time << " ms.\n";
    }
    if (ad > 0)
      total_execution_time += execution_time;
  }  // for ad

  if (advance_number > 1)
    print(1) << "Average time cost for each iteration: " << total_execution_time / (advance_number-1) << " ms.\n";
  print(1) << "Best iteration takes: " << best_execution_time << " ms.\n";

  // notify done
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish = true;
  lock.unlock();
}


void Session::run_autoschedule(TIRMultiGraph multi_graph, int advance_number) {
  // forward the compilation by multiple iterations
  for (int ad = 0; ad < advance_number; ++ad) {
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
        if (!this->emergency_queue.empty()) {
          auto& key = this->emergency_queue.front();

          // the following are repeated
          // TODO: isolate the logic
          // handle emergency
          // get future schedule
          std::shared_future<ScheduleResult> schedule_result = auto_scheduler->schedule_for(
            key, multi_graph.Self()->graphs[key], target, 1);  // priority 1

          // int max_wait_times = 10;
          // int milliseconds = 1000 * 1000;

          // std::future_status status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
          // int count_wait = 0;
          // while (status == std::future_status::deferred) {
          //   count_wait += 1;
          //   status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
          //   if (count_wait > max_wait_times) {
          //     throw std::runtime_error("Long time still deferred.");
          //   }
          // }
          // status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
          // if (status == std::future_status::timeout) {
          //   continue;
          // }

          try {
            ScheduleResult result = schedule_result.get();
            this->emergency_queue.pop();
            
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

            functions[key].push(sch_func);
          } catch (const std::exception& e) {
            print(2) << "Can't get schedule for emergency: " << e.what() << "\n";
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
        std::shared_future<ScheduleResult> schedule_result = auto_scheduler->schedule_for(
          cand, subgraph, target, 0);

        /*
         * these are parameters
         * should be moved to interface
         * TODO: move the parameters
         */
        // int max_wait_times = 10;
        // int milliseconds = 1000 * 1000;

        // wait for schedule
        // std::future_status status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        // int count_wait = 0;
        // while (status == std::future_status::deferred) {
        //   count_wait += 1;
        //   status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        //   if (count_wait > max_wait_times) {
        //     throw std::runtime_error("Long time still deferred.");
        //   }
        // }
        // status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        // if (status == std::future_status::timeout) {
        //   /*
        //    * should tell the thread to stop
        //    * TODO: stop the thread
        //    */
        //   continue;
        // }

        try {
          ScheduleResult result = schedule_result.get();
          
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

          functions[cand].push(sch_func);

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
          print(2) << "Can't get schedule: " << e.what() << "\n";
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
  }  // for ad

  // wait until finished
  while (1) {
    // see if not done
    bool peek_finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    peek_finish = this->finish;
    lock.unlock();
    if (!peek_finish) {
      if (!this->emergency_queue.empty()) {
        auto& key = this->emergency_queue.front();

        // the following are repeated
        // TODO: isolate the logic
        // handle emergency
        // get future schedule
        std::shared_future<ScheduleResult> schedule_result = auto_scheduler->schedule_for(
          key, multi_graph.Self()->graphs[key], target, 1);  // priority 1

        // int max_wait_times = 10;
        // int milliseconds = 1000 * 1000;

        // std::future_status status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        // int count_wait = 0;
        // while (status == std::future_status::deferred) {
        //   count_wait += 1;
        //   status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        //   if (count_wait > max_wait_times) {
        //     throw std::runtime_error("Long time still deferred.");
        //   }
        // }
        // status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        // if (status == std::future_status::timeout) {
        //   continue;
        // }

        try {
          ScheduleResult result = schedule_result.get();
          this->emergency_queue.pop();
          
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

          functions[key].push(sch_func);
        } catch (const std::exception& e) {
          print(2) << "Can't get schedule for emergency: " << e.what() << "\n";
          continue;
        }
      }
    } else {
      break;
    }
  }  // while 1
}


void Session::run(TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  int advance_number = (int)(bindings.size());
  print(1) << "Advancing " << advance_number << " iterations.\n";

  // begin
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish = false;
  lock.unlock();

  // partition graph
  SubGraphPartitionEngine partition_engine;
  TIRMultiGraph multi_graph(graph, partition_engine);

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
  exe_thread.join();
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


void initialize_weights(
  int session_id, TIRGraph graph, std::vector<tvm::runtime::NDArray> bindings) {

  auto sess = get_session(session_id);
  sess->initialize_weights(graph, bindings);
}


void run_graph(
  int session_id, TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  
  auto sess = get_session(session_id);
  sess->run(graph, bindings);
}


TVM_REGISTER_GLOBAL("tg.create_session_option")
.set_body_typed([](
  bool report_profile,
  bool report_iteration,
  int report_iteration_period,
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
  int execution_parallel,
  double execution_timeout
) {
  SessionOption ret = SessionOption(
    report_profile,
    report_iteration,
    report_iteration_period,
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
    execution_parallel,
    execution_timeout);
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


TVM_REGISTER_GLOBAL("tg.initialize_weights")
.set_body_typed([](int session_id, TIRGraph graph, Array<tvm::runtime::NDArray> bindings){
  std::vector<tvm::runtime::NDArray> _bindings;
  for (auto v : bindings) {
    _bindings.push_back(v);
  }
  initialize_weights(session_id, graph, _bindings);
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
  run_graph(session_id, graph, _bindings);
});


}  // namespace tg


}  // namespace tvm