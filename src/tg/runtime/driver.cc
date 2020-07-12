#include "driver.h"
#include <unistd.h>


#include "../graph/concrete_graph.h"


namespace tvm {


namespace tg {

Session::Session(Target target, int dev_id) : target(target) {
  if (target->target_name == "cuda") {
    ctx = DLContext({kDLGPU, dev_id});
  } else if (target->target_name == "llvm") {
    ctx = DLContext({kDLCPU, dev_id});
  } else {
    LOG(FATAL) << "Currently only support CUDA/LLVM but get " << target->target_name << ".";
  }
};


void Session::initialize_weights(TIRGraph graph, std::unordered_map<te::Tensor, tvm::runtime::NDArray> bindings) {
  // static bindings
  // initialize for weights
  for (auto kv : bindings) {
    persistent_tensors[kv.first] = kv.second;
  }
  
  int i = 0;
  for (auto t : graph->gradients) {
    CHECK(persistent_tensors.find(graph->weights[i]) != persistent_tensors.end())
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
  
  std::cout << "check in run functions\n";
  // prepare the call_unpack
  const auto* call_unpack = runtime::Registry::Get("tg.runtime.call_unpack");
  CHECK(call_unpack != nullptr) << "Should prepare call_unpack function.";
  std::cout << "get call_unpack function\n";

  int advance_number = (int)bindings.size();
  std::cout << "get advance number " << advance_number << "\n";

  std::cout << "check multigraph:\n";
  std::cout << "graph attrs:\n";
  for (auto kv : multi_graph->graph_attrs) {
    std::cout << kv.first->value << "\n";
  }
  std::cout << "graphs:\n";
  for (auto kv : multi_graph->graphs) {
    std::cout << kv.first->value << "\n";
  }

  for (int ad = 0; ad < advance_number; ++ad) {
    std::unordered_map<IntKey, int> call_order;
    std::unordered_set<IntKey> free_set;
    for (auto kv : multi_graph->graph_attrs) {
      call_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.first);
      }
    }

    std::function<void(
      IntKey key,
      std::unordered_set<IntKey>& update_set,
      std::unordered_set<IntKey>& delete_set)> run_helper;
    run_helper = [this, ad, &multi_graph, &call_order, &bindings, call_unpack]
      (IntKey key,
      std::unordered_set<IntKey>& update_set,
      std::unordered_set<IntKey>& delete_set) {
      
      std::cout << "check in run helper\n";

      bool succ = false;
      std::cout << "before get subgraph " << key->value << "\n";
      std::cout << "check multi graph graphs\n";
      for (auto kv : multi_graph->graphs) {
        std::cout << "key= " << kv.first->value << " " << (key == (kv.first)) <<  "\n";
      }
      return;
      auto subgraph = multi_graph->graphs[key];
      std::cout << "after get subgraph\n";
      Array<tvm::runtime::NDArray> arrays;
      for (auto tt : subgraph->inputs) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (bindings[ad].find(t) != bindings[ad].end()) {
          arrays.push_back(bindings[ad][t]);
        } else if (this->volatile_tensors.find(t) != this->volatile_tensors.end()) {
          arrays.push_back(this->volatile_tensors[t]);
        } else {
          LOG(FATAL) << "Can't find input " << t;
        }
      }
      for (auto tt : subgraph->labels) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (bindings[ad].find(t) == bindings[ad].end()) {
          LOG(FATAL) << "Can't find label " << t;
        }
        arrays.push_back(bindings[ad][t]);
      }
      for (auto tt : subgraph->outputs) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (volatile_tensors.find(t) == volatile_tensors.end()) {
          LOG(FATAL) << "Can't find output " << t;
        }
        arrays.push_back(this->volatile_tensors[t]);
      }
      for (auto tt : subgraph->weights) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (persistent_tensors.find(t) == persistent_tensors.end()) {
          LOG(FATAL) << "Can't find weight " << t;
        }
        arrays.push_back(this->persistent_tensors[t]);
      }
      if (subgraph->loss.defined()) {
        te::Tensor t = multi_graph.Self()->tensor_index[subgraph->loss];
        if (persistent_tensors.find(t) == persistent_tensors.end()) {
          LOG(FATAL) << "Can't find weight " << t;
        }
        arrays.push_back(this->persistent_tensors[t]);
      }
      for (auto tt : subgraph->gradients) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (persistent_tensors.find(t) == persistent_tensors.end()) {
          LOG(FATAL) << "Can't find gradient " << t;
        }
        arrays.push_back(this->persistent_tensors[t]);
      }
      if (subgraph->lr.defined()) {
        te::Tensor t = multi_graph.Self()->tensor_index[subgraph->lr];
        if (bindings[ad].find(t) == bindings[ad].end()) {
          LOG(FATAL) << "Can't find label " << t;
        }
        arrays.push_back(bindings[ad][t]);
      }
      for (auto tt : subgraph->updates) {
        te::Tensor t = multi_graph.Self()->tensor_index[tt];
        if (persistent_tensors.find(t) == persistent_tensors.end()) {
          LOG(FATAL) << "Can't find update " << t;
        }
        arrays.push_back(this->persistent_tensors[t]);
      }

      std::cout << "check arrays:\n";
      for (auto a : arrays) {
        std::cout << a << "\n";
      }
      
      succ = false;
      while (!succ) {
        bool taken = false;  // record taken a value
        // first, try to get new function
        if (!this->functions[key].empty()) {
          std::cout << "check try new function\n";

          taken = true;
          auto& sch_func = this->functions[key].front();
          this->functions[key].pop();
          auto schedule_result = sch_func.first;
          auto& func = sch_func.second;
          // these are parameters
          int milliseconds = 1000;
          int max_wait_times = 10;

          // first get schedule & function
          auto status = func.wait_for(std::chrono::milliseconds(milliseconds));
          int count_wait = 0;
          while (status == std::future_status::deferred) {
            status = func.wait_for(std::chrono::milliseconds(milliseconds));
            count_wait += 1;
            if (count_wait >= max_wait_times) {
              break;
            }
          }
          status = func.wait_for(std::chrono::milliseconds(milliseconds));
          // get schedule and function if ready
          // else pass this one
          if (status == std::future_status::ready) {
            try {
              std::cout << "check try to get function\n";
              auto module = func.get();
              auto mod_func = module->GetFunction(get_func_name(key));
              std::cout << "check get function successfully\n";

              // call_function(func, arrays);
              auto future = ThreadPool::Global().push_back(
                [call_unpack](tvm::runtime::PackedFunc& f, Array<tvm::runtime::NDArray>& v) {
                  auto start = std::chrono::steady_clock::now();
                  (*call_unpack)(f, v);
                  auto end = std::chrono::steady_clock::now();
                  float elapsed_time = (float)((end - start).count());
                  return elapsed_time;
                }, mod_func, arrays);
              
              // wait for 1s
              status = future.wait_for(std::chrono::milliseconds(milliseconds));
              int count_wait = 0;
              while (status == std::future_status::deferred) {
                status = future.wait_for(std::chrono::milliseconds(milliseconds));
                count_wait += 1;
                if (count_wait >= max_wait_times) {
                  break;
                }
              }
              status = future.wait_for(std::chrono::milliseconds(milliseconds));
              // run this function
              // if not ready, pass
              if (status == std::future_status::ready) {
                try {
                  std::cout << "check try to run function\n";
                  float elapsed_time = future.get();
                  // feedback
                  float gflops = get_gflop(subgraph) / elapsed_time;
                  std::cout << "GFLOPS: " << gflops << "\n";
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
                  // pass
                  std::cout << "can't run function because " << e.what() << "\n";
                }
              }
            } catch (const std::exception& e) {
              // can't get schedule & function
              // pass
              std::cout << "can't get function because " << e.what() << "\n";
            }
          }
        }

        // then, try to use old function
        if (!succ) {
          if (best_functions.find(key) != best_functions.end()) {
            std::cout << "try to use old function\n";
            auto func = best_functions[key].first->GetFunction(get_func_name(key));
            (*call_unpack)(func, arrays);
            succ = true;
          }
        }

        // must check taken because chance is that
        // the scheduler is not ready
        if (!succ && this->functions[key].empty() && taken) {
          // there is no way to run this subgraph
          // report error
          std::cout << "report an error " << key << "\n";
          this->emergency_queue.push(key);
          break;
        }

      }

      if (succ) {
        // update free set
        delete_set.insert(key);
        for (auto v : multi_graph->graph_attrs[key]->successors) {
          call_order[v] -= 1;
          if (call_order[v] == 0) {
            update_set.insert(v);
          }
        }
      }

    };

    while (!free_set.empty()) {
      std::unordered_set<IntKey> update_set, delete_set;

      std::cout << "check free set:\n";
      for (auto k : free_set) {
        std::cout << k << " ";
      }
      std::cout << "\n";

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
  }

  // notify done
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish = true;
  lock.unlock();
}


void Session::run(TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  int advance_number = (int)(bindings.size());
  LOG(INFO) << "Advancing " << advance_number << " iterations.";

  // begin
  std::unique_lock<std::mutex> lock(this->finish_mutex);
  this->finish = false;
  lock.unlock();

  // partition graph
  SubGraphPartitionEngine partition_engine;
  TIRMultiGraph multi_graph(graph, partition_engine);

  // allocate output/loss/gradients/updates buffer
  // the weight buffers should be initialized before
  allocate_output_buffer(multi_graph);

  std::cout << "after allocate buffer\n";
  // std::thread exe_thread(
  //   [this](TIRMultiGraph g, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > b) {
  //     run_functions(g, b);
  //   }, multi_graph, bindings);
  
  std::cout << "after launch runner\n";

  // forward the compilation by multiple iterations
  for (int ad = 0; ad < advance_number; ++ad) {
    std::cout << "in ad " << ad << "\n";
    // initialize call order
    std::unordered_map<IntKey, int> schedule_order;
    std::unordered_set<IntKey> free_set;
    for (auto kv : multi_graph->graph_attrs) {
      schedule_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.first);
      }
    }

    std::cout << "multi_graph graphs:\n";
    for (auto kv : multi_graph->graphs) {
      std::cout << kv.first->value << "\n";
    }

    std::cout << "multi_graph graph attrs:\n";
    for (auto kv : multi_graph->graph_attrs) {
      std::cout << kv.first->value << "\n";
    }

    // schedule and build for subgraphs
    int schedule_count = 0;
    int num_subgraphs = (int)multi_graph->graphs.size();
    while (!free_set.empty()) {
      std::unordered_set<IntKey> update_set;
      std::unordered_set<IntKey> delete_set;

      for (auto cand : free_set) {
        std::cout << "auto schedule for cand= " << cand->value << "\n";
        if(multi_graph->graphs.find(cand) == multi_graph->graphs.end())
          std::cout << "Can't find subgraph " << cand->value << " in multigraph.\n";
        std::cout << "find cand in multi_graph graphs:\n";
        for (auto kv : multi_graph->graphs) {
          std::cout << kv.first->value << " compare with cand: " << (cand == kv.first) << "\n";
        }
        // get future schedule
        usleep(1000*1000);
        std::cout << "sleep done\n";
        auto subgraph = multi_graph->graphs[cand];
        std::future<ScheduleResult> schedule_result = AutoScheduler::Global().schedule_for(
          cand, subgraph, target, 0);

        int max_wait_times = 10;
        int milliseconds = 1000;

        std::future_status status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        int count_wait = 0;
        while (status == std::future_status::deferred) {
          count_wait += 1;
          status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
          if (count_wait > max_wait_times) {
            throw std::runtime_error("Long time still deferred.");
          }
        }
        status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        if (status == std::future_status::timeout) {
          continue;
        }

        try {
          std::cout << "try to get schedule\n";
          ScheduleResult result = schedule_result.get();
          std::cout << "after get schedule\n";
          
          // get future func
          std::pair<ScheduleResult, std::future<tvm::runtime::Module> > sch_func = \
          FunctionBuilder::Global().build_for(
            result,
            target,
            Target::Create("llvm"),
            get_func_name(cand),
            std::unordered_map<te::Tensor, tir::Buffer>(),
            tvm::BuildConfig::Create()
          );

          // store the future func
          // if (func_mutex.find(cand) == func_mutex.end()) {
          //   func_mutex[cand] = std::make_unique<std::mutex>();
          // }
          // std::unique_lock<std::mutex> lock(*func_mutex[cand]);
          // if (functions.find(cand) != functions.end()) {
          //   functions[cand] = std::move(Queue<std::pair<ScheduleResult, std::future<tvm::runtime::Module> > >());
          // }
          // unordered_set should insert a new queue if the key doesn't exist
          std::cout << "before put sch_func\n";
          functions[cand].push(std::move(sch_func));
          std::cout << "after put sch_func\n";
          // lock.unlock();

          // update delete_set
          delete_set.insert(cand);

          // update free set
          for (auto succ : multi_graph->graph_attrs[cand]->successors) {
            schedule_order[succ] -= 1;
            if (schedule_order[succ] == 0) {
              update_set.insert(succ);
            }
          }
          
          // this subgraph is done
          schedule_count += 1;
        } catch (const std::exception& e) {
          continue;
        }
      }

      for (auto deleted : delete_set) {
        free_set.erase(deleted);
      }
      for (auto new_cand : update_set) {
        free_set.insert(new_cand);
      }
    }
    
    // make sure that every subgraph is handled
    // double check
    if (schedule_count != num_subgraphs) {
      throw std::runtime_error("Schedule graph number mismatch");
    }
  }

  while (1) {
    // see if not done
    bool peek_finish = false;
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    peek_finish = this->finish;
    lock.unlock();
    if (!peek_finish) {
      if (!this->emergency_queue.empty()) {
        auto& key = this->emergency_queue.front();
        this->emergency_queue.pop();
        // handle emergency
        // get future schedule
        std::future<ScheduleResult> schedule_result = AutoScheduler::Global().schedule_for(
          key, multi_graph->graphs[key], target, 1);  // priority 1

        int max_wait_times = 10;
        int milliseconds = 1000;

        std::future_status status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        int count_wait = 0;
        while (status == std::future_status::deferred) {
          count_wait += 1;
          status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
          if (count_wait > max_wait_times) {
            throw std::runtime_error("Long time still deferred.");
          }
        }
        status = schedule_result.wait_for(std::chrono::milliseconds(milliseconds));
        if (status == std::future_status::timeout) {
          continue;
        }

        try {
          ScheduleResult result = schedule_result.get();
          
          // get future func
          std::pair<ScheduleResult, std::future<tvm::runtime::Module> > sch_func = \
          FunctionBuilder::Global().build_for(
            result,
            target,
            Target::Create("llvm"),
            get_func_name(key),
            std::unordered_map<te::Tensor, tir::Buffer>(),
            tvm::BuildConfig::Create(),
            1  // priority 1
          );

          // store the future func
          // if (func_mutex.find(cand) == func_mutex.end()) {
          //   func_mutex[cand] = std::make_unique<std::mutex>();
          // }
          // std::unique_lock<std::mutex> lock(*func_mutex[cand]);
          // if (functions.find(cand) != functions.end()) {
          //   functions[cand] = std::move(Queue<std::pair<ScheduleResult, std::future<tvm::runtime::Module> > >());
          // }
          // unordered_set should insert a new queue if the key doesn't exist
          functions[key].push(std::move(sch_func));
          // lock.unlock();
        } catch (const std::exception& e) {
          continue;
        }
      }
    } else {
      break;
    }
  }
  // wait for execution
  // exe_thread.join();
}


std::shared_ptr<Session> create_or_get_session(Target target, int dev_id, int& session_id, bool get_session) {
  static std::unordered_map<int, std::shared_ptr<Session> > sessions;
  static int global_count = 0;
  if (get_session) {
    CHECK(sessions.find(session_id) != sessions.end()) << "Can't find the session " << session_id << ".";
    return sessions[session_id];
  } else {
    // create session
    sessions[global_count] = std::make_shared<Session>(target, dev_id);
    session_id = global_count;  // record the session id
    global_count += 1;
    return sessions[global_count];
  }
}


int create_session(Target target, int dev_id) {
  int ret = -1;
  create_or_get_session(target, dev_id, ret, false);
  CHECK(ret >= 0) << "Invalid session id when creating session: " << ret << ".";
  return ret;
}

std::shared_ptr<Session> get_session(int session_id) {
  // pass dummy target info
  return create_or_get_session(target::llvm(), 0, session_id, true);
}


void initialize_weights(
  int session_id, TIRGraph graph, std::unordered_map<te::Tensor, tvm::runtime::NDArray> bindings) {

  auto sess = get_session(session_id);
  sess->initialize_weights(graph, bindings);
}


void run_graph(
  int session_id, TIRGraph graph, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > bindings) {
  
  auto sess = get_session(session_id);
  sess->run(graph, bindings);
}


TVM_REGISTER_GLOBAL("tg.create_session")
.set_body_typed([](Target target, int dev_id){
  return create_session(target, dev_id);
});


TVM_REGISTER_GLOBAL("tg.get_context_from_session")
.set_body_typed([](int session_id){
  auto sess = get_session(session_id);
  return sess->ctx;
});


TVM_REGISTER_GLOBAL("tg.initialize_weights")
.set_body_typed([](int session_id, TIRGraph graph, Map<te::Tensor, tvm::runtime::NDArray> bindings){
  std::unordered_map<te::Tensor, tvm::runtime::NDArray> _bindings;
  for (auto kv : bindings) {
    _bindings[kv.first] = kv.second;
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