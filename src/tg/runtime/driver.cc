#include "driver.h"


#include "../graph/concrete_graph.h"


namespace tvm {


namespace tg {

Session::Session(Target target, int dev_id) : target(target) {
  if (target->target_name == "cuda") {
    ctx = DLContext({kDLGPU, dev_id});
  } else {
    LOG(FATAL) << "Currently only support CUDA but get " << target->target_name << ".";
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
  
  int advance_number = (int)bindings.size();
  for (int ad = 0; ad < advance_number; ++ad) {
    std::unordered_map<IntKey, int> call_order;
    std::unordered_set<IntKey> free_set;
    for (auto kv : multi_graph->graph_attrs) {
      call_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.second->num_predecessor);
      }
    }

    std::function<void(
      IntKey key,
      std::unordered_set<IntKey>& update_set,
      std::unordered_set<IntKey>& delete_set)> run_helper;
    run_helper = [this, ad, &multi_graph, &call_order, &bindings]
      (IntKey key,
      std::unordered_set<IntKey>& update_set,
      std::unordered_set<IntKey>& delete_set) {

      bool succ = false;
      auto subgraph = multi_graph->graphs[key];
      std::vector<tvm::runtime::NDArray> arrays;
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
      // first, try to get new function
      if (this->func_mutex.find(key) == this->func_mutex.end()) {
        func_mutex[key] = std::make_unique<std::mutex>();
      }
      std::unique_lock<std::mutex> lock(*(this->func_mutex[key]));
      if (this->functions.find(key) != this->functions.end()) {
        int num_f = (int)this->functions[key].size();
        // record the position
        int count_f = 0;
        for (count_f = 0; count_f < num_f; ++count_f) {
          auto& f = this->functions[key][count_f];
          // these are parameters
          int milliseconds = 1000;
          int max_wait_times = 10;

          // first get schedule & function
          auto status = f.wait_for(std::chrono::milliseconds(milliseconds));
          int count_wait = 0;
          while (status == std::future_status::deferred) {
            status = f.wait_for(std::chrono::milliseconds(milliseconds));
            count_wait += 1;
            if (count_wait >= max_wait_times) {
              break;
            }
          }
          status = f.wait_for(std::chrono::milliseconds(milliseconds));
          // fail to get schedule and function
          if (status != std::future_status::ready) {
            continue;
          }

          try {
            auto sch_func = f.get();
            auto func = sch_func.second->GetFunction(get_func_name(key));

            // call_function(func, arrays);
            auto future = ThreadPool::Global().push_back(
              [](tvm::runtime::PackedFunc& f, std::vector<tvm::runtime::NDArray>& v) {
                auto start = std::chrono::steady_clock::now();
                call_function(f, v);
                auto end = std::chrono::steady_clock::now();
                float elapsed_time = (float)((end - start).count());
                return elapsed_time;
              }, func, arrays);
            
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
            // fail to run this function
            if (status != std::future_status::ready) {
              continue;
            }

            try {
              float elapsed_time = future.get();
              // feedback
              float gflops = get_gflop(subgraph) / elapsed_time;
              sch_func.first.Self()->leaf->update_reward(sch_func.first->configs, gflops);
              // store function
              if (best_functions.find(key) == best_functions.end()) {
                best_functions[key] = std::make_pair(sch_func.second, gflops);
              } else {
                if (gflops > best_functions[key].second) {
                  best_functions[key] = std::make_pair(sch_func.second, gflops);
                }
              }

              // success
              succ = true;
              // only run one
              break;
            } catch (const std::exception &e) {
              // can't run the function
              continue;
            }
          } catch (const std::exception& e) {
            // can't get schedule & function
            continue;
          }

        }
        // success
        if (succ) {
          // remove the schedule & functions that are evaluated
          std::vector<std::future<std::pair<ScheduleResult, tvm::runtime::Module> > > tmp;
          for (int tmp_f = count_f + 1; tmp_f < num_f; ++tmp_f) {
            tmp.push_back(std::move(this->functions[key][tmp_f]));
          }
          this->functions[key] = tmp;
        }
      }
      lock.unlock();
      // then, try to use old function
      if (!succ) {
        if (best_functions.find(key) != best_functions.end()) {
          auto func = best_functions[key].first->GetFunction(get_func_name(key));
          call_function(func, arrays);
          succ = true;
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
      } else {
        // must re-schedule and re-build
        this->emergency_queue.push(key);
      }
    };

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
  std::thread exe_thread(
    [this](TIRMultiGraph g, std::vector<std::unordered_map<te::Tensor, tvm::runtime::NDArray> > b) {
      run_functions(g, b);
    }, multi_graph, bindings);

  // forward the compilation by multiple iterations
  for (int ad = 0; ad < advance_number; ++ad) {
    // initialize call order
    std::unordered_map<IntKey, int> schedule_order;
    std::unordered_set<IntKey> free_set;
    for (auto kv : multi_graph->graph_attrs) {
      schedule_order[kv.first] = kv.second->num_predecessor;
      if (kv.second->num_predecessor == 0) {
        free_set.insert(kv.second->num_predecessor);
      }
    }

    // schedule and build for subgraphs
    int schedule_count = 0;
    int num_subgraphs = (int)multi_graph->graphs.size();
    while (!free_set.empty()) {
      std::unordered_set<IntKey> new_set;

      for (auto cand : free_set) {
        // get future schedule
        std::future<ScheduleResult> result = AutoScheduler::Global().schedule_for(
          cand, multi_graph->graphs[cand], target, 0);
        
        // get future func
        std::future<std::pair<ScheduleResult, tvm::runtime::Module> > sch_func = \
        FunctionBuilder::Global().build_for_future(
          result,
          target,
          Target::Create("llvm"),
          get_func_name(cand),
          std::unordered_map<te::Tensor, tir::Buffer>(),
          tvm::BuildConfig::Create()
        );

        // store the future func
        if (func_mutex.find(cand) == func_mutex.end()) {
          func_mutex[cand] = std::make_unique<std::mutex>();
        }
        std::unique_lock<std::mutex> lock(*func_mutex[cand]);
        if (functions.find(cand) != functions.end()) {
          functions[cand] = std::vector<std::future<std::pair<ScheduleResult, tvm::runtime::Module> > >();
        }
        functions[cand].push_back(std::move(sch_func));
        lock.unlock();

        // update free set
        for (auto succ : multi_graph->graph_attrs[cand]->successors) {
          schedule_order[succ] -= 1;
          if (schedule_order[succ] == 0) {
            new_set.insert(succ);
          }
        }
        
        // this subgraph is done
        schedule_count += 1;
      }

      free_set.clear();
      for (auto new_cand : new_set) {
        free_set.insert(new_cand);
      }
    }
    
    // make sure that every subgraph is handled
    if (schedule_count != num_subgraphs) {
      throw std::runtime_error("Schedule graph number mismatch");
    }
  }

  while (1) {
    // see if not done
    std::unique_lock<std::mutex> lock(this->finish_mutex);
    if (!this->finish) {
      lock.unlock();
      if (!this->emergency_queue.empty()) {
        auto key = this->emergency_queue.pop();
        // handle emergency
        // get future schedule
        std::future<ScheduleResult> result = AutoScheduler::Global().schedule_for(
          key, multi_graph->graphs[key], target, 1);  // priority 1
        
        // get future func
        std::future<std::pair<ScheduleResult, tvm::runtime::Module> > sch_func = \
        FunctionBuilder::Global().build_for_future(
          result,
          target,
          Target::Create("llvm"),
          get_func_name(key),
          std::unordered_map<te::Tensor, tir::Buffer>(),
          tvm::BuildConfig::Create(),
          1
        );  // priority 1

        // store the future func
        if (func_mutex.find(key) == func_mutex.end()) {
          func_mutex[key] = std::make_unique<std::mutex>();
        }
        std::unique_lock<std::mutex> lock(*func_mutex[key]);
        if (functions.find(key) != functions.end()) {
          functions[key] = std::vector<std::future<std::pair<ScheduleResult, tvm::runtime::Module> > >();
        }
        functions[key].push_back(std::move(sch_func));
        lock.unlock();
      }
    } else {
      lock.unlock();
      break;
    }
  }
  // wait for execution
  exe_thread.join();
}


}  // namespace tg


}  // namespace tvm