#ifndef BUILD_POOL_H
#define BUILD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

#include "tvm/te/operation.h"
#include "tvm/te/schedule.h"
#include "tvm/te/tensor.h"
#include "tvm/driver/driver_api.h"
#include "tvm/target/target.h"
#include "tvm/runtime/module.h"

class BuildPool {
public:
    BuildPool(size_t);

    template<class... Args>
    std::future<tvm::runtime::Module> push_front(Args&&... args);

    template<class... Args>
    std::future<tvm::runtime::Module> push_back(Args&&... args);

    ~BuildPool();
private:
    static tvm::runtime::Module build_func(
        tvm::te::Schedule sch,
        const tvm::Array<tvm::te::Tensor>& args,
        const tvm::Target& target,
        const tvm::Target& target_host,
        const std::string& name,
        const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
        const tvm::BuildConfig& config);

    std::vector< std::thread > workers;
    std::deque< std::function<void()> > tasks;
    
    std::mutex deque_mutex;
    std::condition_variable condition;
    bool stop;
};
 

inline BuildPool::BuildPool(size_t threads=std::thread::hardware_concurrency()) : stop(false) {
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this] {
                for(;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->deque_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop_front();
                    }

                    task();
                }
            }
        );
}


tvm::runtime::Module BuildPool::build_func(
    tvm::te::Schedule sch,
    const tvm::Array<tvm::te::Tensor>& args,
    const tvm::Target& target,
    const tvm::Target& target_host,
    const std::string& name,
    const std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer>& binds,
    const tvm::BuildConfig& config) {
    
    return tvm::build(
        tvm::lower(sch, args, name, binds, config),
        target,
        target_host,
        config
    );
}


template<class... Args>
std::future<tvm::runtime::Module> BuildPool::push_front(Args&&... args) {
    using return_type = tvm::runtime::Module;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(build_func, std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_frony on stopped BuildPool");

        tasks.emplace_front([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}


template<class... Args>
std::future<tvm::runtime::Module> BuildPool::push_back(Args&&... args) {
    using return_type = tvm::runtime::Module;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(build_func, std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_back on stopped BuildPool");

        tasks.emplace_back([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}


inline BuildPool::~BuildPool() {
    {
        std::unique_lock<std::mutex> lock(deque_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

#endif