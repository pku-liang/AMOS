#include "utils.h"


namespace tvm {

namespace tg {

inline ThreadPool::ThreadPool(size_t threads=std::thread::hardware_concurrency()) : stop(false) {
    for(size_t i = 0;i<threads;++i) {
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
}


ThreadPool& ThreadPool::Global() {
  static ThreadPool* pool = new ThreadPool();
  
  return *pool;
}


inline ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(deque_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

}  // namespace tg


}  // namespace tvm