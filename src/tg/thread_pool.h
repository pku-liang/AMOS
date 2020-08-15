#ifndef TVM_TG_THREAD_POOL_H_
#define TVM_TG_THREAD_POOL_H_

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <exception>
#include <future>
#include <functional>
#include <pthread.h>


namespace tvm {


namespace tg {

class ThreadPool {
public:
  ThreadPool(size_t threads=std::thread::hardware_concurrency(), unsigned int _timeout=1000)
  : num_threads(threads), stop(true), timeout(_timeout) {
    Init();
  }

  void Init() {
    size_t threads = num_threads;
    stop = false;
    workers.clear();
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

  template<typename FType, typename... Args>
  auto push_front_with_timeout(FType&& f, Args&&... args)
    -> std::shared_future<typename std::result_of<FType(Args...)>::type> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(f, std::forward<Args>(args)...)
        );
      
    auto timed_task = std::make_shared< std::packaged_task<return_type()> >(
      [task, this](){
        auto ret = task->get_future();

        std::thread th([task](){ (*task)(); });

        auto status = ret.wait_for(std::chrono::milliseconds(this->timeout));
        if(status != std::future_status::ready) {
          pthread_cancel(th.native_handle());
          th.join();
          throw std::runtime_error("time out");
        } else {
          th.join();
          return ret.get();
        }
      }
    );

    std::shared_future<return_type> res = timed_task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_front on stopped ThreadPool");

        tasks.emplace_front([timed_task]() { (*timed_task)(); });
    }
    condition.notify_one();
    return res;
  }

  template<typename FType, typename... Args>
  auto push_front(FType&& f, Args&&... args)
    -> std::shared_future<typename std::result_of<FType(Args...)>::type> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(f, std::forward<Args>(args)...)
        );

    std::shared_future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_front on stopped ThreadPool");

        tasks.emplace_front([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }

  template<typename FType, typename... Args>
  auto push_back_with_timeout(FType&& f, Args&&... args) -> std::shared_future<typename std::result_of<FType(Args...)>::type> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(f, std::forward<Args>(args)...)
        );
    
    auto timed_task = std::make_shared< std::packaged_task<return_type()> >(
      [task, this](){
        auto ret = task->get_future();

        std::thread th([task](){ (*task)(); });

        auto status = ret.wait_for(std::chrono::milliseconds(this->timeout));
        if(status != std::future_status::ready) {
          pthread_cancel(th.native_handle());
          th.detach();
          throw std::runtime_error("time out in thread pool");
        } else {
          th.join();
          return ret.get();
        }
      }
    );

    std::shared_future<return_type> res = timed_task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_back on stopped ThreadPool");

        tasks.emplace_back([timed_task]() { (*timed_task)(); });
    }
    condition.notify_one();
    return res;
  }

  template<typename FType, typename... Args>
  auto push_back(FType&& f, Args&&... args) -> std::shared_future<typename std::result_of<FType(Args...)>::type> {
    using return_type = decltype(f(args...));

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(f, std::forward<Args>(args)...)
        );

    std::shared_future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        if(stop)
            throw std::runtime_error("push_back on stopped ThreadPool");

        tasks.emplace_back([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
  }

  // static ThreadPool& Global() {
  //   static ThreadPool* pool = new ThreadPool();
  
  //   return *pool;
  // }

  void DropAll() {
    Cancel();
    Stop();
  }

  void Reset() {
    DropAll();
    Join();
    Init();
  }

  void Cancel() {
    {
        std::unique_lock<std::mutex> lock(deque_mutex);

        tasks.clear();
    }
  }

  void Join() {
    condition.notify_all();
    for(std::thread &worker: workers)
      if (worker.joinable())
        worker.join();
      else
        worker.detach();
  }

  void Stop() {
    {
      std::unique_lock<std::mutex> lock(deque_mutex);
      stop = true;
    }
  }

  ~ThreadPool() {
    Reset();
    DropAll();
    Join();
  }
private:
  size_t num_threads;
  bool stop;
  std::vector< std::thread > workers;
  std::deque< std::function<void()> > tasks;
  
  std::mutex deque_mutex;
  std::condition_variable condition;

  unsigned int timeout;

  static const int REFRESH_EPOCH = 128;
};


template<typename T>
class Queue {
 private:
  std::queue<T> q;
  // std::queue<T> write;
  std::mutex mutex;

 public:
  void push(T& value, int num=1) {
    std::unique_lock<std::mutex> lock(mutex);
    for (int i = 0; i < num; ++i)
      q.push(value); // write.push(value);
  }

  void push(T&& value, int num=1) {
    std::unique_lock<std::mutex> lock(mutex);
    for (int i = 0; i < num; ++i)
      q.push(std::move(value)); // write.push(std::move(value));
  }
  
  T& front() {
    // prepare();
    std::unique_lock<std::mutex> lock(mutex);
    return q.front();
  }

  void pop() {
    // prepare();
    std::unique_lock<std::mutex> lock(mutex);
    q.pop();
  }

  bool empty() {
    // prepare();
    std::unique_lock<std::mutex> lock(mutex);
    return q.empty();
  }

  size_t size() {
    // prepare();
    std::unique_lock<std::mutex> lock(mutex);
    return q.size();
  }

  void prepare() {
    // if (q.empty()) {
    //   std::unique_lock<std::mutex> lock(mutex);
    //   while (!write.empty()) {
    //     q.push(write.front());
    //     write.pop();
    //   }
    //   lock.unlock();
    // }
  }
};




}  // namespace tg

}  // namespace tvm


#endif  // TVM_TG_THREAD_POOL_H_
