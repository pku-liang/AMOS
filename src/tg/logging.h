#ifndef TVM_TG_LOGGING_H_
#define TVM_TG_LOGGING_H_

#include <sstream>
#include <chrono>
#include <iostream>

namespace tvm {

namespace tg {

int get_evn_value(std::string name);

std::chrono::milliseconds current_time();


enum class LogLevel {
  tINFO,
  tWARNING,
  tERROR
};


class LazyLogging {
 private:
  LogLevel log_level;
  bool do_print;
  std::string file_;
  int lineno_;
  std::ostringstream oss;
public:
  LazyLogging() = default;
  LazyLogging(const LazyLogging &&other) : log_level(other.log_level), do_print(other.do_print) {}
  LazyLogging(LogLevel level, bool do_print=true, std::string file=__FILE__, int lineno=__LINE__) :
    log_level(level), do_print(do_print), file_(file), lineno_(lineno) {}
  ~LazyLogging() {
    std::chrono::milliseconds ms = current_time();
    if (do_print) {
      switch (log_level)
      {
      case LogLevel::tINFO:
        std::cerr << "[Info] " << "[time=" << ms.count() << "] " << oss.str() << std::flush;
        break;
      case LogLevel::tWARNING:
        std::cerr << "[Warning] " << "[time=" << ms.count() << "] file:"
                  << file_ << " line:" << lineno_ << " " << oss.str() << std::flush;
        break;
      case LogLevel::tERROR:
        {std::cerr << "[Error] " << "[time=" << ms.count() << "] "
                  << file_ << " line:" << lineno_ << " " << oss.str() << std::flush;
        abort();}
        break;
      default:
        break;
      }
    }
  }

  template<typename T>
  LazyLogging &operator<<(T &other) {
      oss << other;
      return *this;
  }

  template<typename T>
  LazyLogging &operator<<(T &&other) {
      oss << other;
      return *this;
  }
};


#define WARN(cond)                                                            \
  (                                                                           \
    [&]()-> LazyLogging {                                                     \
      if (!(cond)) {                                                          \
        return LazyLogging(LogLevel::tWARNING, true, __FILE__, __LINE__);     \
      } else {                                                                \
        return LazyLogging(LogLevel::tINFO, false, __FILE__, __LINE__);       \
      }                                                                       \
    }()                                                                       \
  ) 


#define ASSERT(cond)                                                          \
  (                                                                           \
    [&]()-> LazyLogging {                                                     \
      if (!(cond)) {                                                          \
        return LazyLogging(LogLevel::tERROR, true, __FILE__, __LINE__);       \
      } else {                                                                \
        return LazyLogging(LogLevel::tINFO, false, __FILE__, __LINE__);       \
      }                                                                       \
    }()                                                                       \
  )                                                                           


#define ERROR (ASSERT(false))


class print{
 private:
  bool do_print;
  std::ostream& out;
 public:
  print(int level, std::ostream& out=std::cerr)
  : do_print(level == 0 || level <= get_evn_value("TG_PRINT_LEVEL")), out(out) {}

  template<typename T>
  print& operator<< (T&& x) {
    if (do_print) {
      out << std::forward<T>(x) << std::flush;
    }
    return *this;
  }
};


class ProgressBar {
 private:
  int length;
 public:
  ProgressBar(int length=80) : length(length) {}

  void draw(double progress) {
    if (progress < 0) {
      progress = 0.0;
    } else if (progress > 1) {
      progress = 1.0;
    }
    int pos = (int)(progress * (length - 2));
    std::cerr << "[" << std::flush;
    for (int i = 0; i < length - 2; ++i) {
      if (i < pos) {
        std::cerr << "#" << std::flush;
      } else if (i == pos) {
        std::cerr << "X" << std::flush;
      } else {
        std::cerr << " " << std::flush;
      }
    }
    std::cerr << "]\r" << std::flush;
  }

  void end() {
    std::cerr << "\n" << std::flush;
  }
};


}  // namespace tg

}  // namespace tvm


#endif  // TVM_TG_LOGGING_H_