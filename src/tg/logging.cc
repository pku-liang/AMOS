#include "logging.h"


namespace tvm {

namespace tg {

int get_evn_value(std::string name) {
  char* value = std::getenv(name.c_str());
  if (value != nullptr) {
    return std::stoi(std::string(value));
  } else {
    return 0;
  }
}


std::chrono::milliseconds current_time() {
  return std::chrono::duration_cast< std::chrono::milliseconds >(
          std::chrono::system_clock::now().time_since_epoch()
          );
}

}  // namespace tg

}  // namespace tvm