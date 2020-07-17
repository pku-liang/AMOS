#include "utils.h"


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



}  // namespace tg


}  // namespace tvm