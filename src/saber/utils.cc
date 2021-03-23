#include <vector>

#include <tvm/runtime/ndarray.h>
#include <tvm/tir/expr.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/container.h>
#include <tvm/runtime/data_type.h>

namespace tvm {
using namespace tvm::runtime;
namespace saber {

TVM_REGISTER_GLOBAL("saber.NDArrayView").set_body_typed([](
  NDArray ary, Array<IntImm> shape, String dtype) {
    std::vector<int64_t> shapes;
    for (auto s : shape) {
      shapes.push_back(s->value);
    }
    return ary.CreateView(shapes, String2DLDataType(std::string(dtype)));
});


}  // namespace saber

}  // namespace tvm