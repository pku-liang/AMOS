#include "tvm/te/operation.h"
#include "tvm/te/schedule.h"
#include "tvm/te/tensor.h"
#include "tvm/driver/driver_api.h"
#include "tvm/target/target.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/runtime/module.h"
#include "topi/nn.h"
#include "../src/tg/autoschedule/feature.h"


#include <iostream>

int main() {
    int n = 512;
    auto A = tvm::te::placeholder({n}, tvm::DataType::Float(32), "A");
    auto B = tvm::te::placeholder({n}, tvm::DataType::Float(32), "B");

    auto C = tvm::te::compute({n}, tvm::te::FCompute([=](auto i){ return A(i) + B(i); } )) ;

    auto sch = tvm::te::create_schedule({C->op});

    auto feature = tvm::tg::get_feature(sch, {A, B, C}, tvm::target::llvm());
    std::cout << feature << std::endl;
}