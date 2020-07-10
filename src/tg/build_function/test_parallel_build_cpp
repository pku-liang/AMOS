#include "tvm/te/operation.h"
#include "tvm/te/schedule.h"
#include "tvm/te/tensor.h"
#include "tvm/driver/driver_api.h"
#include "tvm/target/target.h"
#include "tvm/runtime/ndarray.h"
#include "tvm/runtime/module.h"
#include "topi/nn.h"

#include <iostream>
#include <vector>
#include <queue>
#include <future>

#include "build.h"

using namespace std;

int main() {
    BuildPool pool(2);

    int tasks = 10;
    vector<future<tvm::runtime::Module>> funcs;
    for (int i = 0; i < tasks; ++i) {
        int n = 512;

        auto A = tvm::te::placeholder({n});
        auto B = tvm::te::placeholder({n});

        auto C = tvm::te::compute(
            {n},
            tvm::te::FCompute([=](auto i){ return A(i) + B(i); })
    	);

        auto sch = tvm::te::create_schedule({C->op});
        tvm::Target target = tvm::Target::Create("llvm");
        std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer> binds;
        tvm::BuildConfig config = tvm::BuildConfig::Create();

        funcs.push_back(pool.push_back(sch, tvm::Array<tvm::te::Tensor>({A, B, C}), target, target, "add", binds, config));
        funcs.push_back(pool.push_front(sch, tvm::Array<tvm::te::Tensor>({A, B, C}), target, target, "add", binds, config));
    }

    for (auto &func : funcs) {
        try {
            cout << func.get()->GetSource() << '\n';
        }
        catch(const exception& e) {
            cerr << e.what() << '\n';
        }
    }
}
