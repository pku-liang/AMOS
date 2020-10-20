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
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <ctime>

#include "build.h"
#include <chrono>

using namespace std;

vector< tvm::Array<tvm::te::Tensor> > bufs;
vector<tvm::te::Schedule> schs;

int main() {
	int m = 300;
	for (int t = 0; t < m; ++t) {
		int n = 512;
		auto Input = tvm::te::placeholder({n, n, 3});
        auto Filter1 = tvm::te::placeholder({3, 3, 3, 3});
        auto di1 = tvm::te::reduce_axis({0, 3});
        auto dj1 = tvm::te::reduce_axis({0, 3});
        auto dk1 = tvm::te::reduce_axis({0, 3});
        
        auto Conv1 = tvm::te::compute(
            {n - 2, n - 2, 3},
            [=](auto i, auto j, auto k){ return tvm::sum(Input(i + di1, j + dj1, dk1) * Filter1(k, di1, dj1, dk1), {di1, dj1, dk1}); }
    	);
        
        auto Relu1 = topi::leaky_relu(Conv1);

        auto Filter2 = tvm::te::placeholder({3, 3, 3, 3});
        auto di2 = tvm::te::reduce_axis({0, 3});
        auto dj2 = tvm::te::reduce_axis({0, 3});
        auto dk2 = tvm::te::reduce_axis({0, 3});
        
        auto Conv2 = tvm::te::compute(
            {n - 4, n - 4, 3},
            [=](auto i, auto j, auto k){ return tvm::sum(Relu1(i + di2, j + dj2, dk2) * Filter2(k, di2, dj2, dk2), {di2, dj2, dk2}); }
        );

        auto Relu2 = topi::leaky_relu(Conv2);

        auto Affine1 = tvm::te::placeholder({3, 3});
        auto dl1 = tvm::te::reduce_axis({0, 3});

        auto FC1 = tvm::te::compute(
            {m - 4, m - 4, 3},
            [=](auto x, auto y, auto z){ return tvm::sum(Relu2(x, y, dl1) * Affine1(z, dl1), {dl1}); }
        );

        auto Relu3 = topi::leaky_relu(FC1);

        auto Affine2 = tvm::te::placeholder({3, 3});
        auto dl2 = tvm::te::reduce_axis({0, 3});

        auto Output = tvm::te::compute(
            {m - 4, m - 4, 3},
            [=](auto x, auto y, auto z) { return tvm::sum(Relu3(x, y, dl2) * Affine2(z, dl2), {dl2}); }
        );

		tvm::te::Schedule s = tvm::te::create_schedule({Output->op});
		schs.push_back(s);
		bufs.push_back(tvm::Array<tvm::te::Tensor>({Input, Filter1, Filter2, Affine1, Affine2, Output}));
	}

    int threads;
    cin >> threads;
	auto start = std::chrono::steady_clock::now();

	BuildPool pool(threads);
	vector<future<tvm::runtime::Module>> funcs;

	for (int i = 0; i < m; ++i) {
		
		tvm::Target target = tvm::Target::Create("llvm");
		tvm::Target target_host = tvm::Target::Create("llvm");
        std::unordered_map<tvm::te::Tensor, tvm::tir::Buffer> binds;
        tvm::BuildConfig config = tvm::BuildConfig::Create();

		funcs.push_back(pool.push_back(schs[i], bufs[i], target, target_host, "compute", binds, config));
	}

    for (auto &func : funcs) {
        try {
            func.get();
        }
        catch(const exception& e) {
            cerr << e.what() << '\n';
        }
    }

	auto end = std::chrono::steady_clock::now();
    chrono::duration<double> elapsed_seconds = end-start;
    cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    return 0;
}