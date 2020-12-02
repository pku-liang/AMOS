#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#include "tensor.h"


// y = alpha * op ( A ) x + beta * y
// if not transpose:
// A shape: m X n, x: n, y: m

// Vector saves m, n, trans
std::vector<std::tuple<int, int, bool>> inference_server_set = {
    std::make_tuple(1024, 256, false),
    std::make_tuple(512, 1024, false)
    };


template <typename T1, typename T2>
int time_gemv(Tensor<T1> A, Tensor<T1> X, Tensor<T2> Y, int m, int n,
              bool trans, cublasHandle_t cublas_handle) {
  float alpha = 1.0;
  float beta = 1.0;

  int numRepeats = 6;
  cublasStatus_t stat;

  stat =
      cublasSgemv(cublas_handle, trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                   m, n, &alpha, A.begin(), A.dims()[0],
                   X.begin(), 1, &beta,
                   Y.begin(), 1);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("Sgemv failed");
  }

  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numRepeats; ++i) {
    stat = cublasSgemv(cublas_handle, trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                        m, n, &alpha, A.begin(), A.dims()[0],
                        X.begin(), 1, &beta,
                        Y.begin(), 1);

    if (stat != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Sgemv failed");
    }
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();

  return static_cast<int>(
      std::chrono::duration<double, std::micro>(end - start).count() /
      numRepeats);
}

int main(int argc, char **argv) {
  // Get Device Number
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  for (int dev = 0; dev < deviceCount; ++dev) {
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;

    curandGenerator_t curand_gen;
    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      std::cout << "CUBLAS init failed" << std::endl;
    }

    std::cout << "m,n,trans,fp32 time (usec)" << std::endl;

    for (const auto &problem : inference_server_set) {
      int m, n;
      bool trans;
      std::tie(m, n, trans) = problem;
      int time_ms;

      std::cout << m << ",";
      std::cout << n << ",";
      std::cout << trans ? "t" : "n" ;

      // set cublas to not use tensor core
      status = cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS math mode failed" << std::endl;
      }

      // fp32 benchmark
      {
        auto a = rand<float>({trans? n : m, trans? m : n}, curand_gen);
        auto x = rand<float>({trans ? m : n}, curand_gen);
        auto y = zeros<float>({trans ? n : m});
        time_ms =
            time_gemv<float, float>(a, x, y, m, n, trans, cublas_handle);
        std::cout << "," << std::setprecision(6) << time_ms;
      }

      std::cout << std::endl;

    }

    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);
  }

  return 0;
}
