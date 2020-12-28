#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <algorithm>

#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>

#include "tensor.h"
#include "configs.h"

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

template <typename T1, typename T2>
int time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t,
              cublasHandle_t cublas_handle, bool use_tensor_core) {
  const T2 alpha = 1.f;
  const T2 beta = 0.f;

  int m = C.dims()[0];
  int k = a_t ? A.dims()[0] : A.dims()[1];
  int n = C.dims()[1];

  int numRepeats = 20;
  cublasStatus_t stat;

  cudaDataType_t A_type = CUDA_R_32F;
  cudaDataType_t B_type = CUDA_R_32F;
  cudaDataType_t C_type = CUDA_R_32F;
  cudaDataType_t compute_type = CUDA_R_32F;
  cublasGemmAlgo_t algo;

  if (std::is_same<T1, uint16_t>::value) {
    A_type = CUDA_R_16F;
    B_type = CUDA_R_16F;
    C_type = CUDA_R_16F;
    compute_type = CUDA_R_16F;
  }

  if (std::is_same<T1, uint8_t>::value) {
    A_type = CUDA_R_8I;
    B_type = CUDA_R_8I;
    C_type = CUDA_R_32I;
    compute_type = CUDA_R_32I;
  }

  algo = use_tensor_core ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT;

  stat =
      cublasGemmEx(cublas_handle, a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                   b_t ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A.begin(),
                   A_type, A.dims()[0], B.begin(), B_type, B.dims()[0], &beta,
                   C.begin(), C_type, C.dims()[0], compute_type, algo);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("sgemm failed");
  }

  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numRepeats; ++i) {
    stat = cublasGemmEx(cublas_handle, a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                        b_t ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha,
                        A.begin(), A_type, A.dims()[0], B.begin(), B_type,
                        B.dims()[0], &beta, C.begin(), C_type, C.dims()[0],
                        compute_type, algo);

    if (stat != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("sgemm failed");
    }
  }
  cudaDeviceSynchronize();

  auto end = std::chrono::steady_clock::now();

  return static_cast<int>(
      std::chrono::duration<double, std::micro>(end - start).count() /
      numRepeats);
}

template<typename T1>
void rand_matrix(T1 *ptr, int size) {
  for (int i = 0; i < size; ++i)
    ptr[i] = i;
}

template <typename T1, typename T2>
bool check_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t,
              cublasHandle_t cublas_handle, bool use_tensor_core) {

  const T2 alpha = 1.f;
  const T2 beta = 0.f;

  int m = C.dims()[0];
  int k = a_t ? A.dims()[0] : A.dims()[1];
  int n = C.dims()[1];

  int lda = A.dims()[0], ldb = B.dims()[0], ldc = C.dims()[0];
  T1 *A_host, *B_host;
  T2 *C_host, *C_reference;

  A_host = (T1 *)malloc(A.size() * sizeof(T1));
  rand_matrix(A_host, A.size());
  cudaMemcpy(A.begin(), A_host, A.size() * sizeof(T1), cudaMemcpyHostToDevice);
  B_host = (T1 *)malloc(B.size() * sizeof(T1));
  rand_matrix(B_host, B.size());
  cudaMemcpy(B.begin(), B_host, B.size() * sizeof(T1), cudaMemcpyHostToDevice);
  C_reference = (T2 *)malloc(C.size() * sizeof(T2));
  cudaMemcpy(C_reference, C.begin(), C.size() * sizeof(T2), cudaMemcpyDeviceToHost);

  // int numRepeats = 6;
  cublasStatus_t stat;

  cudaDataType_t A_type = CUDA_R_32F;
  cudaDataType_t B_type = CUDA_R_32F;
  cudaDataType_t C_type = CUDA_R_32F;
  cudaDataType_t compute_type = CUDA_R_32F;
  cublasGemmAlgo_t algo;

  if (std::is_same<T1, uint16_t>::value) {
    A_type = CUDA_R_16F;
    B_type = CUDA_R_16F;
    C_type = CUDA_R_16F;
    compute_type = CUDA_R_16F;
  }

  if (std::is_same<T1, uint8_t>::value) {
    A_type = CUDA_R_8I;
    B_type = CUDA_R_8I;
    C_type = CUDA_R_32I;
    compute_type = CUDA_R_32I;
  }

  algo = use_tensor_core ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT;

  stat =
      cublasGemmEx(cublas_handle, a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                   b_t ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, A.begin(),
                   A_type, A.dims()[0], B.begin(), B_type, B.dims()[0], &beta,
                   C.begin(), C_type, C.dims()[0], compute_type, algo);
  
  C_host = (T2 *)malloc(C.size() * sizeof(T2));
  cudaMemcpy(C_host, C.begin(), C.size() * sizeof(T2), cudaMemcpyDeviceToHost);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error("sgemm failed");
  }
  
  for (int i = 0; i < m; ++i)
  for (int j = 0; j < n; ++j) {
    T2 accumulator = 0;
    for (int z = 0; z < k; ++z) {
      accumulator += A_host[i * k + z] * B_host[z * n + j];
    }
    C_reference[i * n + j] = alpha * accumulator + beta * C_reference[i * n + j];
  }

  float rtol = 1e-2, atol = 1e-2;
  for (int i = 0; i < m; ++i)
  for (int j = 0; j < n; ++j) {
    float C_diff = std::abs(C_host[i * n + j] - C_reference[i * n + j]);
    float C_min = std::min(std::abs(C_host[i * n + j]), std::abs(C_reference[i * n + j]));
    if (!(C_diff <= atol && C_diff / C_min <= rtol)) {
      std::cout << "\n(" << i << "," << j << "): atol=" << C_diff << ", rtol=" << C_diff/C_min << std::endl;
      std::cout << "-------" << std::endl;
      std::cout << C_host[i * n + j] << "," << C_reference[i * n + j] << std::endl;
      std ::cout << A.dims()[0] << "," << A.dims()[1] << std::endl;
      std ::cout << B.dims()[0] << "," << B.dims()[1] << std::endl;
      std ::cout << C.dims()[0] << "," << C.dims()[1] << std::endl;
      return false;
    }
  }
  return true;
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

  int inference = 1;
  if (argc > 1) {
    std::string inf = "inference";
    inference = argv[1] == inf ? 1 : 0;
  }

  if (inference) {
    std::cout << "Running inference benchmark " << std::endl;
  } else {
    std::cout << "Running training benchmark " << std::endl;
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

    std::cout
        << "m,n,k,a_t,b_t,fp32 time (usec),fp16 time (usec),int8 time "
           "(usec),fp16 tensor core time (usec),int8 tensor core time (usec)"
        << std::endl;

    int pad_kernels_count = 0;


    //--------------------------------------------
    int m = 32, n = 32, k = 16;
    bool a_t = false, b_t = false;
    auto a = rand<float>({a_t ? k : m, a_t ? m : k}, curand_gen);
    auto b = rand<float>({b_t ? n : k, b_t ? k : n}, curand_gen);
    auto c = zeros<float>({m, n});
    if (!check_gemm<float, float>(a, b, c, a_t, b_t, cublas_handle, false)) {
      std::cout << "gemm check fail!" << std::endl;
      return 0;
    }
    else std::cout << "gemm check pass!" << std::endl;

    //--------------------------------------------

    for (const auto &problem : gemm_set) {
      int m, n, k;
      bool a_t, b_t;
      std::tie(m, n, k, a_t, b_t) = problem;
      int time_ms;

      std::cout << m << ",";
      std::cout << n << ",";
      std::cout << k << ",";
      std::cout << "n"
                << ",";
      std::cout << "n";

      // set cublas to not use tensor core
      status = cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS math mode failed" << std::endl;
      }


      // fp32 benchmark
      {
        auto a = rand<float>({a_t ? k : m, a_t ? m : k}, curand_gen);
        auto b = rand<float>({b_t ? n : k, b_t ? k : n}, curand_gen);
        auto c = zeros<float>({m, n});
        time_ms =
            time_gemm<float, float>(a, b, c, a_t, b_t, cublas_handle, false);
        std::cout << "," << std::setprecision(6) << time_ms;
      }

      // fp16 benchmark
      {
        auto a = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, curand_gen);
        auto b = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
        auto c = zeros<uint16_t>({m, n});
        time_ms = time_gemm<uint16_t, uint16_t>(a, b, c, a_t, b_t,
                                                cublas_handle, false);
        std::cout << "," << std::setprecision(6) << time_ms;
      }

      // int8 benchmark
      {
        int pad_m;
        pad_m = m;
        if (pad_m % 4) {
          pad_kernels_count++;
          pad_dim(pad_m, 4);
        }

        auto a = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, curand_gen);
        auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
        auto c = zeros<int>({pad_m, n});
        time_ms =
            time_gemm<uint8_t, int>(a, b, c, a_t, b_t, cublas_handle, false);
        std::cout << "," << std::setprecision(6) << time_ms;
      }

      // set cublas to use tensor core
      status = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS math mode failed" << std::endl;
      }

      // fp16 tensor core benchmark
      {
        auto a = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, curand_gen);
        auto b = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
        auto c = zeros<uint16_t>({m, n});
        time_ms = time_gemm<uint16_t, uint16_t>(a, b, c, a_t, b_t,
                                                cublas_handle, true);
        std::cout << "," << std::setprecision(6) << time_ms;
      }

      // int8 tensor core benchmark
      {
        int pad_m;
        pad_m = m;
        if (pad_m % 4) {
          pad_kernels_count++;
          pad_dim(pad_m, 4);
        }

        auto a = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, curand_gen);
        auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, curand_gen);
        auto c = zeros<int>({pad_m, n});
        time_ms =
            time_gemm<uint8_t, int>(a, b, c, a_t, b_t, cublas_handle, true);
        std::cout << "," << std::setprecision(6) << time_ms;
      }

      // std::stringstream ss;
      // ss << "Unsupported precision requested. Precision: " << precision << "
      // Inference: " << inference;

      std::cout << std::endl;
    }

    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);
  }

  return 0;
}
