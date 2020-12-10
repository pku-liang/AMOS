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

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

// Vector saves batch, m, n, k, a_t, b_t
std::vector<std::tuple<int, int, int, int, bool, bool>> inference_server_set = {
      std::make_tuple(2, 32, 32, 32, false, false),
      std::make_tuple(2, 35, 700, 2048, false, false),
      std::make_tuple(2, 5124, 700, 2560, false, false)
};


/*
Usage:

The default precision is set based on the architecture and mode.

By default, the program runs the benchmark in training mode.

bin/gemm_bench

To run inference mode, use the following command:

bin/gemm_bench inference


To change the precision for training/inference, use:

bin/gemm_bench train <precision>
bin/gemm_bench inference <precision>

Supported precision types:

For Maxwell GPUS:
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

template <typename T1, typename T2>
int time_gemm(T1** A_array, T1** B_array, T2** C_array, bool a_t, bool b_t,
              cublasHandle_t cublas_handle, bool use_tensor_core, int batchCount, int m, int n, int k,
	      int lda, int ldb, int ldc) {
  //const int alpha = 1.f;
  //const int beta = 1.f;

  const T2 alpha = (T2)1.0f;
  const T2 beta = (T2)0.0f;

  //std::cout << std::endl << "m, k, n = " << m << "," << k << "," << n << std::endl;


  int numRepeats = 6;
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
      cublasGemmBatchedEx(cublas_handle, a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
			  b_t ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, (const void* const*)A_array,
			  A_type, lda, (const void* const*)(B_array),
			  B_type, ldb, &beta,
			  (void* const*)(C_array),
			  C_type, ldc, batchCount, compute_type, algo);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << "CUBLAS_STATUS_NOT_INITIALIZED = " << CUBLAS_STATUS_NOT_INITIALIZED << std::endl;
    std::cout << "CUBLAS_STATUS_ARCH_MISMATCH = " << CUBLAS_STATUS_ARCH_MISMATCH << std::endl;
    std::cout << "CUBLAS_STATUS_NOT_SUPPORTED = " << CUBLAS_STATUS_NOT_SUPPORTED << std::endl;
    std::cout << "CUBLAS_STATUS_INVALID_VALUE = " << CUBLAS_STATUS_INVALID_VALUE << std::endl;
    std::cout << "CUBLAS_STATUS_EXECUTION_FAILED = " << CUBLAS_STATUS_EXECUTION_FAILED << std::endl;
    std::cout <<  "stat = " << stat  << std::endl;
    throw std::runtime_error("sgemm failed, too bad!");
  }

  cudaDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < numRepeats; ++i) {
    stat = cublasGemmBatchedEx(cublas_handle, a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
			       b_t ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, (const void* const*)(A_array),
			       A_type, lda, (const void* const*)(B_array),
			       B_type, ldb, &beta,
			       (void* const*)(C_array),
			       C_type, ldc, batchCount, compute_type, algo);

    
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cout << "CUBLAS_STATUS_NOT_INITIALIZED = " << CUBLAS_STATUS_NOT_INITIALIZED << std::endl;
      std::cout << "CUBLAS_STATUS_ARCH_MISMATCH = " << CUBLAS_STATUS_ARCH_MISMATCH << std::endl;
      std::cout << "CUBLAS_STATUS_NOT_SUPPORTED = " << CUBLAS_STATUS_NOT_SUPPORTED << std::endl;
      std::cout << "CUBLAS_STATUS_INVALID_VALUE = " << CUBLAS_STATUS_INVALID_VALUE << std::endl;
      std::cout << "CUBLAS_STATUS_EXECUTION_FAILED = " << CUBLAS_STATUS_EXECUTION_FAILED << std::endl;
      std::cout <<  "stat = " << stat  << std::endl;
      throw std::runtime_error("sgemm failed, too bad!");
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
        << "b,m,n,k,a_t,b_t,fp32 time (usec),fp16 time (usec),int8 time "
           "(usec),fp16 tensor core time (usec),int8 tensor core time (usec)"
        << std::endl;

    int pad_kernels_count = 0;

    for (const auto &problem : inference_server_set) {
      int batch, m, n, k;
      bool a_t, b_t;
      std::tie(batch, m, n, k, a_t, b_t) = problem;
      int time_ms;

      std::cout << batch << ",";
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
        float** a; float** a_h; 
        float** b; float** b_h;
        float** c; float** c_h;
        std::tie(a, a_h) = rand<float>({a_t ? k : m, a_t ? m : k}, curand_gen, batch);
        std::tie(b, b_h) = rand<float>({b_t ? n : k, b_t ? k : n}, curand_gen, batch);
        std::tie(c, c_h) = zeros<float>({m, n}, batch);
        time_ms =
	        time_gemm<float, float>(a, b, c, a_t, b_t, cublas_handle, false, batch, m, n, k,
				  a_t? k : m, b_t? n : k, m);
        std::cout << "," << std::setprecision(6) << time_ms;
        for (int i = 0; i < batch; i++) {
          cudaFree(a_h[i]);
          cudaFree(b_h[i]);
          cudaFree(c_h[i]);
        }
        cudaFree(a_h);
        cudaFree(b_h);
        cudaFree(c_h);
      }

      // fp16 benchmark
      {
        uint16_t** a; uint16_t** a_h;
        uint16_t** b; uint16_t** b_h;
        uint16_t** c; uint16_t** c_h;
        std::tie(a, a_h) = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, curand_gen, batch);
        std::tie(b, b_h) = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, curand_gen, batch);
        std::tie(c, c_h) = zeros<uint16_t>({m, n}, batch);
        time_ms = time_gemm<uint16_t, uint16_t>(a, b, c, a_t, b_t,
                                                cublas_handle, false, batch, m, n, k,
						a_t? k : m, b_t? n : k, m);
        std::cout << "," << std::setprecision(6) << time_ms;
        for (int i = 0; i < batch; i++) {
          cudaFree(a_h[i]);
          cudaFree(b_h[i]);
          cudaFree(c_h[i]);
        }
        cudaFree(a_h);
        cudaFree(b_h);
        cudaFree(c_h);
      }
      
      // int8 benchmark
      std::cout << "," << "N/A";
      
      // {
      //   int pad_m;
      //   pad_m = m;
      //   if (pad_m % 4) {
      //     pad_kernels_count++;
      //     pad_dim(pad_m, 4);
      //   }
      //   uint8_t** a; uint8_t** a_h;
      //   uint8_t** b; uint8_t** b_h;
      //   int** c; int** c_h;
      //   std::tie(a, a_h) = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, curand_gen, batch);
      //   std::tie(b, b_h) = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, curand_gen, batch);
      //   std::tie(c, c_h) = zeros<int>({pad_m, n}, batch);
      //   time_ms =
	    //     time_gemm<uint8_t, int>(a, b, c, a_t, b_t, cublas_handle, false, batch, pad_m, n, k,
			// 	  a_t? k : pad_m, b_t? n : k, pad_m);
      //   std::cout << "," << std::setprecision(6) << time_ms;
      //   for (int i = 0; i < batch; i++) {
      //     cudaFree(a_h[i]);
      //     cudaFree(b_h[i]);
      //     cudaFree(c_h[i]);
      //   }
      //   cudaFree(a_h);
      //   cudaFree(b_h);
      //   cudaFree(c_h);
      // }
      

      // set cublas to use tensor core
      status = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS math mode failed" << std::endl;
      }

      // fp16 tensor core benchmark
      {
        uint16_t** a; uint16_t** a_h;
        uint16_t** b; uint16_t** b_h;
        uint16_t** c; uint16_t** c_h;
        std::tie(a, a_h) = rand<uint16_t>({a_t ? k : m, a_t ? m : k}, curand_gen, batch);
        std::tie(b, b_h) = rand<uint16_t>({b_t ? n : k, b_t ? k : n}, curand_gen, batch);
        std::tie(c, c_h) = zeros<uint16_t>({m, n}, batch);
        time_ms = time_gemm<uint16_t, uint16_t>(a, b, c, a_t, b_t,
                                                cublas_handle, true, batch, m, n, k,
						a_t? k : m, b_t? n : k, m);
        std::cout << "," << std::setprecision(6) << time_ms;
        for (int i = 0; i < batch; i++) {
          cudaFree(a_h[i]);
          cudaFree(b_h[i]);
          cudaFree(c_h[i]);
        }
        cudaFree(a_h);
        cudaFree(b_h);
        cudaFree(c_h);
      }

      // int8 tensor core benchmark
      std::cout << "," << "N/A";
      
  //     {
  //       int pad_m;
  //       pad_m = m;
  //       if (pad_m % 4) {
  //         pad_kernels_count++;
  //         pad_dim(pad_m, 4);
  //       }

  //       auto a = rand<uint8_t>({a_t ? k : pad_m, a_t ? pad_m : k}, curand_gen, batch);
  //       auto b = rand<uint8_t>({b_t ? n : k, b_t ? k : n}, curand_gen, batch);
  //       auto c = zeros<int>({pad_m, n}, batch);
	//       std::cout << "int8 tc" << std::endl;
  //       time_ms =
	//   time_gemm<uint8_t, int>(a, b, c, a_t, b_t, cublas_handle, true, batch, pad_m, n, k,
	// 			  a_t? k : pad_m, b_t? n : k, pad_m);
  //       std::cout << "," << std::setprecision(6) << time_ms;
  //     }
      

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
