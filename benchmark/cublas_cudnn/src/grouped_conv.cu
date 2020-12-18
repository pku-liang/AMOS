#include <chrono>
#include <iomanip>
#include <memory>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cudnn.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "cudnn_helper.h"
#include "tensor.h"
#include "configs.h"

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

// Vector saves b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int>>
    inference_server_set = {
        std::make_tuple(100, 50, 3, 64, 32, 32, 3, 3, 0, 0, 1, 1)
      };

/*
Usage:

The default precision is set based on the architecture and mode.

By default, the program runs the benchmark in training mode.

bin/conv_bench

To run inference mode, use the following command:

bin/conv_bench inference


To change the precision for training/inference, use:

bin/conv_bench train <precision>
bin/conv_bench inference <precision>

Supported precision types:

For Maxwell GPUS:
float for training and inference

For Pascal GPUS:
float, half for training
float, half, int8 for inference

*/

// T1 is used as the data type for inputs, weights and outputs.
// T2 is used to describe the compute precision. This is used in inference mode
// in the INT8_CONFIG
template <typename T1, typename T2> class cudnnCNN {
  TensorDescriptor4d<T1> x_desc_;
  TensorDescriptor4d<T1> h_desc_;

  FilterDescriptor4d<T1> w_desc_;

  std::vector<int> output_dims_;
  int num_repeats_;

  size_t fwd_workspace_size_;
  Tensor<float> fwd_workspace_;
  cudnnConvolutionFwdAlgo_t fwd_algo_;

  const T2 alpha_ = 1.f;
  const T2 beta_ = 0.f;

  ConvolutionDescriptor<T2> conv_desc_;
  CudnnHandle cudnn_handle_;

  public:
  cudnnCNN(int b_j, int b_i, int c_in, int c_out, int h, int w,
           int kh, int kw, int padh, int padw, int strideh, int stridew,
           bool use_tensor_core)
      : cudnn_handle_(), conv_desc_(padh, padw, strideh, stridew) {
    int outh, outw, outc, outn;

    CHECK_CUDNN_ERROR(cudnnSetConvolutionGroupCount(conv_desc_.desc(), b_i));

    cudnnTensorFormat_t format;
    // For int8 inference, the supported format is NHWC
    if (std::is_same<T1, uint8_t>::value) {
      format = CUDNN_TENSOR_NHWC;
    } else {
      format = CUDNN_TENSOR_NCHW;
    }

    x_desc_ = TensorDescriptor4d<T1>(format, b_j, b_i * c_in, h, w);
    w_desc_ = FilterDescriptor4d<T1>(format, b_i * c_out, c_in, kh, kw);

    cudnnMathType_t algo =
        use_tensor_core ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;

    cudnnSetConvolutionMathType(conv_desc_.desc(), algo);
    // Get output dimensions
    CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(
        conv_desc_.desc(), x_desc_.desc(), w_desc_.desc(), &outn, &outc,
        &outh, &outw));

    h_desc_ = TensorDescriptor4d<T1>(format, outn, outc, outh, outw);

    output_dims_ = {outw, outh, outc, outn};

    // Pick forward convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    int ret_count;

    CHECK_CUDNN_ERROR(cudnnFindConvolutionForwardAlgorithm(
        cudnn_handle_.handle(), x_desc_.desc(), w_desc_.desc(),
        conv_desc_.desc(), h_desc_.desc(), 1, &ret_count, &fwd_perf));
    fwd_algo_ = fwd_perf.algo;

    if (use_tensor_core) {
      // Tensor Op math only supports IMPLICIT_PRECOMP_GEMM algorithm
      fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
    if (std::is_same<T1, uint8_t>::value) {
      // Note: cudnn workspace size function doesn't work for INT8_CONFIG
      fwd_workspace_size_ = 1073741824;
    } else {
      // Set fwd workspace size
      CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
          cudnn_handle_.handle(), x_desc_.desc(), w_desc_.desc(),
          conv_desc_.desc(), h_desc_.desc(), fwd_algo_, &fwd_workspace_size_));
    }

    fwd_workspace_ = zeros<float>(std::vector<int>{
        static_cast<int>(fwd_workspace_size_ / sizeof(float)), 1});
  }

  std::vector<int> get_output_dims() { return output_dims_; }

  std::string get_fwd_algo_string() {
    if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
      return "IMPLICIT_GEMM";
    else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
      return "IMPLICIT_PRECOMP_GEMM";
    else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
      return "GEMM";
    else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
      return "DIRECT";
    else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
      return "FFT";
    else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
      return "FFT_TILING";
    else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
      return "WINOGRAD";
#if CUDNN_MAJOR >= 6
    else if (fwd_algo_ == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED)
      return "WINOGRAD_NONFUSED";
#endif
    else {
      std::stringstream ss;
      ss << "Illegal algorithm passed to get_fwd_algo_string. Algo: "
         << fwd_algo_ << std::endl;
      throw std::runtime_error(ss.str());
    }
  }

  void forward(Tensor<T1> x, Tensor<T1> filter, Tensor<T1> h) {
    // Convolution forward.
    CHECK_CUDNN_ERROR(cudnnConvolutionForward(
        cudnn_handle_.handle(), &alpha_, x_desc_.desc(), x.begin(),
        w_desc_.desc(), filter.begin(), conv_desc_.desc(), fwd_algo_,
        fwd_workspace_.begin(), fwd_workspace_size_, &beta_, h_desc_.desc(),
        h.begin()));
  }
};

template <typename T1, typename T2>
int time_cnn(int b_j, int b_i, int c_in, int c_out, int h, int w,
             int kh, int kw, int padh, int padw, int strideh, int stridew,
             int num_repeats, curandGenerator_t curand_gen, bool use_tensor_core) {

  cudnnCNN<T1, T2> cnn(b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew,
                       use_tensor_core);

  // Allocate memory for filter
  auto filter = rand<T1>(std::vector<int>{kw, kh, c_in, b_i * c_out}, curand_gen);

  // Allocate memory for input
  auto input = rand<T1>(std::vector<int>{w, h, b_i * c_in, b_j}, curand_gen);

  // Allocate memory for output tensor
  auto output = zeros<T1>(cnn.get_output_dims());

  // Warm up
  cnn.forward(input, filter, output);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < num_repeats; ++i) {
    cnn.forward(input, filter, output);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  int fwd_time = static_cast<int>(
      std::chrono::duration<double, std::micro>(end - start).count() /
      num_repeats);

  return fwd_time;
}

int main(int argc, char **argv) {
  int num_repeats = 20;

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

    std::cout
        << "b_j,b_i,c_in,c_out,h,w,kh,kw,padh,padw,strideh,stridew,fp32 time "
           "(usec),fp16 time (usec),int8 time "
           "(usec),fp16 tensor core time (usec),int8 tensor core time (usec)"
        << std::endl;

    int pad_kernels_count = 0;

    for (const auto &problem : inference_server_set) {
      int b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew;
      std::tie(b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew) = problem;
      // n = batch;
      int fwd_time;

      std::cout << b_j << ",";
      std::cout << b_i << ",";
      std::cout << c_in << ",";
      std::cout << c_out << ",";
      std::cout << h << ",";
      std::cout << w << ",";
      std::cout << kh << ",";
      std::cout << kw << ",";
      std::cout << padh << ",";
      std::cout << padw << ",";
      std::cout << strideh << ",";
      std::cout << stridew;

      // fp32 benchmark
      {

        fwd_time = time_cnn<float, float>(
            b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew,
            num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // fp16 benchmark
      {

        fwd_time = time_cnn<uint16_t, uint16_t>(
            b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew,
            num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // int8 benchmark
      {
        int pad_value;

        pad_value = 4;
        if (c_in % pad_value || w % pad_value || h % pad_value) {
          pad_kernels_count++;
          pad_dim(c_in, pad_value);
          pad_dim(h, pad_value);
          pad_dim(w, pad_value);
        }
        fwd_time = time_cnn<uint8_t, int>(
            b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew,
            num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // fp16 tensor core benchmark
      {

        fwd_time = time_cnn<uint16_t, uint16_t>(
            b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew,
            num_repeats, curand_gen, true);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // int8 tensor core benchmark
      {
        int pad_value;

        pad_value = 4;
        if (c_in % pad_value || w % pad_value || h % pad_value) {
          pad_kernels_count++;
          pad_dim(c_in, pad_value);
          pad_dim(h, pad_value);
          pad_dim(w, pad_value);
        }
        fwd_time = time_cnn<uint8_t, int>(
            b_j, b_i, c_in, c_out, h, w, kh, kw, padh, padw, strideh, stridew,
            num_repeats, curand_gen, true);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      std::cout << std::endl;
    }

    // Destroy all the handles
    curandDestroyGenerator(curand_gen);
  }

  return 0;
}