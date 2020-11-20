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

#ifndef PAD_KERNELS
#define PAD_KERNELS 1
#endif

// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride,
// hstride
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int>>
    inference_server_set = {
        std::make_tuple(700, 161, 1, 1, 32, 20, 5, 0, 0, 2, 2),
        std::make_tuple(700, 161, 1, 2, 32, 20, 5, 0, 0, 2, 2),
        std::make_tuple(700, 161, 1, 4, 32, 20, 5, 0, 0, 2, 2),
        std::make_tuple(341, 79, 32, 1, 32, 10, 5, 0, 0, 2, 2),
        std::make_tuple(341, 79, 32, 2, 32, 10, 5, 0, 0, 2, 2),
        std::make_tuple(341, 79, 32, 4, 32, 10, 5, 0, 0, 2, 2),
        std::make_tuple(480, 48, 1, 1, 16, 3, 3, 1, 1, 1, 1),
        std::make_tuple(240, 24, 16, 1, 32, 3, 3, 1, 1, 1, 1),
        std::make_tuple(120, 12, 32, 1, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(60, 6, 64, 1, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(108, 108, 3, 1, 64, 3, 3, 1, 1, 2, 2),
        std::make_tuple(54, 54, 64, 1, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(27, 27, 128, 1, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 128, 1, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(7, 7, 256, 1, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(112, 112, 64, 1, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 128, 1, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 256, 1, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(7, 7, 512, 1, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(224, 224, 3, 2, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(112, 112, 64, 2, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 128, 2, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 256, 2, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 512, 2, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(7, 7, 512, 2, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(224, 224, 3, 1, 64, 7, 7, 3, 3, 2, 2),
        std::make_tuple(28, 28, 192, 1, 32, 5, 5, 2, 2, 1, 1),
        std::make_tuple(28, 28, 192, 1, 64, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 512, 1, 48, 5, 5, 2, 2, 1, 1),
        std::make_tuple(14, 14, 512, 1, 192, 1, 1, 0, 0, 1, 1),
        std::make_tuple(7, 7, 832, 1, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(7, 7, 832, 1, 128, 5, 5, 2, 2, 1, 1),
        std::make_tuple(224, 224, 3, 2, 64, 7, 7, 3, 3, 2, 2),
        std::make_tuple(28, 28, 192, 2, 32, 5, 5, 2, 2, 1, 1),
        std::make_tuple(28, 28, 192, 2, 64, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 512, 2, 48, 5, 5, 2, 2, 1, 1),
        std::make_tuple(14, 14, 512, 2, 192, 1, 1, 0, 0, 1, 1),
        std::make_tuple(7, 7, 832, 2, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(7, 7, 832, 2, 128, 5, 5, 2, 2, 1, 1),
        std::make_tuple(56, 56, 64, 1, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 64, 1, 256, 1, 1, 0, 0, 2, 2),
        std::make_tuple(28, 28, 128, 1, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 128, 1, 512, 1, 1, 0, 0, 2, 2),
        std::make_tuple(14, 14, 256, 1, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 256, 1, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 256, 1, 1024, 1, 1, 0, 0, 2, 2),
        std::make_tuple(7, 7, 512, 1, 512, 1, 1, 0, 0, 1, 1),
        std::make_tuple(7, 7, 2048, 1, 512, 1, 1, 3, 3, 2, 2),
        std::make_tuple(56, 56, 64, 2, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(56, 56, 64, 2, 256, 1, 1, 0, 0, 2, 2),
        std::make_tuple(28, 28, 128, 2, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(28, 28, 128, 2, 512, 1, 1, 0, 0, 2, 2),
        std::make_tuple(14, 14, 256, 2, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 256, 2, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(14, 14, 256, 2, 1024, 1, 1, 0, 0, 2, 2),
        std::make_tuple(7, 7, 512, 2, 512, 1, 1, 0, 0, 1, 1),
        std::make_tuple(7, 7, 2048, 2, 512, 1, 1, 3, 3, 2, 2),
        std::make_tuple(700, 161, 1, 1, 64, 5, 5, 1, 1, 2, 2),
        std::make_tuple(350, 80, 64, 1, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(350, 80, 64, 1, 128, 5, 5, 1, 1, 2, 2),
        std::make_tuple(175, 40, 128, 1, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(175, 40, 128, 1, 256, 5, 5, 1, 1, 2, 2),
        std::make_tuple(84, 20, 256, 1, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(84, 20, 256, 1, 512, 5, 5, 1, 1, 2, 2),
        std::make_tuple(42, 10, 512, 1, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(700, 161, 1, 2, 64, 5, 5, 1, 1, 2, 2),
        std::make_tuple(350, 80, 64, 2, 64, 3, 3, 1, 1, 1, 1),
        std::make_tuple(350, 80, 64, 2, 128, 5, 5, 1, 1, 2, 2),
        std::make_tuple(175, 40, 128, 2, 128, 3, 3, 1, 1, 1, 1),
        std::make_tuple(175, 40, 128, 2, 256, 5, 5, 1, 1, 2, 2),
        std::make_tuple(84, 20, 256, 2, 256, 3, 3, 1, 1, 1, 1),
        std::make_tuple(84, 20, 256, 2, 512, 5, 5, 1, 1, 2, 2),
        std::make_tuple(42, 10, 512, 2, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(112, 112, 64, 1, 64, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 64, 1, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 256, 1, 64, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 256, 1, 128, 1, 1, 0, 0, 2, 2),
        std::make_tuple(28, 28, 128, 1, 512, 1, 1, 0, 0, 1, 1),
        std::make_tuple(28, 28, 512, 1, 128, 1, 1, 0, 0, 1, 1),
        std::make_tuple(28, 28, 512, 1, 256, 1, 1, 0, 0, 2, 2),
        std::make_tuple(14, 14, 256, 1, 1024, 1, 1, 0, 0, 1, 1),
        std::make_tuple(28, 28, 512, 1, 1024, 1, 1, 0, 0, 2, 2),
        std::make_tuple(14, 14, 1024, 1, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 256, 1, 1024, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 1024, 1, 512, 1, 1, 0, 0, 2, 2),
        std::make_tuple(7, 7, 512, 1, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(7, 7, 512, 1, 2048, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 1024, 1, 2048, 1, 1, 0, 0, 2, 2),
        std::make_tuple(7, 7, 2048, 1, 512, 1, 1, 0, 0, 1, 1),
        std::make_tuple(112, 112, 64, 2, 64, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 64, 2, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 256, 2, 64, 1, 1, 0, 0, 1, 1),
        std::make_tuple(56, 56, 256, 2, 128, 1, 1, 0, 0, 2, 2),
        std::make_tuple(28, 28, 128, 2, 512, 1, 1, 0, 0, 1, 1),
        std::make_tuple(28, 28, 512, 2, 128, 1, 1, 0, 0, 1, 1),
        std::make_tuple(28, 28, 512, 2, 256, 1, 1, 0, 0, 2, 2),
        std::make_tuple(14, 14, 256, 2, 1024, 1, 1, 0, 0, 1, 1),
        std::make_tuple(28, 28, 512, 2, 1024, 1, 1, 0, 0, 2, 2),
        std::make_tuple(14, 14, 1024, 2, 256, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 256, 2, 1024, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 1024, 2, 512, 1, 1, 0, 0, 2, 2),
        std::make_tuple(7, 7, 512, 2, 512, 3, 3, 1, 1, 1, 1),
        std::make_tuple(7, 7, 512, 2, 2048, 1, 1, 0, 0, 1, 1),
        std::make_tuple(14, 14, 1024, 2, 2048, 1, 1, 0, 0, 2, 2),
        std::make_tuple(7, 7, 2048, 2, 512, 1, 1, 0, 0, 1, 1)};

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

  const float alpha_ = 1.f;
  const float beta_ = 0.f;

  ConvolutionDescriptor<T2> conv_desc_;
  CudnnHandle cudnn_handle_;

public:
  cudnnCNN(int w, int h, int c, int n, int k, int r, int s, int pad_w,
           int pad_h, int wstride, int hstride, bool use_tensor_core)
      : cudnn_handle_(), conv_desc_(pad_h, pad_w, hstride, wstride) {
    int out_h, out_w, out_c, out_n;

    cudnnTensorFormat_t format;
    // For int8 inference, the supported format is NHWC
    if (std::is_same<T1, uint8_t>::value) {
      format = CUDNN_TENSOR_NHWC;
    } else {
      format = CUDNN_TENSOR_NCHW;
    }

    x_desc_ = TensorDescriptor4d<T1>(format, n, c, h, w);
    w_desc_ = FilterDescriptor4d<T1>(format, k, c, r, s);

    cudnnMathType_t algo =
        use_tensor_core ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;

    cudnnSetConvolutionMathType(conv_desc_.desc(), algo);
    // Get output dimensions
    CHECK_CUDNN_ERROR(cudnnGetConvolution2dForwardOutputDim(
        conv_desc_.desc(), x_desc_.desc(), w_desc_.desc(), &out_n, &out_c,
        &out_h, &out_w));

    h_desc_ = TensorDescriptor4d<T1>(format, out_n, out_c, out_h, out_w);

    output_dims_ = {out_w, out_h, out_c, out_n};

    // Pick forward convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t fwd_perf;
    int ret_count;

    if (std::is_same<T1, uint8_t>::value) {
      // Note: cuDNN only supports IMPLICIT_PRECOMP_GEMM for int8 data type.
      fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {
      CHECK_CUDNN_ERROR(cudnnFindConvolutionForwardAlgorithm(
          cudnn_handle_.handle(), x_desc_.desc(), w_desc_.desc(),
          conv_desc_.desc(), h_desc_.desc(), 1, &ret_count, &fwd_perf));
      fwd_algo_ = fwd_perf.algo;
    }

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
int time_cnn(int k, int c, int r, int s, int n, int h, int w, int pad_h,
             int pad_w, int hstride, int wstride, int num_repeats,
             curandGenerator_t curand_gen, bool use_tensor_core) {

  cudnnCNN<T1, T2> cnn(w, h, c, n, k, r, s, pad_w, pad_h, wstride, hstride,
                       use_tensor_core);

  // Allocate memory for filter
  auto filter = rand<T1>(std::vector<int>{s, r, c, k}, curand_gen);

  // Allocate memory for input
  auto input = rand<T1>(std::vector<int>{w, h, c, n}, curand_gen);

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
        << "w,h,c,n,k,f_w,f_h,,pad_w,pad_h,stride_w,stride_h,fp32 time "
           "(usec),fp16 time (usec),int8 time "
           "(usec),fp16 tensor core time (usec),int8 tensor core time (usec)"
        << std::endl;

    int pad_kernels_count = 0;

    for (const auto &problem : inference_server_set) {
      // Filter parameters
      int k, c, r, s; // r - filter_h (f_h), s - filter_w (f_w)
      // Input parameters
      int n, w, h;
      // Padding
      int pad_w, pad_h;
      // Stride
      int wstride, hstride;
      std::tie(w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride) = problem;

      int fwd_time;

      std::cout << w << ",";
      std::cout << h << ",";
      std::cout << c << ",";
      std::cout << n << ",";
      std::cout << k << ",";
      std::cout << s << ",";
      std::cout << r << ",";
      std::cout << pad_w << ",";
      std::cout << pad_h << ",";
      std::cout << wstride << ",";
      std::cout << hstride;

      // fp32 benchmark
      {
        int padded_c, padded_w, padded_h;

        padded_c = c;
        padded_h = h;
        padded_w = w;

        fwd_time = time_cnn<float, float>(
            k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride,
            wstride, num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // fp16 benchmark
      {
        int padded_c, padded_w, padded_h;

        padded_c = c;
        padded_h = h;
        padded_w = w;

        fwd_time = time_cnn<uint16_t, uint16_t>(
            k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride,
            wstride, num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // int8 benchmark
      {
        int padded_c, padded_w, padded_h;
        int pad_value;

        padded_c = c;
        padded_h = h;
        padded_w = w;

        pad_value = 4;
        if (c % pad_value || w % pad_value || h % pad_value) {
          pad_kernels_count++;
          pad_dim(padded_c, pad_value);
          pad_dim(padded_h, pad_value);
          pad_dim(padded_w, pad_value);
        }
        fwd_time = time_cnn<uint8_t, int>(
            k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride,
            wstride, num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // fp16 tensor core benchmark
      {
        int padded_c, padded_w, padded_h;

        padded_c = c;
        padded_h = h;
        padded_w = w;

        fwd_time = time_cnn<uint16_t, uint16_t>(
            k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride,
            wstride, num_repeats, curand_gen, true);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // int8 tensor core benchmark
      {
        int padded_c, padded_w, padded_h;
        int pad_value;

        padded_c = c;
        padded_h = h;
        padded_w = w;

        pad_value = 4;
        if (c % pad_value || w % pad_value || h % pad_value) {
          pad_kernels_count++;
          pad_dim(padded_c, pad_value);
          pad_dim(padded_h, pad_value);
          pad_dim(padded_w, pad_value);
        }
        fwd_time = time_cnn<uint8_t, int>(
            k, padded_c, r, s, n, padded_h, padded_w, pad_h, pad_w, hstride,
            wstride, num_repeats, curand_gen, true);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      std::cout << std::endl;
    }

    // Destroy all the handles
    curandDestroyGenerator(curand_gen);
  }

  return 0;
}