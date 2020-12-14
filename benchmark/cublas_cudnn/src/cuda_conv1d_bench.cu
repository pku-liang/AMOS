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

// T1 is used as the data type for inputs, weights and outputs.
// T2 is used to describe the compute precision. This is used in inference mode
// in the INT8_CONFIG
template <typename T1, typename T2> class cudnnCNN {
  TensorDescriptorNd<T1> x_desc_;
  TensorDescriptorNd<T1> h_desc_;

  FilterDescriptorNd<T1> w_desc_;

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
  cudnnCNN(int l, int c, int n, int k, int filter, int pad, int stride,
           bool use_tensor_core)
      : cudnn_handle_(), conv_desc_(pad, stride) {

    cudnnTensorFormat_t format;
    // For int8 inference, the supported format is NHWC
    if (std::is_same<T1, uint8_t>::value) {
      format = CUDNN_TENSOR_NHWC;
    } else {
      format = CUDNN_TENSOR_NCHW;
    }

    x_desc_ = TensorDescriptorNd<T1>(std::vector<int>{n, c, l});
    w_desc_ = FilterDescriptorNd<T1>(format, std::vector<int>{k, c, filter});

    cudnnMathType_t algo =
        use_tensor_core ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;

    cudnnSetConvolutionMathType(conv_desc_.desc(), algo);
    // Get output dimensions
    // outputDim = 1 + ( inputDim + 2*pad - (((filterDim-1)*dilation)+1) )/convolutionStride
    int tensorOuputDimA[] = {n, k, 1+(l + 2 * pad - (((filter-1)*1)+1))/stride};
    CHECK_CUDNN_ERROR(cudnnGetConvolutionNdForwardOutputDim(
        conv_desc_.desc(), x_desc_.desc(), w_desc_.desc(), 3, tensorOuputDimA));

    h_desc_ = TensorDescriptorNd<T1>(std::vector<int>{tensorOuputDimA[0], tensorOuputDimA[1], tensorOuputDimA[2]});

    output_dims_ = {tensorOuputDimA[0], tensorOuputDimA[1], tensorOuputDimA[2]};

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
    std::cout << "ret_cout = " << ret_count << std::endl;

    if (use_tensor_core) {
      // Tensor Op math only supports IMPLICIT_PRECOMP_GEMM algorithm
      fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
    fwd_workspace_size_ = 1073741824;
    // if (std::is_same<T1, uint8_t>::value) {
    //   // Note: cudnn workspace size function doesn't work for INT8_CONFIG
    //   fwd_workspace_size_ = 1073741824;
    // } else {
    //   // Set fwd workspace size
    //   CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
    //       cudnn_handle_.handle(), x_desc_.desc(), w_desc_.desc(),
    //       conv_desc_.desc(), h_desc_.desc(), fwd_algo_, &fwd_workspace_size_));
    // }

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
int time_cnn(int l, int c, int n, int k, int filter, int pad, int stride,
             int num_repeats, curandGenerator_t curand_gen, bool use_tensor_core) {

  cudnnCNN<T1, T2> cnn(l, c, n, k, filter, pad, stride,
                       use_tensor_core);

  // Allocate memory for weight
  auto weight = rand<T1>(std::vector<int>{k, c, filter}, curand_gen);

  // Allocate memory for input
  auto input = rand<T1>(std::vector<int>{n, c, l}, curand_gen);

  // Allocate memory for output tensor
  auto output = zeros<T1>(cnn.get_output_dims());

  // Warm up
  cnn.forward(input, weight, output);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < num_repeats; ++i) {
    cnn.forward(input, weight, output);
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
        << "l,c,n,k,filter,pad,stride,fp32 time "
           "(usec),fp16 time (usec),int8 time "
           "(usec),fp16 tensor core time (usec),int8 tensor core time (usec)"
        << std::endl;

    // int pad_kernels_count = 0;

    int batch = 1;

    for (const auto &problem : conv_1d) {
      // Filter parameters
      int k, c, l;
      // Input parameters
      int n, filter;
      // Padding
      int pad;
      // Stride
      int stride;
      std::tie(l, c, n, k, filter, pad, stride) = problem;
      n = batch;
      int fwd_time;

      std::cout << l << ",";
      std::cout << c << ",";
      std::cout << n << ",";
      std::cout << k << ",";
      std::cout << filter << ",";
      std::cout << pad << ",";
      std::cout << stride << ",";

      // fp32 benchmark
      {
        fwd_time = time_cnn<float, float>(l, c, n, k, filter, pad, stride, num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // fp16 benchmark
      {
        fwd_time = time_cnn<uint16_t, uint16_t>(l, c, n, k, filter, pad, stride, num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // int8 benchmark
      {
        fwd_time = time_cnn<uint8_t, int>(l, c, n, k, filter, pad, stride, num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // fp16 tensor core benchmark
      {
        fwd_time = time_cnn<uint16_t, uint16_t>(l, c, n, k, filter, pad, stride, num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      // int8 tensor core benchmark
      {
        fwd_time = time_cnn<uint8_t, int>(l, c, n, k, filter, pad, stride, num_repeats, curand_gen, false);
        std::cout << "," << std::setprecision(6) << fwd_time;
      }

      std::cout << std::endl;
    }

    // Destroy all the handles
    curandDestroyGenerator(curand_gen);
  }

  return 0;
}