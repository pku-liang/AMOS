#include <assert.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <malloc.h>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <cstring>

/*
 how to get time accurately: refer to https://www.cnblogs.com/dwdxdy/p/3214905.html
*/
#if defined (__i386__)
static __inline__ unsigned long long GetCycleCount(void)
{
    unsigned long long int x;
    __asm__ volatile ("rdtsc":"=A"(x));
    return x;
}
#elif defined (__x86_64__)
static __inline__ unsigned long long GetCycleCount(void)
{
    unsigned hi, lo;
    __asm__ volatile("rdtsc":"=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#endif

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      exit(EXIT_FAILURE);                                    \
    }                                                        \
  }


  #define checkCUDA(expression)                              \
  {                                                          \
    cudaError_t status = (expression);                       \
    if (status != cudaSuccess) {                             \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudaGetErrorString(status) << std::endl;  \
      exit(EXIT_FAILURE);                                    \
    }                                                        \
  }


/*
 To run this program:
 export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1:/usr/local/cuda-10.1/nvvm/bin${PATH:+:${PATH}} && 
 export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} &&
 make &&
 ./cudnn_conv
*/


cudnnConvolutionFwdAlgo_t conv_algorithms[] = {
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
  CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
  CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
  CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
  CUDNN_CONVOLUTION_FWD_ALGO_FFT,
  CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
  CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
};


const char* conv_algorithms_strings[] = {
  "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM",
  "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM",
  "CUDNN_CONVOLUTION_FWD_ALGO_GEMM",
  "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT",
  "CUDNN_CONVOLUTION_FWD_ALGO_FFT",
  "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING",
  "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD",
  "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"
};


size_t get_algorithm_choice_size() {
  return sizeof(conv_algorithms) / sizeof(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
}


std::string get_conv_algorithm_string(cudnnConvolutionFwdAlgo_t algo) {
  for (int i = 0; i < (int)(get_algorithm_choice_size()); ++i) {
    if (algo == conv_algorithms[i]) {
      return std::string(conv_algorithms_strings[i]);
    }
  }
  return "";
}


float holistic_conv(int C, int K, int H, int W, int batch_size, int kernel_size,
           int stride, int padding, std::string& chosen_algorithm,
           int times=1000, bool new_stream=false, int algorithm=-1)
{
    srand((unsigned)time(NULL));
    auto format = CUDNN_TENSOR_NHWC;

    cudaStream_t stream = 0;
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));
    if (new_stream) {
      checkCUDA(cudaStreamCreate(&stream));
      checkCUDNN(cudnnSetStream(cudnn, stream));
    }

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/C,
                                        /*image_height=*/H,
                                        /*image_width=*/W));
    cudnnTensorDescriptor_t output_descriptor;
    size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/format,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/batch_size,
                                        /*channels=*/K,
                                        /*image_height=*/H_out,
                                        /*image_width=*/W_out));
    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor, 
                                        /*dataType=*/CUDNN_DATA_FLOAT, 
                                        /*format=*/format, 
                                        /*out_channels=*/K, 
                                        /*in_channels=*/C, 
                                        /*kernel_height=*/kernel_size, 
                                        /*kernel_width=*/kernel_size));
    
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor, 
                                        /*pad_height=*/padding, 
                                        /*pad_width=*/padding, 
                                        /*vertical_stride=*/stride, 
                                        /*horizonal_stride=*/stride, 
                                        /*dilation_height=*/1, 
                                        /*dilation_width=*/1, 
                                        /*mode=*/CUDNN_CROSS_CORRELATION, 
                                        /*computeType=*/CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    int max_algorithm_count = 0;
    int returen_algorithm_count = 0;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &max_algorithm_count));
    cudnnConvolutionFwdAlgoPerf_t *perf_results = new cudnnConvolutionFwdAlgoPerf_t[max_algorithm_count];
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, 
              input_descriptor, 
              kernel_descriptor, 
              convolution_descriptor, 
              output_descriptor, 
              max_algorithm_count, 
              &returen_algorithm_count, 
              perf_results));
    // checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, 
    //                                               input_descriptor, 
    //                                               kernel_descriptor, 
    //                                               convolution_descriptor, 
    //                                               output_descriptor, 
    //                                               CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
    //                                               /*memoryLimitInBytes=*/0, 
    //                                               &convolution_algorithm));
    if (returen_algorithm_count <= 0) return 1e20;
    convolution_algorithm = perf_results[0].algo;
    chosen_algorithm = get_conv_algorithm_string(convolution_algorithm);
    
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, 
                                        input_descriptor, 
                                        kernel_descriptor,
                                        convolution_descriptor, 
                                        output_descriptor, 
                                        convolution_algorithm/*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/, 
                                        &workspace_bytes));
    std::cerr << "Workspace size: " << (float(workspace_bytes) / 1048576.0) << "MB" 
    << std::endl;
    void *d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);

    size_t image_bytes = batch_size * C * H * W * sizeof(float);

    float *d_input{nullptr};
    cudaMalloc(&d_input, image_bytes);
    float *h_input{nullptr};
    h_input = (float*)malloc(image_bytes);
    for(int i=0; i < batch_size * C * H * W; ++i)
    {
        *(h_input + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
    }
    cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice);

    size_t output_bytes = batch_size * K * H_out * W_out * sizeof(float);

    float *d_output{nullptr};
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);
    float *h_output{nullptr};
    h_output = (float*)malloc(output_bytes);

    size_t filter_bytes = K * C * kernel_size * kernel_size * sizeof(float);

    float *d_filter{nullptr};
    cudaMalloc(&d_filter, filter_bytes);
    float *h_filter{nullptr};
    h_filter = (float*)malloc(filter_bytes);
    for(int i=0; i < K * C * kernel_size * kernel_size; ++i)
    {
        *(h_filter + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
    }
    cudaMemcpy(d_filter, h_filter, filter_bytes, cudaMemcpyHostToDevice);
    const float alpha = 1, beta = 0;
    auto beg = (unsigned long long)GetCycleCount();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float sum = 0.0;
    for(int i = 0; i < times + 1; ++i)
    {
        cudaEventRecord(start, 0);
        checkCUDNN(cudnnConvolutionForward(cudnn, 
                                        &alpha, 
                                        input_descriptor, 
                                        d_input, 
                                        kernel_descriptor, 
                                        d_filter, 
                                        convolution_descriptor, 
                                        convolution_algorithm/*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/, 
                                        d_workspace, 
                                        workspace_bytes, 
                                        &beta, 
                                        output_descriptor, 
                                        d_output));
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        if (i > 0)
        {
            sum += elapsed;
        }
    }
    auto end = (unsigned long long)GetCycleCount();
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
    free(h_input);
    free(h_filter);
    free(h_output);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_filter);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);

    return sum;//float(end - beg);
}


float two_stream_conv_split_K(int C, int K, int H, int W, int batch_size, int kernel_size,
  int stride, int padding, std::string& chosen_algorithm1, std::string& chosen_algorithm2,
  int times=1000, float split_ratio=0.5, int algorithm1=-1, int algorithm2=-1)
{
  srand((unsigned)time(NULL));
  auto format = CUDNN_TENSOR_NHWC;

  cudaStream_t stream1 = 0;
  cudaStream_t stream2 = 0;
  cudnnHandle_t cudnn;
  checkCUDNN(cudnnCreate(&cudnn));

  checkCUDA(cudaStreamCreate(&stream1));
  checkCUDA(cudaStreamCreate(&stream2));

  if (split_ratio < 0.1) split_ratio = 0.1;
  if (split_ratio > 0.9) split_ratio = 0.9;

  int K1 = (int)(K * split_ratio);
  int K2 = K - K1;
  std::cout << "Split two convolution via output channel, K1=" << K1 << ", K2=" << K2 << "\n";
  if (K1 <= 0 || K2 <= 0) {
    std::cout << "Please choose other input scale and split ratio.";
    abort();
  }

  // prepare the first convolution
  checkCUDNN(cudnnSetStream(cudnn, stream1));

  cudnnTensorDescriptor_t input_descriptor1;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor1));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor1,
                                /*format=*/format,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*batch_size=*/batch_size,
                                /*channels=*/C,
                                /*image_height=*/H,
                                /*image_width=*/W));
  cudnnTensorDescriptor_t output_descriptor1;
  size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
  size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor1));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor1,
                                /*format=*/format,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*batch_size=*/batch_size,
                                /*channels=*/K1,
                                /*image_height=*/H_out,
                                /*image_width=*/W_out));
  cudnnFilterDescriptor_t kernel_descriptor1;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor1));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor1, 
                                /*dataType=*/CUDNN_DATA_FLOAT, 
                                /*format=*/format, 
                                /*out_channels=*/K1, 
                                /*in_channels=*/C, 
                                /*kernel_height=*/kernel_size, 
                                /*kernel_width=*/kernel_size));

  cudnnConvolutionDescriptor_t convolution_descriptor1;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor1));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor1, 
                                /*pad_height=*/padding, 
                                /*pad_width=*/padding, 
                                /*vertical_stride=*/stride, 
                                /*horizonal_stride=*/stride, 
                                /*dilation_height=*/1, 
                                /*dilation_width=*/1, 
                                /*mode=*/CUDNN_CROSS_CORRELATION, 
                                /*computeType=*/CUDNN_DATA_FLOAT));

  cudnnConvolutionFwdAlgo_t convolution_algorithm1;
  int max_algorithm_count1 = 0;
  int returen_algorithm_count1 = 0;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &max_algorithm_count1));
  cudnnConvolutionFwdAlgoPerf_t *perf_results1 = new cudnnConvolutionFwdAlgoPerf_t[max_algorithm_count1];
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, 
            input_descriptor1, 
            kernel_descriptor1, 
            convolution_descriptor1, 
            output_descriptor1, 
            max_algorithm_count1, 
            &returen_algorithm_count1, 
            perf_results1));
  if (returen_algorithm_count1 <= 0) return 1e20;
  convolution_algorithm1 = perf_results1[0].algo;
  // if (algorithm1 < 0 || algorithm1 >= (int)get_algorithm_choice_size()) {
    // checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, 
    //                                           input_descriptor1, 
    //                                           kernel_descriptor1, 
    //                                           convolution_descriptor1, 
    //                                           output_descriptor1, 
    //                                           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
    //                                           /*memoryLimitInBytes=*/0, 
    //                                           &convolution_algorithm1));
  // } else {
  //   convolution_algorithm1 = conv_algorithms[algorithm1];
  // }
  chosen_algorithm1 = get_conv_algorithm_string(convolution_algorithm1);

  size_t workspace_bytes1 = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, 
                                input_descriptor1, 
                                kernel_descriptor1, 
                                convolution_descriptor1, 
                                output_descriptor1, 
                                convolution_algorithm1/*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/, 
                                &workspace_bytes1));
  std::cerr << "Convolution 1 workspace size: " << (float(workspace_bytes1) / 1048576.0) << "MB" 
  << std::endl;
  void *d_workspace1{nullptr};
  cudaMalloc(&d_workspace1, workspace_bytes1);

  size_t image_bytes1 = batch_size * C * H * W * sizeof(float);

  float *d_input1{nullptr};
  cudaMalloc(&d_input1, image_bytes1);
  float *h_input1{nullptr};
  h_input1 = (float*)malloc(image_bytes1);
  for(int i=0; i < batch_size * C * H * W; ++i)
  {
    *(h_input1 + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
  }
  cudaMemcpy(d_input1, h_input1, image_bytes1, cudaMemcpyHostToDevice);

  size_t output_bytes1 = batch_size * K1 * H_out * W_out * sizeof(float);

  float *d_output1{nullptr};
  cudaMalloc(&d_output1, output_bytes1);
  cudaMemset(d_output1, 0, output_bytes1);
  float *h_output1{nullptr};
  h_output1 = (float*)malloc(output_bytes1);

  size_t filter_bytes1 = K1 * C * kernel_size * kernel_size * sizeof(float);

  float *d_filter1{nullptr};
  cudaMalloc(&d_filter1, filter_bytes1);
  float *h_filter1{nullptr};
  h_filter1 = (float*)malloc(filter_bytes1);
  for(int i=0; i < K1 * C * kernel_size * kernel_size; ++i)
  {
    *(h_filter1 + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
  }
  cudaMemcpy(d_filter1, h_filter1, filter_bytes1, cudaMemcpyHostToDevice);

  // prepare the second convolution
  checkCUDNN(cudnnSetStream(cudnn, stream2));

  cudnnTensorDescriptor_t input_descriptor2;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor2));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor2,
                                /*format=*/format,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*batch_size=*/batch_size,
                                /*channels=*/C,
                                /*image_height=*/H,
                                /*image_width=*/W));
  cudnnTensorDescriptor_t output_descriptor2;
  // size_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
  // size_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor2));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor2,
                                /*format=*/format,
                                /*dataType=*/CUDNN_DATA_FLOAT,
                                /*batch_size=*/batch_size,
                                /*channels=*/K2,
                                /*image_height=*/H_out,
                                /*image_width=*/W_out));
  cudnnFilterDescriptor_t kernel_descriptor2;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor2));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor2, 
                                /*dataType=*/CUDNN_DATA_FLOAT, 
                                /*format=*/format, 
                                /*out_channels=*/K2, 
                                /*in_channels=*/C, 
                                /*kernel_height=*/kernel_size, 
                                /*kernel_width=*/kernel_size));

  cudnnConvolutionDescriptor_t convolution_descriptor2;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor2));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor2, 
                                /*pad_height=*/padding, 
                                /*pad_width=*/padding, 
                                /*vertical_stride=*/stride, 
                                /*horizonal_stride=*/stride, 
                                /*dilation_height=*/1, 
                                /*dilation_width=*/1, 
                                /*mode=*/CUDNN_CROSS_CORRELATION, 
                                /*computeType=*/CUDNN_DATA_FLOAT));

  cudnnConvolutionFwdAlgo_t convolution_algorithm2;
  int max_algorithm_count2 = 0;
  int returen_algorithm_count2 = 0;
  checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &max_algorithm_count2));
  cudnnConvolutionFwdAlgoPerf_t *perf_results2 = new cudnnConvolutionFwdAlgoPerf_t[max_algorithm_count2];
  checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, 
            input_descriptor2, 
            kernel_descriptor2, 
            convolution_descriptor2, 
            output_descriptor2, 
            max_algorithm_count2, 
            &returen_algorithm_count2, 
            perf_results2));
  if (returen_algorithm_count2 <= 0) return 1e20;
  convolution_algorithm2 = perf_results2[0].algo;
  // if (algorithm2 < 0 || algorithm2 >= (int)get_algorithm_choice_size()) {
    // checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn, 
    //                                           input_descriptor2, 
    //                                           kernel_descriptor2, 
    //                                           convolution_descriptor2, 
    //                                           output_descriptor2, 
    //                                           CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 
    //                                           /*memoryLimitInBytes=*/0, 
    //                                           &convolution_algorithm2));
  // } else {
  //   convolution_algorithm2 = conv_algorithms[algorithm2];
  // }
  chosen_algorithm2 = get_conv_algorithm_string(convolution_algorithm2);

  size_t workspace_bytes2 = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, 
                                input_descriptor2, 
                                kernel_descriptor2, 
                                convolution_descriptor2, 
                                output_descriptor2, 
                                convolution_algorithm2/*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/, 
                                &workspace_bytes2));
  std::cerr << "Convolution 2 workspace size: " << (float(workspace_bytes2) / 1048576.0) << "MB" 
  << std::endl;
  void *d_workspace2{nullptr};
  cudaMalloc(&d_workspace2, workspace_bytes2);

  size_t image_bytes2 = batch_size * C * H * W * sizeof(float);

  float *d_input2{nullptr};
  cudaMalloc(&d_input2, image_bytes2);
  float *h_input2{nullptr};
  h_input2 = (float*)malloc(image_bytes2);
  for(int i=0; i < batch_size * C * H * W; ++i)
  {
    *(h_input2 + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
  }
  cudaMemcpy(d_input2, h_input2, image_bytes2, cudaMemcpyHostToDevice);

  size_t output_bytes2 = batch_size * K2 * H_out * W_out * sizeof(float);

  float *d_output2{nullptr};
  cudaMalloc(&d_output2, output_bytes2);
  cudaMemset(d_output2, 0, output_bytes2);
  float *h_output2{nullptr};
  h_output2 = (float*)malloc(output_bytes2);

  size_t filter_bytes2 = K2 * C * kernel_size * kernel_size * sizeof(float);

  float *d_filter2{nullptr};
  cudaMalloc(&d_filter2, filter_bytes2);
  float *h_filter2{nullptr};
  h_filter2 = (float*)malloc(filter_bytes2);
  for(int i=0; i < K2 * C * kernel_size * kernel_size; ++i)
  {
    *(h_filter2 + i) = (float(rand()) - (RAND_MAX >> 1)) / RAND_MAX;
  }
  cudaMemcpy(d_filter2, h_filter2, filter_bytes2, cudaMemcpyHostToDevice);

  // launch two convolutions
  const float alpha = 1, beta = 0;
  auto beg = (unsigned long long)GetCycleCount();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float sum = 0.0;
  for(int i = 0; i < times + 1; ++i)
  {
    cudaEventRecord(start, 0);
    checkCUDNN(cudnnSetStream(cudnn, stream1));
    checkCUDNN(cudnnConvolutionForward(cudnn, 
                                  &alpha, 
                                  input_descriptor1, 
                                  d_input1, 
                                  kernel_descriptor1, 
                                  d_filter1, 
                                  convolution_descriptor1, 
                                  convolution_algorithm1/*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/, 
                                  d_workspace1, 
                                  workspace_bytes1, 
                                  &beta, 
                                  output_descriptor1, 
                                  d_output1));
    checkCUDNN(cudnnSetStream(cudnn, stream2));
    checkCUDNN(cudnnConvolutionForward(cudnn, 
                                  &alpha, 
                                  input_descriptor2, 
                                  d_input2, 
                                  kernel_descriptor2, 
                                  d_filter2, 
                                  convolution_descriptor2, 
                                  convolution_algorithm2/*CUDNN_CONVOLUTION_FWD_ALGO_DIRECT*/, 
                                  d_workspace2, 
                                  workspace_bytes2, 
                                  &beta, 
                                  output_descriptor2, 
                                  d_output2));
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    if (i > 0)
    {
      sum += elapsed;
    }
  }
  auto end = (unsigned long long)GetCycleCount();

  checkCUDNN(cudnnSetStream(cudnn, stream1));
  cudaMemcpy(h_output1, d_output1, output_bytes1, cudaMemcpyDeviceToHost);
  free(h_input1);
  free(h_filter1);
  free(h_output1);
  cudaFree(d_input1);
  cudaFree(d_output1);
  cudaFree(d_filter1);
  cudaFree(d_workspace1);
  cudnnDestroyTensorDescriptor(input_descriptor1);
  cudnnDestroyTensorDescriptor(output_descriptor1);
  cudnnDestroyFilterDescriptor(kernel_descriptor1);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor1);

  checkCUDNN(cudnnSetStream(cudnn, stream2));
  cudaMemcpy(h_output2, d_output2, output_bytes2, cudaMemcpyDeviceToHost);
  free(h_input2);
  free(h_filter2);
  free(h_output2);
  cudaFree(d_input2);
  cudaFree(d_output2);
  cudaFree(d_filter2);
  cudaFree(d_workspace2);
  cudnnDestroyTensorDescriptor(input_descriptor2);
  cudnnDestroyTensorDescriptor(output_descriptor2);
  cudnnDestroyFilterDescriptor(kernel_descriptor2);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor2);

  cudnnDestroy(cudnn);

  return sum;//float(end - beg);
}


void help() {
  std::cout << "Arguments:\n"
            << "times: int, >0\n"
            << "try all the algorithms: y/n\n"
            << "run mode: holistic/split-K\n"
            << "batch size: int, >0\n";
}


int arg_lst[][8] = {
  //{256, 256, 14, 14, 3, 512, 1, 1},
  // {1, 1024, 7, 7, 3, 1024, 1, 1},
  // {8, 1024, 7, 7, 3, 1024, 1, 1},
  // {64, 1024, 7, 7, 3, 1024, 1, 1},
  // {256, 1024, 7, 7, 3, 1024, 1, 1},
  // {1, 1024, 14, 14, 1, 512, 1, 0},
  // {1, 256, 28, 28, 3, 512, 1, 1},
  // {1, 512, 28, 28, 1, 256, 1, 0},
  // {1, 128, 56, 56, 3, 256, 1, 1},
  // {1, 192, 56, 56, 1, 128, 1, 0},
  // {1, 64, 112, 112, 3, 192, 1, 1},
  // {1, 3, 448, 448, 7, 64, 2, 3}
  {1, 3, 448, 448, 7, 64, 2, 3}, 
  {1, 64, 112, 112, 3, 192, 1, 1},  
  {1, 192, 56, 56, 1, 128, 1, 0},  
  {1, 128, 56, 56, 3, 256, 1, 1}, 
  {1, 256, 56, 56, 1, 256, 1, 0},
  {1, 256, 56, 56, 3, 512, 1, 1}, 
  {1, 512, 28, 28, 1, 256, 1, 0},  
  {1, 256, 28, 28, 3, 512, 1, 1}, 
  {1, 512, 28, 28, 1, 512, 1, 0},  // conv15      8
  {1, 512, 28, 28, 3, 1024, 1, 1},  // conv16     9
  {1, 1024, 14, 14, 1, 512, 1, 0},  // conv17    10
  {1, 512, 14, 14, 3, 1024, 1, 1},  // conv18     11
  {1, 1024, 14, 14, 3, 1024, 1, 1},  // conv21   12
  {1, 1024, 14, 14, 3, 1024, 2, 1}, // conv22   13
  {1, 1024, 7, 7, 3, 1024, 1, 1},  // conv23     14
};


int run_all_batch(int times, int algorithm_end)
{    // this is hard coded for dumping log
  #define BATCH_TRIAL 9
  #define BATCH_INIT 1
  float scoretable[15][BATCH_TRIAL + 1][2] = {0};

  float ratio_choices[] = {1.0/8, 2.0/8, 3.0/8, 4.0/8, 5.0/8, 6.0/8, 7.0/8};
  int num_ratio_choices = sizeof(ratio_choices) / sizeof(float);
  int batch_size = BATCH_INIT;
  for (int b = 0; b <= BATCH_TRIAL; b += 1) {
    for(int i=0; i < 15; ++i)
    {
      int C = arg_lst[i][1];
      int H = arg_lst[i][2];
      int W = arg_lst[i][3];
      int kernel_size = arg_lst[i][4];
      int K = arg_lst[i][5];
      int stride = arg_lst[i][6];
      int padding = arg_lst[i][7];
      std::cout << "shape is (" << batch_size << "," << H << "," << W << "," << C << ","
                      << kernel_size << "," << K << "," << stride << "," << padding << ")\n";
      // run mode 0
      std::cout << "run holistic convolution kernel...\n";
      std::string best_algo = "none";
      float best_perf = 1e10;
      for (int algo = -1; algo < algorithm_end; ++algo) {
        usleep(10);
        std::string algorithm{"none"};
        float cost = 1e20;
        try {
          cost = holistic_conv(C, K, H, W, batch_size, kernel_size, stride, padding, algorithm, times, false, algo);
        } catch (const std::exception& e) {
          std::cout << "Can't run...\n";
          cost = 1e20;
        }
        if (algo < 0) {
          std::cout << "algorithm is chosen by cudnn: " << algorithm << " "
                    << "Use time " << cost / times << "ms" << std::endl;
        } else {
          std::cout << "algorithm is assigned: " << std::string(conv_algorithms_strings[algo])
                    << ", chosen one is: " << algorithm << " "
                    << " Use time " << cost / times << "ms" << std::endl;
        }
        if (cost / times < best_perf) {
          best_perf = cost / times;
          best_algo = algorithm;
        }
      }
      std::cout << "[Best]: " << best_algo << ", " << best_perf << "ms\n";
      scoretable[i][b][0] = best_perf;
      
      // run mode 1
      std::cout << "run two stream convolution split by K...\n";
      std::string best_algo1 = "none", best_algo2 = "none";
      float best_ratio = -1;
      best_perf = 1e10;
      for (int ratio_choice = 0; ratio_choice < num_ratio_choices; ++ratio_choice) {
        for (int algo1 = -1; algo1 < algorithm_end; ++algo1) {
          for (int algo2 = -1; algo2 < algorithm_end; ++algo2) {
            usleep(10);
            std::string algorithm1{"none"}, algorithm2{"none"};
            float cost = 1e20;
            try {
              cost = two_stream_conv_split_K(
                C, K, H, W, batch_size, kernel_size, stride, padding, algorithm1, algorithm2, times,
                ratio_choices[ratio_choice], algo1, algo2);
            } catch (const std::exception& e) {
              std::cout << "Can't run...\n";
              cost = 1e20;
            }
            if (algo1 < 0) {
              std::cout << "algorithm 1 is chosen by cudnn: " << algorithm1 << "\n";
            } else {
              std::cout << "algorithm 1 is assigned: " << std::string(conv_algorithms_strings[algo1])
                        << ", chosen one is: " << algorithm1 << "\n";
            }
            if (algo2 < 0) {
              std::cout << "algorithm 2 is chosen by cudnn: " << algorithm2 << "\n";
            } else {
              std::cout << "algorithm 2 is assigned: " << std::string(conv_algorithms_strings[algo2])
                        << ", chosen one is: " << algorithm2 << "\n";
            }
            std::cout << "Use time " << cost / times << "ms" << std::endl;
            if (cost / times < best_perf) {
              best_perf = cost / times;
              best_algo1 = algorithm1;
              best_algo2 = algorithm2;
              best_ratio = ratio_choices[ratio_choice];
            }
          }
        }
      }
      std::cout << "[Best]: " << best_algo1 << ", " << best_algo2 << ", " << best_ratio << ", " << best_perf << "ms\n";
      scoretable[i][b][1] = best_perf;

      std::cout << "\n";
    }

    batch_size *= 2;
  }

  // dump scoretable
  std::ofstream fout("result.csv", std::ios::out);
  int bb = BATCH_INIT;
  fout << "type,batch,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15\n";
  for (int b = 0; b <= BATCH_TRIAL; ++b) {
    fout << "holistic" << "," << bb << ",";
    for (int i = 0; i < 15; ++i) {
      fout << scoretable[i][b][0] << ((i == 14) ? "" : ",");
    }
    fout << "\n";
    fout << "two-stream" << "," << bb << ",";
    for (int i = 0; i < 15; ++i) {
      fout << scoretable[i][b][1] << ((i == 14) ? "" : ",");
    }
    fout << "\n";
    bb *= 2;
  }
  fout.close();
  return 0;
}


int main(int argc, char const* argv[])
{
  float ratio_choices[] = {1.0/8, 2.0/8, 3.0/8, 4.0/8, 5.0/8, 6.0/8, 7.0/8};
  int num_ratio_choices = sizeof(ratio_choices) / sizeof(float);

  int times = 10;
  int algorithm_end = 0; // (int)get_algorithm_choice_size();
  int run_mode = 0; // 0: holistic, 1: two stream split K, 2: two stream split batch
  int batch_size = 1;
  bool try_all_batch = false;
  if(argc > 1)
  {
    if (argc >= 2) {
      if (strcmp(argv[1], "--help")==0) {
        help();
        return 0;
      }
      times = std::atoi(argv[1]);
    }
    if (argc >= 3) {
      if(strcmp(argv[2], "n")==0)
      {
        algorithm_end = 0; // only set cudnn choose its best
      }
    }
    if (argc >= 4) {
      if (strcmp(argv[3], "holistic")==0)
      {
        run_mode = 0;
      } else if (strcmp(argv[3], "split-K")==0) {
        run_mode = 1;
      } else if (strcmp(argv[3], "split-N")==0) {
        run_mode = 2;
      }
    }
    if (argc >= 5) {
      if (strcmp(argv[4], "all")==0) {
        try_all_batch = true;
      } else {
        batch_size = std::atoi(argv[4]);
      }
    }
  }

  std::cout << "measure repeat times=" << times << std::endl;
  std::cout << "algorithm end=" << algorithm_end << std::endl;
  std::cout << "run mode=" << run_mode << std::endl;

  if (try_all_batch) {
    std::cout << "try all batch size from 1 to 1024...\n";
    run_all_batch(times, algorithm_end);
    return 0;
  } else {
    std::cout << "batch size=" << batch_size << std::endl;

    for(int i=0; i < 15; ++i)
    {
      // int batch_size = 16; // arg_lst[i][0];
      std::cout << "####################################################\n";
      int C = arg_lst[i][1];
      int H = arg_lst[i][2];
      int W = arg_lst[i][3];
      int kernel_size = arg_lst[i][4];
      int K = arg_lst[i][5];
      int stride = arg_lst[i][6];
      int padding = arg_lst[i][7];
      std::cout << "shape is (" << batch_size << "," << H << "," << W << "," << C << ","
                      << kernel_size << "," << K << "," << stride << "," << padding << ")\n";
      
      if (run_mode == 0) {
        std::cout << "run holistic convolution kernel...\n";
        std::string best_algo = "none";
        float best_perf = 1e10;
        for (int algo = -1; algo < algorithm_end; ++algo) {
          usleep(10);
          std::string algorithm{"none"};
          float cost = 1e20;
          try {
            cost = holistic_conv(C, K, H, W, batch_size, kernel_size, stride, padding, algorithm, times, false, algo);
          } catch (const std::exception& e) {
            std::cout << "Can't run...\n";
            cost = 1e20;
          }
          if (algo < 0) {
            std::cout << "algorithm is chosen by cudnn: " << algorithm << " "
                      << "Use time " << cost / times << "ms" << std::endl;
          } else {
            std::cout << "algorithm is assigned: " << std::string(conv_algorithms_strings[algo])
                      << ", chosen one is: " << algorithm << " "
                      << " Use time " << cost / times << "ms" << std::endl;
          }
          if (cost / times < best_perf) {
            best_perf = cost / times;
            best_algo = algorithm;
          }
        }
        std::cout << "[Best]: " << best_algo << ", " << best_perf << "ms\n";
      } else if (run_mode == 1) {
        std::cout << "run two stream convolution split by K...\n";
        std::string best_algo1 = "none", best_algo2 = "none";
        float best_ratio = -1;
        float best_perf = 1e10;
        for (int ratio_choice = 0; ratio_choice < num_ratio_choices; ++ratio_choice) {
          for (int algo1 = -1; algo1 < algorithm_end; ++algo1) {
            for (int algo2 = -1; algo2 < algorithm_end; ++algo2) {
              usleep(10);
              std::string algorithm1{"none"}, algorithm2{"none"};
              float cost = 1e20;
              try {
                cost = two_stream_conv_split_K(
                  C, K, H, W, batch_size, kernel_size, stride, padding, algorithm1, algorithm2, times,
                  ratio_choices[ratio_choice], algo1, algo2);
              } catch (const std::exception& e) {
                std::cout << "Can't run...\n";
                cost = 1e20;
              }
              if (algo1 < 0) {
                std::cout << "algorithm 1 is chosen by cudnn: " << algorithm1 << "\n";
              } else {
                std::cout << "algorithm 1 is assigned: " << std::string(conv_algorithms_strings[algo1])
                          << ", chosen one is: " << algorithm1 << "\n";
              }
              if (algo2 < 0) {
                std::cout << "algorithm 2 is chosen by cudnn: " << algorithm2 << "\n";
              } else {
                std::cout << "algorithm 2 is assigned: " << std::string(conv_algorithms_strings[algo2])
                          << ", chosen one is: " << algorithm2 << "\n";
              }
              std::cout << "Use time " << cost / times << "ms" << std::endl;
              if (cost / times < best_perf) {
                best_perf = cost / times;
                best_algo1 = algorithm1;
                best_algo2 = algorithm2;
                best_ratio = ratio_choices[ratio_choice];
              }
            }
          }
        }
        std::cout << "[Best]: " << best_algo1 << ", " << best_algo2 << ", " << best_ratio << ", " << best_perf << "ms\n";
      } else if (run_mode == 2) {

      }
      std::cout << "\n";
    }
    return 0;
  }
}