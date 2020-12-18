#ifndef CUDNN_CONV_H_
#define CUDNN_CONV_H_

#include <cudnn.h>
#include <cuda_runtime.h>
#include <iostream>
#include "basic.h"

using std::cout;
using std::endl;
extern cudnnHandle_t cudnn_handle_g;

extern void* input_data_g;
extern void* output_data_g;
extern void* weight_data_g;
extern void* cudnn_workspace_g;

#define CHECK_EXIT(sts, str) \
    if (sts != CUDNN_STATUS_SUCCESS) \
    {       \
        std::cerr << str << ": " << cudnnGetErrorString(sts) << " " << __FILE__ << " "  << __LINE__ << std::endl; \
        exit(0);\
    }

class CudnnConv {
public:
    CudnnConv(int in_n, int in_c, int in_d,  int in_h, int in_w,
              int k_o_c, int k_i_c, int k_d, int k_h, int k_w,
              int p_d, int p_h, int p_w,
              int s_d, int s_h, int s_w,
              int d_d, int d_h, int d_w,
              int group,
              cudnnDataType_t in_type,
              cudnnDataType_t weight_type,
              cudnnDataType_t out_type,
              cudnnTensorFormat_t in_format,
              cudnnTensorFormat_t weight_format,
              cudnnTensorFormat_t output_format);
    ~CudnnConv() {
        cout << "~CudnnConv() ..." << endl;
    }

    void Run(void* input,
             void* weight,
             void* output,
             void* cudnn_workspace,
             cudnnHandle_t handle);

    int output_h() {
        int kernel_extent = dilation_h_ * (kernel_h_ - 1) + 1;
        return (input_h_ + 2 * pad_h_ - kernel_extent) / stride_h_ + 1;
    }
    int output_w() {
        int kernel_extent = dilation_w_ * (kernel_w_ - 1) + 1;
        return (input_w_ + 2 * pad_w_ - kernel_extent) / stride_w_ + 1;
    }
    int output_d() {
        return input_d_;
    }
    int output_size() {
        return output_n() * output_c() * output_d() * output_h() * output_w();
    }
    int output_n() {
        return input_n_;
    }
    int output_c() {
        return kernel_out_c_;
    }
    cudnnDataType_t conv_type();
    void InitAlgo(cudnnHandle_t handle);
    cudnnConvolutionFwdAlgo_t algo() {
        return algo_;
    }
    void cudnn_handle(cudnnHandle_t handle) {
        cudnn_handle_ = handle;
    }
    void input_data(void* data) {
        input_data_ = data;
    }
    void output_data(void* data) {
        output_data_ = data;
    }
    void weight_data(void* data) {
        weight_data_ = data;
    }
    void cudnn_workspace(void* data) {
        cudnn_workspace_ = data;
    }
    void* input_data() {
        return input_data_;
    }
    void* output_data() {
        return output_data_;
    }
    void* weight_data() {
        return weight_data_;
    }

    const size_t &workspace_size() {
        return cudnn_workspace_size_;
    }

private:
    int input_n_;
    int input_c_;
    int input_d_;
    int input_h_;
    int input_w_;

    int kernel_in_c_;
    int kernel_out_c_;
    int kernel_d_;
    int kernel_h_;
    int kernel_w_;
    int pad_d_;
    int pad_h_;
    int pad_w_;
    int stride_d_;
    int stride_h_;
    int stride_w_;
    int dilation_d_;
    int dilation_h_;
    int dilation_w_;
    int group_;

    cudnnDataType_t input_type_;
    cudnnDataType_t weight_type_;
    cudnnDataType_t output_type_;
    cudnnDataType_t conv_type_;

    cudnnTensorFormat_t input_format_;
    cudnnTensorFormat_t weight_format_;
    cudnnTensorFormat_t output_format_;

    cudnnConvolutionFwdAlgo_t algo_;
    void* input_data_;
    void* weight_data_;
    void* output_data_;

    cudnnTensorDescriptor_t input_desc_;
    cudnnTensorDescriptor_t output_desc_;
    cudnnFilterDescriptor_t weight_desc_;
    cudnnConvolutionDescriptor_t conv_desc_;

    void* cudnn_workspace_;
    size_t cudnn_workspace_size_;
    cudnnHandle_t cudnn_handle_;
};

#endif


