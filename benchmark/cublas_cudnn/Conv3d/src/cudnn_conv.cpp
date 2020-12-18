#include <iostream>
#include <memory>
#include <vector>
#include "basic.h"
#include "cudnn_conv.h"

using namespace std;
using perf_t = cudnnConvolutionFwdAlgoPerf_t;

cudnnMathType_t math_type = CUDNN_TENSOR_OP_MATH;
cudnnHandle_t cudnn_handle_g;

CudnnConv::CudnnConv(int in_n, int in_c, int in_d,  int in_h, int in_w,
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
              cudnnTensorFormat_t output_format)
        : input_n_(in_n), input_c_(in_c), input_d_(in_d), input_h_(in_h), input_w_(in_w),
          kernel_in_c_(k_i_c), kernel_out_c_(k_o_c), kernel_d_(k_d), kernel_h_(k_h), kernel_w_(k_w),
          pad_d_(p_d), pad_h_(p_h), pad_w_(p_w),
          stride_d_(s_d), stride_h_(s_h), stride_w_(s_w),
          dilation_d_(d_d), dilation_h_(d_h), dilation_w_(d_w),
          group_(group),
          input_type_(in_type), weight_type_(weight_type), output_type_(out_type),
          input_format_(in_format), weight_format_(weight_format), output_format_(output_format),
          input_data_(nullptr),
          weight_data_(nullptr),
          output_data_(nullptr),
          cudnn_workspace_(nullptr){
        if (group_ != 1) {
            std::cout << "only support group != 1 now" << std::endl;
            exit(0);
        }
        cudnnStatus_t sts;
        CHECK_EXIT(cudnnCreateTensorDescriptor(&input_desc_), "Fail in create cudnnCreateTensorDescriptor");
        CHECK_EXIT(cudnnCreateTensorDescriptor(&output_desc_), "Fail in create cudnnCreateTensorDescriptor");
        CHECK_EXIT(cudnnCreateFilterDescriptor(&weight_desc_), "Fail in create cudnnCreateTensorDescriptor");
        CHECK_EXIT(cudnnCreateConvolutionDescriptor(&conv_desc_), "Fail in create cudnnCreateConvolutionDescriptor");

        // 设置输入的Descriptor 
        std::vector<int> input_shape_v = {input_n_, input_c_, input_d_, input_h_, input_w_}; 
        std::vector<int> input_stride_v = {input_c_ * input_d_ * input_h_ * input_w_, input_d_ * input_h_ * input_w_, input_h_ * input_w_, input_w_, 1};
        sts = cudnnSetTensorNdDescriptor(input_desc_,
                                         input_type_,
                                         input_shape_v.size(),
                                         input_shape_v.data(),
                                         input_stride_v.data()
                                         );        

        CHECK_EXIT(sts, "Error in setting Input's cudnnSetTensorNdDescriptorEx");
        printf("Input shape is (n %d, c %d, d %d, h %d, w %d)\n", input_n_, input_c_, input_d_, input_h_, input_w_);

        // 设置输出的Descriptor
        std::vector<int> output_shape_v = {output_n(), output_c(), output_d(), output_h(), output_w()};
        std::vector<int> output_stride_v = {output_c() * output_d() * output_h() * output_w(), output_d() * output_h() * output_w(), output_h() * output_w(), output_w(), 1};
        sts = cudnnSetTensorNdDescriptor(output_desc_,
                                         output_type_,
                                         output_shape_v.size(),
                                         output_shape_v.data(),
                                         output_stride_v.data()
                                         );
        CHECK_EXIT(sts, "Error in setting Output's cudnnSetTensorNdDescriptorEx");
        printf("Output shape is (n %d, c %d, d %d, h %d, w %d)\n", output_n(), output_c(), output_d(), output_h(), output_w());

        // 设置卷积函数的Descriptor
        std::vector<int> padding_shape_v = {p_d, p_h, p_w};
        std::vector<int> stride_shape_v = {s_d, s_h, s_w};
        std::vector<int> dilation_shape_v = {d_d, d_h, d_w};
        cudnnDataType_t conv_t = conv_type();
        sts = cudnnSetConvolutionNdDescriptor(conv_desc_, 
                                              padding_shape_v.size(),
                                              padding_shape_v.data(),
                                              stride_shape_v.data(),
                                              dilation_shape_v.data(),
                                              CUDNN_CROSS_CORRELATION,
                                              conv_t);
        CHECK_EXIT(sts, "Error in setting convolution's cudnnSetConvolutionNdDescriptor");

        // 设置权重函数的Descriptor
        std::vector<int> weight_shape_v = {k_o_c, k_i_c, k_d, k_h, k_w}; 
        sts = cudnnSetFilterNdDescriptor(weight_desc_, 
                                         weight_type_,
                                         weight_format_,
                                         weight_shape_v.size(),
                                         weight_shape_v.data());
        CHECK_EXIT(sts, "Error in setting filter's cudnnSetFilterNdDescriptor");
        printf("Weight shape is (output channel %d, input channel %d, d %d, h %d, w %d)\n", k_o_c, k_i_c, k_d, k_h, k_w);
}

cudnnDataType_t CudnnConv::conv_type() {
    if ((input_type_ == CUDNN_DATA_FLOAT) &&
        (output_type_ == CUDNN_DATA_FLOAT) &&
        (weight_type_ == CUDNN_DATA_FLOAT)) {
        return CUDNN_DATA_FLOAT;
    } else if ((input_type_ == CUDNN_DATA_HALF) &&
            (output_type_ == CUDNN_DATA_HALF) && 
             (weight_type_ == CUDNN_DATA_HALF)) {
        return CUDNN_DATA_HALF;
    } else if ((input_type_ == CUDNN_DATA_INT8) &&
            (output_type_ == CUDNN_DATA_INT8) && 
             (weight_type_ == CUDNN_DATA_INT8)) {
        return CUDNN_DATA_INT32;
    } else if ((input_type_ == CUDNN_DATA_INT8x4) &&
            (output_type_ == CUDNN_DATA_INT8x4) && 
             (weight_type_ == CUDNN_DATA_INT8x4)) {
        return CUDNN_DATA_INT32;
    } else if ((input_type_ == CUDNN_DATA_INT8x32) &&
            (output_type_ == CUDNN_DATA_INT8x32) && 
             (weight_type_ == CUDNN_DATA_INT8x32)) {
        return CUDNN_DATA_INT32;
    } else {
        std::cout << "conv_type not support" << std::endl;
        exit(0);
    }
}

const char* conv_type_str[] = {
 "CUDNN_DATA_FLOAT   ",
 "CUDNN_DATA_DOUBLE  ",
 "CUDNN_DATA_HALF    ",
 "CUDNN_DATA_INT8    ",
 "CUDNN_DATA_INT32   ",
 "CUDNN_DATA_INT8x4  ",
 "CUDNN_DATA_UINT8   ",
 "CUDNN_DATA_UINT8x4 ",
 "CUDNN_DATA_INT8x32 ",
};

void CudnnConv::InitAlgo(cudnnHandle_t handle) {
    cudnnStatus_t sts;
    sts = cudnnSetConvolutionMathType(conv_desc_, math_type);
    CHECK_EXIT(sts, "cudnnSetConvolutionMathType");
    cout << "conv_type: " << conv_type_str[conv_type()] << endl;

    cudnnConvolutionFwdAlgoPerf_t perf[10];
    if (conv_type() == CUDNN_DATA_INT32) {
        algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {
        int returnedAlgoCount = -1;
        sts = cudnnGetConvolutionForwardAlgorithm_v7(handle, 
                                                  input_desc_,
                                                  weight_desc_,
                                                  conv_desc_,
                                                  output_desc_,
                                                  10,
                                                  &returnedAlgoCount,
                                                  perf);
        std::cout << "returnedAlgoCount: " << returnedAlgoCount << std::endl;
        algo_ = perf[0].algo;
        CHECK_EXIT(sts, "Fail in getting cudnnGetConvolutionForwardAlgorithm");
    }

    cout << "algo_: " << algo_ << endl;
    sts = cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                  input_desc_,
                                                  weight_desc_,
                                                  conv_desc_,
                                                  output_desc_,
                                                  algo_,
                                                  &cudnn_workspace_size_);
    CHECK_EXIT(sts, "Fail setting workspace size cudnnGetConvolutionForwardWorkspaceSize");
    printf("The workspace size is %d\n", cudnn_workspace_size_);
}

void CudnnConv::Run(void* input,
                    void* weight,
                    void* output,
                    void* cudnn_workspace,
                    cudnnHandle_t handle) {
    float alpha = 1.0f;
    float beta = 1.0f;
    cudnnStatus_t sts;
    sts = cudnnConvolutionForward(handle,
                                  &alpha,
                                  input_desc_,
                                  input,
                                  weight_desc_,
                                  weight,
                                  conv_desc_,
                                  algo_,
                                  cudnn_workspace,
                                  cudnn_workspace_size_,
                                  &beta,
                                  output_desc_,
                                  output);
    CHECK_EXIT(sts, "cudnnConvolutionForward");
}
