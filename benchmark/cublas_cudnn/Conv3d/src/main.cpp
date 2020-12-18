#include <algorithm> 
#include <cctype>
#include <locale>
#include <iostream>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include "cudnn_conv.h"
#include "profile.h"
using namespace std;

void* input_data_g = nullptr;
void* output_data_g = nullptr;
void* weight_data_g = nullptr;
void* cudnn_workspace_g = nullptr;
void* input_data_host_g = nullptr;
void* output_data_host_g = nullptr;
void* weight_data_host_g = nullptr;

template<typename input_t>
void InitData(int data_size, int weight_size, int output_size, int workspace_size) {
    cout << "Allocating Data ..........." << endl;

    if (!input_data_g) cudaMalloc(&input_data_g, data_size);
    if (!weight_data_g) cudaMalloc(&weight_data_g, weight_size);
    if (!output_data_g) cudaMalloc(&output_data_g, output_size);
    if (!cudnn_workspace_g) cudaMalloc(&cudnn_workspace_g, workspace_size);
    if (!input_data_host_g) input_data_host_g = malloc(data_size);
    if (!weight_data_host_g) weight_data_host_g = malloc(weight_size);
    if (!output_data_host_g) output_data_host_g = malloc(output_size);

    cout << "Initing Data ..." << endl;
    for (int i = 0; i < data_size / sizeof(input_t); i++) {
        ((input_t*)input_data_host_g)[i] = 0.5;
    }
    for (int i = 0; i < weight_size / sizeof(input_t); i++) {
        ((input_t*)weight_data_host_g)[i] = 1;
    }

    cudaMemcpy(input_data_g, input_data_host_g, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(weight_data_g, weight_data_host_g, weight_size, cudaMemcpyHostToDevice);
}

void ReleaseData() {
    cout << "Release Data ..........." << endl;
    if (input_data_g) cudaFree(input_data_g);
    if (weight_data_g) cudaFree(weight_data_g);
    if (output_data_g) cudaFree(output_data_g);
    if (cudnn_workspace_g) cudaFree(cudnn_workspace_g);
    if (input_data_host_g) free(input_data_host_g);
    if (weight_data_host_g) free(weight_data_host_g);
    if (output_data_host_g) free(output_data_host_g);
}

template<typename input_t, typename output_t>
void TestConv3D(int input_n, int input_c, int input_d, int input_h, int input_w, 
                   int k_n, int k_c, int k_d, int k_h, int k_w, 
                   int p_d, int p_h, int p_w, 
                   int s_d, int s_h, int s_w,
                   int d_d, int d_h, int d_w,
                   int group,
                   cudnnDataType_t in_type,
                   cudnnDataType_t weight_type,
                   cudnnDataType_t out_type,
                   cudnnTensorFormat_t in_format,
                   cudnnTensorFormat_t weight_format,
                   cudnnTensorFormat_t output_format,
                   bool validate = true) {

    CudnnConv conv(input_n, input_c, input_d, input_h, input_w,
                   k_n, k_c, k_d, k_h, k_w,
                   p_d, p_h, p_w, 
                   s_d, s_h, s_w, 
                   d_d, d_h, d_w,
                   group,
                   in_type, weight_type, out_type,
                   in_format, weight_format, output_format);

    CHECK_EXIT(cudnnCreate(&cudnn_handle_g), "cudnnCreate");

    conv.InitAlgo(cudnn_handle_g);

    int input_size = input_n * input_c * input_h * input_w * sizeof(input_t);
    int weight_size = k_n * k_c * k_d * k_h * k_w * sizeof(input_t);
    int output_size = conv.output_size() * sizeof(output_t);

    InitData<input_t>(input_size, weight_size, output_size, conv.workspace_size());

    OPT_PROFILE_TIME_RESET(0);
#ifdef NVPROFILE
    int profile_count = 1;
#else
    int profile_count = 1;
#endif
    OPT_PROFILE_TIME_START(0);
    for (int i = 0; i < profile_count; i++) {
        conv.Run(input_data_g, weight_data_g, output_data_g, cudnn_workspace_g, cudnn_handle_g);
    }
    OPT_PROFILE_TIME_STOP(0, "Run", profile_count, 1);

    ReleaseData();
    cudnnDestroy(cudnn_handle_g);
}

std::vector<std::tuple<int, int, int, int, int,
                       int, int, int, int, int,
                       int, int, int,
                       int, int, int,
                       int, int, int,
                       int>>
    config = {
        /* 
        int input_n, int input_c, int input_d, int input_h, int input_w, 
        int k_n, int k_c, int k_d, int k_h, int k_w, 
        int p_d, int p_h, int p_w, 
        int s_d, int s_h, int s_w,
        int d_d, int d_h, int d_w,
        int group,*/
        std::make_tuple(
            1, 3, 8, 128, 128,
            64, 3, 1, 3, 3,
            0, 1, 1,
            1, 1, 1,
            1, 1, 1,
            1),
        std::make_tuple(
            1, 3, 8, 350, 640,
            64, 3, 1, 7, 7,
            0, 3, 3,
            1, 2, 2,
            1, 1, 1,
            1),
};

int main() {
    for (const auto &problem : config) {
        int input_n, input_c, input_d, input_h, input_w, 
            k_n, k_c, k_d, k_h, k_w, 
            p_d, p_h, p_w, 
            s_d, s_h, s_w,
            d_d, d_h, d_w,
            group;
        
        std::tie(input_n, input_c, input_d, input_h, input_w, 
            k_n, k_c, k_d, k_h, k_w, 
            p_d, p_h, p_w, 
            s_d, s_h, s_w,
            d_d, d_h, d_w,
            group) = problem;
        
        // fp32
        {
            cudnnDataType_t type = CUDNN_DATA_FLOAT;
            cudnnTensorFormat_t layout = CUDNN_TENSOR_NCHW;

            TestConv3D<float, float>(input_n, input_c, input_d, input_h, input_w, 
                        k_n, k_c, k_d, k_h, k_w, 
                        p_d, p_h, p_w, 
                        s_d, s_h, s_w,
                        d_d, d_h, d_w,
                        group,
                        type, type, type,
                        layout, layout, layout,
                        false);
        }

        // fp16
        {
            cudnnDataType_t type = CUDNN_DATA_HALF;
            cudnnTensorFormat_t layout = CUDNN_TENSOR_NCHW;

            TestConv3D<uint16_t, uint16_t>(input_n, input_c, input_d, input_h, input_w, 
                        k_n, k_c, k_d, k_h, k_w, 
                        p_d, p_h, p_w, 
                        s_d, s_h, s_w,
                        d_d, d_h, d_w,
                        group,
                        type, type, type,
                        layout, layout, layout,
                        false);
        }
    }
    return 0;
}
