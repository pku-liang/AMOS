# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cublas_gemm_benchmark LANGUAGES CXX CUDA)

# Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# add include
include_directories("include")

# cublas_gemm_benchmark
add_executable(cublas_gemm_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_gemm_bench.cu
)

target_link_libraries(cublas_gemm_benchmark -lcublas -lcurand)

# cublas_gemv_benchmark_noTC, invoking cublasSgemv without tensor core
add_executable(cublas_gemv_benchmark_noTC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_gemv_bench_noTC.cu
)

target_link_libraries(cublas_gemv_benchmark_noTC -lcublas -lcurand)

# cublas_gemv_benchmark, converting to gemm for computation
add_executable(cublas_gemv_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_gemv_bench.cu
)

target_link_libraries(cublas_gemv_benchmark -lcublas -lcurand)

# cublas_batch_gemm_benchmark
add_executable(cublas_batch_gemm_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_batch_gemm_bench.cu
)

target_link_libraries(cublas_batch_gemm_benchmark -lcublas -lcurand)

# cudnn_conv_benchmark
add_executable(cudnn_conv_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_conv_bench.cu
    )

target_link_libraries(cudnn_conv_benchmark -lcublas -lcurand -lcudnn)

# cudnn_conv_dilated_benchmark
add_executable(cudnn_conv_dilated_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_conv_dilated_bench.cu
    )

target_link_libraries(cudnn_conv_dilated_benchmark -lcublas -lcurand -lcudnn)

# grouped_conv
add_executable(cudnn_grouped_conv
    ${CMAKE_CURRENT_SOURCE_DIR}/src/grouped_conv.cu
    )

target_link_libraries(cudnn_grouped_conv -lcublas -lcurand -lcudnn)

# batched_conv
add_executable(cudnn_batched_conv
    ${CMAKE_CURRENT_SOURCE_DIR}/src/batched_conv.cu
    )

target_link_libraries(cudnn_batched_conv -lcublas -lcurand -lcudnn)

# cudnn_depth_conv_benchmark
add_executable(cudnn_depth_conv_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_depth_conv_bench.cu
    )

target_link_libraries(cudnn_depth_conv_benchmark -lcublas -lcurand -lcudnn)

# cudnn_conv3d_benchmark
add_executable(cudnn_conv3d_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_conv3d_bench.cu
    )

target_link_libraries(cudnn_conv3d_benchmark -lcublas -lcurand -lcudnn)

# cudnn_conv1d_benchmark
add_executable(cudnn_conv1d_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_conv1d_bench.cu
    )

target_link_libraries(cudnn_conv1d_benchmark -lcublas -lcurand -lcudnn)
