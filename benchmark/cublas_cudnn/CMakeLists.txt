# We need cmake >= 3.8, since 3.8 introduced CUDA as a first class language
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(cublas_gemm_benchmark LANGUAGES CXX CUDA)

# Use C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Enable all compile warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# cublas_gemm_benchmark
add_executable(cublas_gemm_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_gemm_bench.cu
)

target_link_libraries(cublas_gemm_benchmark -lcublas -lcurand)

# cudnn_conv_benchmark
add_executable(cudnn_conv_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_conv_bench.cu
    )

target_link_libraries(cudnn_conv_benchmark -lcublas -lcurand -lcudnn)


