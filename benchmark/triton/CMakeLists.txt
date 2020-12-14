cmake_minimum_required(VERSION 2.8)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# # LLVM
# find_package(LLVM REQUIRED)
# link_directories(${LLVM_LIBRARY_DIRS})
# include_directories(${LLVM_INCLUDE_DIRS})
# add_definitions(${LLVM_DEFINITIONS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS -fvisibility=default -std=gnu++14")

message(STATUS "Adding C++ tests")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common")
add_subdirectory(bench)
