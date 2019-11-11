cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)
#set(CMAKE_CXX_FLAGS "-O0 -g")
file(GLOB SOURCES "src/test/*.cpp" "src/*.cpp" "include/*.hpp" "src/knn/*.cu" "include/knn/*.cuh")
add_library(knn_cuda SHARED ${SOURCES})
target_compile_features(knn_cuda PUBLIC cxx_std_11)
target_include_directories(knn_cuda PRIVATE include)
set_target_properties(knn_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(knn_test test_knn_cuda.cpp)
set_target_properties(knn_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(knn_test PRIVATE knn_cuda)
target_include_directories(knn_test PRIVATE include)

target_compile_options(knn_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -arch=sm_70
                       >)

#target_compile_options(knn_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
#                       -G
#                       --generate-line-info
#                       --use_fast_math
#                       --relocatable-device-code=true
#                       >)

