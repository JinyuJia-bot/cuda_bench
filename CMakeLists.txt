cmake_minimum_required(VERSION 3.24)
project(gemm_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA Configuartion
set(CMAKE_CUDA_STANDARD 17)
set(CAMKE_CUDA_STANDARD_REQUIRED ON)

# 也可以手动指定，例如： -DCMAKE_CUDA_ARCHITECTURES=89
include(cmake/cuda_arch.cmake)

# 找 CUDA Toolkit (包含 cuBLAS 目标)
find_package(CUDAToolkit REQUIRED)

# ---- libraries ----
# lib naive
add_library(gemm_naive STATIC
    src/naive_gemm.cu)
target_include_directories(gemm_naive PUBLIC include)
target_compile_options(gemm_naive PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
set_target_properties(gemm_naive PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# lib shared
add_library(gemm_shared STATIC
    src/shared_gemm.cu)
target_include_directories(gemm_shared PUBLIC include)
target_compile_options(gemm_shared PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
set_target_properties(gemm_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# lib reg4x4
add_library(gemm_reg4x4 STATIC
    src/reg4x4_gemm.cu)
target_include_directories(gemm_reg4x4 PUBLIC include)
target_compile_options(gemm_reg4x4 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
set_target_properties(gemm_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# lib cublas
add_library(gemm_cublas STATIC
    src/cublas_gemm.cu)
target_include_directories(gemm_cublas PUBLIC include)
target_link_libraries(gemm_cublas PUBLIC CUDA::cublas)
set_target_properties(gemm_cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---- executbale ----
add_executable(bench_gemm 
    src/main.cpp
)
target_include_directories(bench_gemm PRIVATE include)
target_link_libraries(bench_gemm PRIVATE gemm_naive gemm_cublas gemm_shared gemm_reg4x4 CUDA::cudart)

# 常见优化
target_compile_options(bench_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)