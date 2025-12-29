cmake_minimum_required(VERSION 3.24)
project(gemm_bench LANGUAGES CXX CUDA)

# 基础全局配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON) 

# 自动检测 CUDA 架构
include(cmake/cuda_arch.cmake)

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# cnpy 需要 ZLIB
find_library(CNPY_LIB NAMES cnpy HINTS /usr/local/lib)
find_package(ZLIB REQUIRED)

# 1. 定义所有需要编译的库名称（除了 cublas 单独处理）
set(GEMM_LIBS 
    gemm_naive 
    gemm_shared 
    gemm_reg4x4 
    gemm_vecload
)

# 2. 批量配置通用库
foreach(lib ${GEMM_LIBS})
    # 提取库名后缀（如 naive、shared），拼接源文件路径
    string(REPLACE "gemm_" "" lib_suffix ${lib})
    add_library(${lib} STATIC src/gemm/${lib_suffix}_gemm.cu)
    
    # 通用配置（所有库共用）
    target_include_directories(${lib} PUBLIC include) 
    target_compile_options(${lib} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

# 3. 处理 cublas 库
add_library(gemm_cublas STATIC src/gemm/cublas_gemm.cu)
target_include_directories(gemm_cublas PUBLIC include)
target_link_libraries(gemm_cublas PUBLIC CUDA::cublas)
set_target_properties(gemm_cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# --------------------------
# 可执行文件配置
# --------------------------
add_executable(bench_gemm src/gemm/gemm.cpp)
target_include_directories(bench_gemm PRIVATE include)
# 链接所有 GEMM 库（通用库 + cublas 库）
target_link_libraries(bench_gemm PRIVATE 
    ${GEMM_LIBS} 
    gemm_cublas 
    CUDA::cudart
)

# 通用优化选项（批量应用到可执行文件）
target_compile_options(bench_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

set(NMS_LIBS
    rotated_nms
)

foreach(lib ${NMS_LIBS})
    string(REPLACE "_nms" "" lib_suffix ${lib})
    add_library(${lib} STATIC src/nms/${lib_suffix}_nms.cu)

    # 通用配置（所有库共用）
    target_include_directories(${lib} PUBLIC include) 
    target_compile_options(${lib} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    set_target_properties(${lib} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

add_executable(bench_nms src/nms/nms.cpp)
target_include_directories(bench_nms PRIVATE 
    include
    ${ZLIB_INCLUDE_DIRS}
)
target_link_libraries(bench_nms PRIVATE 
    ${NMS_LIBS} 
    CUDA::cudart
    ${ZLIB_LIBRARIES}   # zlib（cnpy 必需）
    ${CNPY_LIB}
)

# 通用优化选项（批量应用到可执行文件）
target_compile_options(bench_nms PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)


