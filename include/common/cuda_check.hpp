#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do {           \
    cudaError_t _e = (call);            \
    if(_e != cudaSuccess) {             \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e));   \
        std::exit(1);                   \
    }                                   \
} while(0);                              

#define CHECK_CUBLAS(call) do {         \
    cublasStatus_t _s = (call);         \
    if(_s != CUBLAS_STATUS_SUCCESS) {   \
        fprintf(stderr, "cuBLAS error %s:%d: status=%d\n", __FILE__, __LINE__, (int)_s); \
        std::exit(1);                   \
    }\
} while(0);