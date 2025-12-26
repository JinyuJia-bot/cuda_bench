#pragma once
#include <cublas_v2.h>
#include <cstdint>

struct GemmDesc {
    int M, N, K;
    float alpha, beta;
};

// row-major A(MxK), B(KxN), C(MxN)
void gemm_naive_rowmajor(const float* dA, const float* dB, float* dC, const GemmDesc& d);

void gemm_shared_rowmajor(const float* dA, const float* dB, float* dC, const GemmDesc& d);

void gemm_reg4x4_rowmajor(const float* dA, const float* dB, float* dC, const GemmDesc& d);

void gemm_cublas_rowmajor(cublasHandle_t handle, const float* dA, const float* dB, float* dC, const GemmDesc& d);