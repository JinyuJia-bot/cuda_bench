#include "gemm/gemm_api.hpp"
#include "common/cuda_check.hpp"
#include <cublas_v2.h>

void gemm_cublas_rowmajor(cublasHandle_t h,
                          const float* dA, const float* dB, float* dC,
                          const GemmDesc& d)
{
    const float alpha = d.alpha;
    const float beta = d.beta;

    // Row-major trick:
    // C(MxN) Row-major == C^T (NxM)
    // C^T = B^T x A^T
    CHECK_CUBLAS(cublasSgemm(
        h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        d.N, d.M, d.K,              // (N x M) = (N x K) * (k * M)
        &alpha,
        dB, d.N,
        dA, d.K,
        &beta,
        dC, d.N
    ));
}