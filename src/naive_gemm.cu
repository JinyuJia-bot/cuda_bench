#include "gemm/gemm_api.hpp"
#include <cuda_runtime.h>

__global__ void gemm_naive_rm_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc = fmaf(A[row * K + k], B[k * N + col], acc); //
    }
    float cold = C[row * N + col];
    C[row * N + col] = alpha * acc + beta * cold;
}

void gemm_naive_rowmajor(const float* dA, const float* dB, float* dC, const GemmDesc& d)
{
    dim3 block(16, 16);
    dim3 grid((d.N + block.x - 1) / block.x, (d.M + block.y - 1) / block.y);
    gemm_naive_rm_kernel<<<grid, block>>>(dA, dB, dC, d.M, d.N, d.K, d.alpha, d.beta);
}
