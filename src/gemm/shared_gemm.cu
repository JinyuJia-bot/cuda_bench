#include "gemm/gemm_api.hpp"
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 16

__global__ void gemm_shared_rm_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    // 如果 thread-wise return了 后面的__syncthreads() 就会卡死
    // if (row >= M || col >= N) return; 

    int num_threads = BM * BN;
    int thread_idx = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tile = (K + BK - 1) / BK;
    float acc = 0.f;
    for (int t = 0; t < tile; t++) {
        for (int idx = thread_idx; idx < BM*BK; idx += num_threads) {
            int c = idx % BK;
            int r = idx / BK;
            int cg = t * BK + c;
            int rg = blockIdx.y * BM + r;
            As[r][c] = (rg < M && cg < K) ? A[K * rg + cg] : 0.f;
        }

        for (int idx = thread_idx; idx < BK*BN; idx += num_threads) {
            int c = idx % BN;
            int r = idx / BN;
            int cg = blockIdx.x * BN + c;
            int rg = t * BK + r;
            Bs[r][c] = (rg < K && cg < N) ? B[N * rg + cg] : 0.f;
        }
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            acc = fmaf(As[threadIdx.y][k], Bs[k][threadIdx.x], acc);
        }
        __syncthreads();
    }


    if (row < M && col < N) {
        float cold = C[row * N + col];
        C[row * N + col] = alpha * acc + beta * cold;
    }
}


void gemm_shared_rowmajor(const float* dA, const float* dB, float* dC, const GemmDesc& d)
{
    dim3 block(BN, BM);
    dim3 grid((d.N + BN - 1) / BN, (d.M + BM - 1) / BM);
    gemm_shared_rm_kernel<<<grid, block>>>(dA, dB, dC, d.M, d.N, d.K, d.alpha, d.beta);
}