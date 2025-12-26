#include "gemm/gemm_api.hpp"
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define TM 4
#define TN 4
#define BK 16


__global__ void gemm_reg4x4_rm_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int M, int N, int K,
                                float alpha, float beta)
{
    int num_threads = BM * BN;
    int thread_idx = blockDim.x * threadIdx.y + threadIdx.x;

    __shared__ float As[BM*TM][BK];
    __shared__ float Bs[BK][BN*TN];

    int tile = (K + BK - 1) / BK;
    float acc[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            acc[m][n] = 0.f;
        }
    }

    for (int t = 0; t < tile; t++) {
        for (int idx = thread_idx; idx < BM*TM*BK; idx += num_threads) {
            int c = idx % BK;
            int r = idx / BK;
            int cg = t * BK + c;
            int rg = blockIdx.y * BM * TM + r;
            As[r][c] = (rg < M && cg < K) ? A[K * rg + cg] : 0.f;
        }

        for (int idx = thread_idx; idx < BK*BN*TN; idx += num_threads) {
            int c = idx % (BN * TN);
            int r = idx / (BN * TN);
            int cg = blockIdx.x * BN * TN + c;
            int rg = t * BK + r;
            Bs[r][c] = (rg < K && cg < N) ? B[N * rg + cg] : 0.f;
        }
        __syncthreads();
        
        // 从 shared -> register 也可以优化
        float aReg[TM];
        float bReg[TN];
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int m =0; m < TM; m++) {
                aReg[m] = As[threadIdx.y * TM + m][k];
            }

            #pragma unroll
            for(int n = 0; n < TN; n++) {
                bReg[n] = Bs[k][threadIdx.x * TN + n];
            }
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    acc[m][n] = fmaf(aReg[m], bReg[n], acc[m][n]); 
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; m++) {
        int rg = blockIdx.y * BM * TM + threadIdx.y * TM + m;
        if (rg >= M) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int cg = blockIdx.x * BN * TN + threadIdx.x * TN + n;
            if (cg >= N) continue;
            C[rg * N + cg] = alpha * acc[m][n] + beta * C[rg * N + cg]; 
        }
    }
}


void gemm_reg4x4_rowmajor(const float* dA, const float* dB, float* dC, const GemmDesc& d)
{
    dim3 block(BN, BM);
    dim3 grid((d.N + BN*TN - 1) / (BN*TN), (d.M + BM*TM - 1) / (BM*TM));
    gemm_reg4x4_rm_kernel<<<grid, block>>>(dA, dB, dC, d.M, d.N, d.K, d.alpha, d.beta);
}