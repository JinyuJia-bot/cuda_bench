#include "gemm/gemm_api.hpp"
#include <cuda_runtime.h>

#define BM 16
#define BN 16
#define BK 64
#define TM 4
#define TN 4

__device__ __forceinline__ float4 load_float4_guarded(const float* ptr, int valid4)
{
    // valid4: 0..4 表示最多有多少个 float 是有效的
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid4 >= 4) {
        v = *reinterpret_cast<const float4*>(ptr);
    } else {
        if (valid4 >= 1) v.x = ptr[0];
        if (valid4 >= 2) v.y = ptr[1];
        if (valid4 >= 3) v.z = ptr[2];
        // v.w remains 0
    }
    return v;
}

__global__ void gemm_vecload_rm_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K,
                                         float alpha, float beta)
{
    // blockDim = (BN, BM)
    const int tx = threadIdx.x; // 0..15
    const int ty = threadIdx.y; // 0..15
    const int num_threads = BM * BN;
    const int tid = BN * ty + tx;

    // 这个 block 负责的 C tile 左上角
    const int block_row0 = blockIdx.y * (BM * TM); // 16*4=64 rows
    const int block_col0 = blockIdx.x * (BN * TN); // 16*4=64 cols

    // shared: A 按 K 方向 float4；B 按 N 方向 float4
    __shared__ float4 As4[BM * TM][BK / 4];          // 64 x 16
    __shared__ float4 Bs4[BK][(BN * TN) / 4];        // 64 x 16

    float acc[TM][TN] = {0.0f};

    const int tiles = (K + BK - 1) / BK;

    for (int t = 0; t < tiles; ++t) {
        // ---- load A tile: (BM*TM) x BK  -> (64 x 64)
        // As4 has (64 x 16) float4 = 1024 float4
        for (int idx = tid; idx < (BM * TM) * (BK / 4); idx += num_threads) {
            const int r  = idx / (BK / 4);     // 0..63
            const int kv = idx % (BK / 4);     // 0..15

            const int rg = block_row0 + r;
            const int cg = t * BK + kv * 4;

            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (rg < M && cg < K) {
                const int remain = K - cg;                 // 可能 < 4
                const int valid4 = remain >= 4 ? 4 : max(remain, 0);
                v = load_float4_guarded(&A[rg * K + cg], valid4);
            }
            As4[r][kv] = v;
        }

        // ---- load B tile: BK x (BN*TN) -> (64 x 64)
        // Bs4 has (64 x 16) float4 = 1024 float4
        for (int idx = tid; idx < BK * ((BN * TN) / 4); idx += num_threads) {
            const int k  = idx / ((BN * TN) / 4); // 0..63
            const int cv = idx % ((BN * TN) / 4); // 0..15

            const int rg = t * BK + k;
            const int cg = block_col0 + cv * 4;

            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (rg < K && cg < N) {
                const int remain = N - cg;
                const int valid4 = remain >= 4 ? 4 : max(remain, 0);
                v = load_float4_guarded(&B[rg * N + cg], valid4);
            }
            Bs4[k][cv] = v;
        }

        __syncthreads();

        // ---- compute
        #pragma unroll
        for (int kv = 0; kv < BK / 4; ++kv) {
            // 每个 kv 对应 k = 4*kv .. 4*kv+3
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const int r = ty * TM + m;          // 0..63
                float4 a4 = As4[r][kv];

                // k0
                float4 b0 = Bs4[4 * kv + 0][tx];
                acc[m][0] = fmaf(a4.x, b0.x, acc[m][0]);
                acc[m][1] = fmaf(a4.x, b0.y, acc[m][1]);
                acc[m][2] = fmaf(a4.x, b0.z, acc[m][2]);
                acc[m][3] = fmaf(a4.x, b0.w, acc[m][3]);

                // k1
                float4 b1 = Bs4[4 * kv + 1][tx];
                acc[m][0] = fmaf(a4.y, b1.x, acc[m][0]);
                acc[m][1] = fmaf(a4.y, b1.y, acc[m][1]);
                acc[m][2] = fmaf(a4.y, b1.z, acc[m][2]);
                acc[m][3] = fmaf(a4.y, b1.w, acc[m][3]);

                // k2
                float4 b2 = Bs4[4 * kv + 2][tx];
                acc[m][0] = fmaf(a4.z, b2.x, acc[m][0]);
                acc[m][1] = fmaf(a4.z, b2.y, acc[m][1]);
                acc[m][2] = fmaf(a4.z, b2.z, acc[m][2]);
                acc[m][3] = fmaf(a4.z, b2.w, acc[m][3]);

                // k3
                float4 b3 = Bs4[4 * kv + 3][tx];
                acc[m][0] = fmaf(a4.w, b3.x, acc[m][0]);
                acc[m][1] = fmaf(a4.w, b3.y, acc[m][1]);
                acc[m][2] = fmaf(a4.w, b3.z, acc[m][2]);
                acc[m][3] = fmaf(a4.w, b3.w, acc[m][3]);
            }
        }

        __syncthreads();
    }

    // ---- store C
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const int rg = block_row0 + ty * TM + m;
        if (rg >= M) continue;

        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            const int cg = block_col0 + tx * TN + n;
            if (cg >= N) continue;

            float cold = C[rg * N + cg];
            C[rg * N + cg] = alpha * acc[m][n] + beta * cold;
        }
    }
}

void gemm_vecload_rowmajor(const float* dA, const float* dB, float* dC, const GemmDesc& d)
{
    dim3 block(BN, BM); // (16,16)
    dim3 grid((d.N + BN * TN - 1) / (BN * TN),
              (d.M + BM * TM - 1) / (BM * TM));

    gemm_vecload_rm_kernel<<<grid, block>>>(dA, dB, dC, d.M, d.N, d.K, d.alpha, d.beta);
}