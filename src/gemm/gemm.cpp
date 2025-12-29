#include "common/cuda_check.hpp"
#include "common/timer.hpp"
#include "gemm/gemm_api.hpp"

#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>

static void fill_random(std::vector<float>& h, unsigned seed = 123) {
    uint32_t x = seed;
    for(auto& v : h) {
        x = 1664525u * x + 1013904223u;
        v = float((int)(x & 0x00FFFFFF) / (double)0x00800000 - 1.0);
    }
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b){
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m ,std::fabs(a[i] - b[i]));
    return m;
}

// Run one case: M, N, K
static void run_case(int M, int N, int K, int iters) {
    printf("\n==== Case M=%d N=%d K=%d (iters=%d) ====\n", M, N, K, iters);

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M*K), hB((size_t)K*N), hC0((size_t)M*N);
    fill_random(hA, 1);
    fill_random(hB, 2);
    fill_random(hC0, 3);

    float *dA = nullptr, *dB = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));

    float *dC_naive = nullptr; 
    CHECK_CUDA(cudaMalloc(&dC_naive, bytesC));
    CHECK_CUDA(cudaMemcpy(dC_naive, hC0.data(), bytesC, cudaMemcpyHostToDevice));

    float *dC_shared = nullptr;
    CHECK_CUDA(cudaMalloc(&dC_shared, bytesC));
    CHECK_CUDA(cudaMemcpy(dC_shared, hC0.data(), bytesC, cudaMemcpyHostToDevice));

    float *dC_reg4x4 = nullptr;
    CHECK_CUDA(cudaMalloc(&dC_reg4x4, bytesC));
    CHECK_CUDA(cudaMemcpy(dC_reg4x4, hC0.data(), bytesC, cudaMemcpyHostToDevice));

    float *dC_vecload = nullptr;
    CHECK_CUDA(cudaMalloc(&dC_vecload, bytesC));
    CHECK_CUDA(cudaMemcpy(dC_vecload, hC0.data(), bytesC, cudaMemcpyHostToDevice));

    float *dC_cublas = nullptr;
    CHECK_CUDA(cudaMalloc(&dC_cublas, bytesC));
    CHECK_CUDA(cudaMemcpy(dC_cublas, hC0.data(), bytesC, cudaMemcpyHostToDevice));

    GemmDesc desc{M, N, K, 1.f, 0.f};
    
    auto naive_call = [&]() {
        gemm_naive_rowmajor(dA, dB, dC_naive, desc);
    };
    float naive_ms = time_cuda_events(naive_call, iters);

    // auto shared_call = [&]() {
    //     gemm_shared_rowmajor(dA, dB, dC_shared, desc);
    // };
    // float shared_ms = time_cuda_events(shared_call, iters);

    // auto reg4x4_call = [&]() {
    //     gemm_reg4x4_rowmajor(dA, dB, dC_reg4x4, desc);
    // };
    // float reg4x4_ms = time_cuda_events(reg4x4_call, iters);

    // auto vecload_call = [&]() {
    //     gemm_vecload_rowmajor(dA, dB, dC_vecload, desc);
    // };
    // float vecload_ms = time_cuda_events(vecload_call, iters);
    
    // cublasHandle_t handle;
    // CHECK_CUBLAS(cublasCreate(&handle));
    // // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)); // 可选： TF32 开关 （先建议关掉验证 diff）
    // auto cublas_call = [&]() {
    //     gemm_cublas_rowmajor(handle, dA, dB, dC_cublas, desc);
    // };
    // float cublas_ms = time_cuda_events(cublas_call, iters);
    // CHECK_CUBLAS(cublasDestroy(handle));

    // --- Correctness check (copy back once) ---
    std::vector<float> hC_naive((size_t)M*N);
    CHECK_CUDA(cudaMemcpy(hC_naive.data(), dC_naive, bytesC, cudaMemcpyDeviceToHost)); 
    
    std::vector<float> hC_shared((size_t)M*N);
    CHECK_CUDA(cudaMemcpy(hC_shared.data(), dC_shared, bytesC, cudaMemcpyDeviceToHost));

    std::vector<float> hC_reg4x4((size_t)M*N);
    CHECK_CUDA(cudaMemcpy(hC_reg4x4.data(), dC_reg4x4, bytesC, cudaMemcpyDeviceToHost));

    std::vector<float> hC_vecload((size_t)M*N);
    CHECK_CUDA(cudaMemcpy(hC_vecload.data(), dC_vecload, bytesC, cudaMemcpyDeviceToHost));

    std::vector<float> hC_cublas((size_t)M*N);
    CHECK_CUDA(cudaMemcpy(hC_cublas.data(), dC_cublas, bytesC, cudaMemcpyDeviceToHost));
    
    // float diff_naive = max_abs_diff(hC_naive, hC_cublas);
    // float diff_shared = max_abs_diff(hC_shared, hC_cublas);
    // float diff_reg4x4 = max_abs_diff(hC_reg4x4, hC_cublas);
    // float diff_vecload = max_abs_diff(hC_vecload, hC_cublas);

    // --- Report ---
    // Flops for GEMM: 2*M*N*K
    // double flops = 2.0 * (double)M * (double)N * (double)K;
    // double naive_t = naive_ms * 1e-3;
    // double shared_t = shared_ms * 1e-3;
    // double reg4x4_t = reg4x4_ms * 1e-3;
    // double vecload_t = vecload_ms * 1e-3;
    // double cublas_t = cublas_ms * 1e-3;
    
    // double naive_gflops = flops / naive_t / 1e9;
    // double shared_gflops = flops / shared_t / 1e9;
    // double reg4x4_gflops = flops / reg4x4_t / 1e9;
    // double vecload_gflops = flops / vecload_t / 1e9;
    // double cublas_gflops = flops / cublas_t / 1e9;

    // printf("naive   : %.3f ms, %.2f GFLOP/s\n", naive_ms, naive_gflops);
    // printf("shared  : %.3f ms, %.2f GFLOP/s\n", shared_ms, shared_gflops);
    // printf("reg4x4  : %.3f ms, %.2f GFLOP/s\n", reg4x4_ms, reg4x4_gflops);
    // printf("vecload : %.3f ms, %.2f GFLOP/s\n", vecload_ms, vecload_gflops);
    // printf("cuBLAS  : %.3f ms, %.2f GFLOP/s\n", cublas_ms, cublas_gflops);
    
    // printf("max|diff|: naive  vs cuBLAS: %.6g\n", diff_naive);
    // printf("max|diff|: shared vs cuBLAS: %.6g\n", diff_shared);
    // printf("max|diff|: reg4x4 vs cuBLAS: %.6g\n", diff_reg4x4);
    // printf("max|diff|: vecload vs cuBLAS: %.6g\n", diff_vecload);

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC_naive));
    CHECK_CUDA(cudaFree(dC_cublas));
    CHECK_CUDA(cudaFree(dC_shared));
    CHECK_CUDA(cudaFree(dC_reg4x4));
    CHECK_CUDA(cudaFree(dC_vecload));
}

int main() {
    CHECK_CUDA(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s, SM=%d, CC=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    run_case(64, 64, 64, 1);
    run_case(512, 512, 512, 1);
    run_case(2048, 2048, 2048, 1);
    return 0;
}