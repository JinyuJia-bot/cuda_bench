#pragma once
#include "common/cuda_check.hpp"
#include <functional>

inline float time_cuda_events(const std::function<void()>& fn, int iters, int warmup=1) {
    for (int i = 0; i < warmup; i++) fn();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t st, ed;
    CHECK_CUDA(cudaEventCreate(&st));
    CHECK_CUDA(cudaEventCreate(&ed));
    CHECK_CUDA(cudaEventRecord(st));
    for (int i = 0; i < iters; i++) fn();
    CHECK_CUDA(cudaEventRecord(ed));
    CHECK_CUDA(cudaEventSynchronize(ed));

    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));
    CHECK_CUDA(cudaEventDestroy(st));
    CHECK_CUDA(cudaEventDestroy(ed));
    
    return ms;
}