#include "common/cuda_check.hpp"
#include "common/timer.hpp"
#include "common/cnpy.h"
#include "nms/nms_api.hpp"

#include <algorithm>
#include <stdio.h>

Boxes load_boxes_npy(const char* path) {
    cnpy::NpyArray arr = cnpy::npy_load(path);
    if (arr.word_size != sizeof(float))
        std::exit(1);
    
    int n = arr.shape[0];
    int c = arr.shape[1];

    if (c != DET_CHANNEL) {
        printf("ERROR: expect DET_CHANNEL=%d, got %d\n", DET_CHANNEL, c);
        std::exit(1);
    }

    Boxes b;
    b.n = n;
    b.c = c;
    b.data.resize(n*c);
    std::memcpy(b.data.data(), arr.data<float>(), n * c * sizeof(float));
    return b;
}

float benchmark_nms_from_npy(const Boxes& boxes, 
                             float iou_th,
                             int iters,
                             int warmup=2) {
    int n = boxes.n;
    // sort
    std::vector<float> boxes_sorted = boxes.data;

    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),[&](int i1, int i2){
        float score1 = boxes_sorted[i1 * DET_CHANNEL + 6];
        float score2 = boxes_sorted[i2 * DET_CHANNEL + 6];
        return score1 > score2;
    });
    std::vector<float> tmp(n * DET_CHANNEL);
    for (int i = 0; i < n; i++) {
        int src_idx = indices[i] * DET_CHANNEL;     // 原 box 起始位置
        int dst_idx = i * DET_CHANNEL;              // 新 box 起始位置
        // 复制单个 box 的 8 个元素
        std::copy(boxes_sorted.begin() + src_idx, 
                  boxes_sorted.begin() + src_idx + DET_CHANNEL,
                  tmp.begin() + dst_idx);
    }
    boxes_sorted.swap(tmp);
    float* d_boxes = nullptr;
    uint64_t* d_mask = nullptr;

    int col_blocks = (n + NMS_THREADS_PER_BLOCK - 1) / NMS_THREADS_PER_BLOCK; // 一个 box 用 col_blocks 个 uint64_t 记录
    size_t mask_elems = (size_t)n * col_blocks;

    CHECK_CUDA(cudaMalloc(&d_boxes, boxes.n * DET_CHANNEL * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_mask, mask_elems * sizeof(uint64_t)));
    CHECK_CUDA(cudaMemcpy(d_boxes, 
                          boxes_sorted.data(), 
                          boxes.n * DET_CHANNEL * sizeof(float),
                          cudaMemcpyHostToDevice));
                          
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // warmup
    for (int i = 0; i < warmup; i++) {
        rotated_nms_launch(n, d_boxes, iou_th, d_mask, stream);
        // CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t e0, e1;
    CHECK_CUDA(cudaEventCreate(&e0));
    CHECK_CUDA(cudaEventCreate(&e1));

    CHECK_CUDA(cudaEventRecord(e0, stream));
    for (int i = 0; i < iters; i++) {
        rotated_nms_launch(n, d_boxes, iou_th, d_mask, stream);
        // CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaEventRecord(e1, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, e0, e1));

    CHECK_CUDA(cudaEventDestroy(e0));
    CHECK_CUDA(cudaEventDestroy(e1));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_boxes));
    CHECK_CUDA(cudaFree(d_mask));

    return ms / iters;
}


int main(int argc, char** argv) {
    if (argc < 2) {
        printf("usage: %s boxes.npy [iters] [iou]\n", argv[0]);
    }
    CHECK_CUDA(cudaSetDevice(0));

    const char* npy_path = argv[1];
    int iters = (argc > 2) ? std::atoi(argv[2]) : 10;
    float iou = (argc > 3) ? std::atof(argv[3]) : 0.5f;

    Boxes boxes = load_boxes_npy(npy_path);
    printf("Loaded boxes: N=%d, C=%d\n", boxes.n, boxes.c);

    float avg_ms = benchmark_nms_from_npy(boxes, iou, iters, 10);

    double pairs = 0.5 * (double)boxes.n * boxes.n;
    double pairs_per_s = pairs / (avg_ms * 1e-3);

    printf("avg %.4f ms, ~%.3e box-pairs/s\n", avg_ms, pairs_per_s);
    return 0;
}

