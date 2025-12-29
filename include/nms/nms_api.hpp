#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

#define NMS_THREADS_PER_BLOCK 64 
#define DET_CHANNEL 8
#define MARGIN 1e-2

struct Boxes {
    std::vector<float> data;
    int n;
    int c;
};

int rotated_nms_launch(unsigned int boxes_num,
                       const float *boxes_sorted,
                       float nms_iou_threashold,
                       uint64_t* mask,
                       cudaStream_t stream);