#include <cuda_runtime.h>
#include <common/cuda_check.hpp>
#include <nms/nms_api.hpp>

#include <cstdint>
#include <cmath>
#include <algorithm>

__device__ inline void rotate_around_center(const float2& center, 
                          const float& angle_cos,
                          const float& angle_sin,
                          float2& corner) {
    float x = corner.x - center.x;
    float y = corner.y - center.y;
    corner.x = x * angle_cos - y * angle_sin + center.x;
    corner.y = x * angle_sin + y * angle_cos + center.y;
}

__device__ inline float cross(const float2& p1, const float2& q, const float2& p0) {
    return (p1.x - p0.x) * (q.y - p0.y) - (q.x - p0.x) * (p1.y - p0.y);
}

__device__ inline bool intersection(const float2& p1, 
                                    const float2& p2, 
                                    const float2& q1, 
                                    const float2& q2, 
                                    float2& cp) {
    // 1. 快速排斥 
    // 两box相交，x轴和y轴投影相交 -> x轴或y轴上投影不相交 -> 不相交
    if (fminf(p1.x, p2.x) > fmaxf(q1.x, q2.x) || 
        fmaxf(p1.x, p2.x) < fminf(q1.x, q2.x) ||
        fminf(p1.y, p2.y) > fmaxf(q1.y, q2.y) || 
        fmaxf(p1.y, p2.y) < fminf(q1.y, q2.y))
        return false;

    // 2. 跨立实验
    float s1 = cross(p2, q1, p1);
    float s2 = cross(p2, q2, p1);
    float s3 = cross(q2, p1, q1);
    float s4 = cross(q2, p2, q1);

    if (s1*s2 > 0 || s3*s4 > 0) return false; // 等于 0 的情况（端点落在线上）会在后续的 checkBox 处理

    if (fabs(s2 - s1) < 1e-8) return false;     // 接近共线的情况
    
    // 3. 等比定交点
    cp.x = (s1 * q2.x - s2 * q1.x) / (s1 - s2);
    cp.y = (s1 * q2.y - s2 * q1.y) / (s1 - s2);
    return true;
}

__device__ inline bool check_box2d(const float* box, const float2 p) {
    float angle_cos = cos(-box[6]), angle_sin = sin(-box[6]);
    float center_x = box[0], center_y = box[1];
    float x = (p.x - center_x), y = (p.y - center_y);
    float rot_x = x * angle_cos - y * angle_sin;
    float rot_y = x * angle_sin + y * angle_cos;
    return ((fabs(rot_x) < box[3] / 2 + MARGIN) && (fabs(rot_y) < box[4] / 2 + MARGIN));
}

__device__ bool devIoU(const float* box_a, const float* box_b, const float& nms_th) {
    float a_angle = box_a[6], b_angle = box_b[6];
    float a_dx_half = box_a[3] / 2, b_dx_half = box_b[3] / 2; 
    float a_dy_half = box_a[4] / 2, b_dy_half = box_b[4] / 2;
    float2 box_a_corners[5];
    float2 box_b_corners[5];

    float a_x1 = box_a[0] - a_dx_half, a_y1 = box_a[1] - a_dy_half;
    float a_x2 = box_a[0] + a_dx_half, a_y2 = box_a[1] + a_dy_half;
    float b_x1 = box_b[0] - b_dx_half, b_y1 = box_b[1] - b_dy_half;
    float b_x2 = box_b[0] + b_dx_half, b_y2 = box_b[1] + b_dy_half;

    float2 center_a = {box_a[0], box_a[1]};
    float2 center_b = {box_b[0], box_b[1]};

    float2 cross_points[16]; // 两个矩形最多8个交点 如果两个矩形几乎重叠 checkBox的时候还会各有4个角点被算作交点
    float2 poly_center = {0, 0};

    int cnt = 0;
    bool flag = false;

    box_a_corners[0] = float2{a_x1, a_y1}; // left top
    box_a_corners[1] = float2{a_x2, a_y1}; // right top
    box_a_corners[2] = float2{a_x2, a_y2}; // right bottom
    box_a_corners[3] = float2{a_x1, a_y2}; // left bottom

    box_b_corners[0] = float2{b_x1, b_y1}; // left top
    box_b_corners[1] = float2{b_x2, b_y1}; // right top
    box_b_corners[2] = float2{b_x2, b_y2}; // right bottom
    box_b_corners[3] = float2{b_x1, b_y2}; // left bottom

    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }
    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i], box_a_corners[i+1], box_b_corners[j], box_b_corners[j+1], cross_points[cnt]);
            if (flag) {
                poly_center.x += cross_points[cnt].x;
                poly_center.y += cross_points[cnt].y;
                cnt++;
            }
        }
    }

    #pragma unroll
    for (int k = 0; k < 4; k++) {
        if (check_box2d(box_a, box_b_corners[k])) {
            poly_center.x += box_b_corners[k].x;
            poly_center.y += box_b_corners[k].y;
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }

        if (check_box2d(box_b, box_a_corners[k])) {
            poly_center.x += box_a_corners[k].x;
            poly_center.y += box_a_corners[k].y;
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }
    
    if (cnt < 3) return false;

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    // 对所有的交点进行排序（按照交点中心顺时针排序）
    float2 temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
                atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
                ) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    // 鞋带公式求面积
    float area = 0.f;
    for (int k = 0; k < cnt - 1; k++) {
        float2 a = {cross_points[k].x - cross_points[0].x,
                    cross_points[k].y - cross_points[0].y};
        float2 b = {cross_points[k + 1].x - cross_points[0].x,
                    cross_points[k + 1].y - cross_points[0].y};
        area += (a.x * b.y - a.y * b.x);
    }

    float s_overlap = fabs(area) / 2.0;
    float sa = box_a[3] * box_a[4];
    float sb = box_b[3] * box_b[4];
    float iou = s_overlap / fmaxf(sa + sb - s_overlap, 1e-8);

    return iou > nms_th;
}

/**
  * @brief NMS kernel, computes IoU and generates mask
  * 
  * @param n_boxes Number of boxes
  * @param iou_threshold IoU threshold for NMS
  * @param dev_boxes Boxes on device [x, y, z, dx, dy, dz, ...]
  * @param dev_mask Output mask
  */
 __global__ void nms_cuda(const int n_boxes, 
    const float iou_threshold, 
    const float *dev_boxes, 
    uint64_t *dev_mask) {
    const int row0 = blockIdx.y;
    const int col0 = blockIdx.x;
    const int tid = threadIdx.x;

    /*
    row 0: 0 1 2 3
    row 1:   1 2 3
    row 2:     2 3
    row 3:       3
    */
    if (row0 > col0) return;

    const int row_size = min(n_boxes - row0 * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);
    const int col_size = min(n_boxes - col0 * NMS_THREADS_PER_BLOCK, NMS_THREADS_PER_BLOCK);

    __shared__ float block_boxes[NMS_THREADS_PER_BLOCK][7];

    // para load (考虑用一个 float4 + float3 load 优化？)
    if (tid < col_size) {
        block_boxes[tid][0] = dev_boxes[(NMS_THREADS_PER_BLOCK * col0 + tid) * DET_CHANNEL + 0];
        block_boxes[tid][1] = dev_boxes[(NMS_THREADS_PER_BLOCK * col0 + tid) * DET_CHANNEL + 1];
        block_boxes[tid][2] = dev_boxes[(NMS_THREADS_PER_BLOCK * col0 + tid) * DET_CHANNEL + 2];
        block_boxes[tid][3] = dev_boxes[(NMS_THREADS_PER_BLOCK * col0 + tid) * DET_CHANNEL + 3];
        block_boxes[tid][4] = dev_boxes[(NMS_THREADS_PER_BLOCK * col0 + tid) * DET_CHANNEL + 4];
        block_boxes[tid][5] = dev_boxes[(NMS_THREADS_PER_BLOCK * col0 + tid) * DET_CHANNEL + 5];
        block_boxes[tid][6] = dev_boxes[(NMS_THREADS_PER_BLOCK * col0 + tid) * DET_CHANNEL + 6];
    }
    __syncthreads();

    // for each row box
    if (tid < row_size) {
        const int cur_box_idx = NMS_THREADS_PER_BLOCK * row0 + tid;
        const float *cur_box = dev_boxes + cur_box_idx * DET_CHANNEL;

        uint64_t cur_mask = 0;
        int start = (row0 == col0) ? tid + 1 : 0; // 对角线格子上，也只进行上三角的计算

        for (int i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes[i], iou_threshold))
                cur_mask |= 1ULL << i;
        }
        // gridDim.x 就是 DIVUP(n_boxes, NMS_THREADS_PER_BLOCK)
        dev_mask[cur_box_idx * gridDim.x + col0] = cur_mask;
    }
}

/**
  * @brief rotated NMS kernel launcher
  * 
  * @param boxes_num Number of boxes
  * @param boxes_sorted Boxes sorted by score [x, y, z, dx, dy, dz, ...]
  * @param nms_iou_threshold IoU threshold for NMS
  * @param mask Output mask
  * @param stream CUDA stream
  * @return int 0 if success, -1 otherwise
  */
int rotated_nms_launch(unsigned int boxes_num,
    const float *boxes_sorted,
    float nms_iou_threashold,
    uint64_t* mask,
    cudaStream_t stream) {
    int col_blocks = (boxes_num + NMS_THREADS_PER_BLOCK - 1) / NMS_THREADS_PER_BLOCK;
    dim3 grids(col_blocks, col_blocks);
    dim3 blocks(NMS_THREADS_PER_BLOCK);

    nms_cuda<<<grids, blocks, 0, stream>>>(boxes_num, nms_iou_threashold, boxes_sorted, mask);

    CHECK_CUDA(cudaGetLastError());

    return 0;
}