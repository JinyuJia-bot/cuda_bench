import numpy as np
import time

def generate_boxes_rotated(
        n = 4096, R = 50.0,
        # size ranges 
        w_range=(1.6, 2.2), 
        l_range=(3.5, 5.0),
        # yaw
        yaw_mode='uniform',  # "uniform" | "bimodal" | "near0"
        # score
        score_shape=(2.0, 5.0),
        clusters=0, clusters_std=1.0,
        # corner cases injection
        p_edge=0.05,
        p_tiny=0.01,
        p_huge=0.01,
        seed=42):
    rng = np.random.default_rng(seed)

    # ---- center sampling ----
    if clusters > 0:
        centers = rng.uniform(-R, R, size=(clusters, 2))
        cid = rng.integers(0, clusters, size=n)
        cxy = centers[cid] + rng.normal(0, clusters_std, size=(n, 2))
        cx, cy = cxy[:, 0], cxy[:, 1]
    else:
        cx = rng.uniform(-R, R, size=n)
        cy = rng.uniform(-R, R, size=n)
    
    # ---- size sampling ----
    w = rng.uniform(w_range[0], w_range[1], size=n)
    l = rng.uniform(l_range[0], l_range[1], size=n)

    # --- yaw sampling ----
    if yaw_mode == 'uniform':
        yaw = rng.uniform(-np.pi, np.pi, size=n)
    elif yaw_mode == 'bimodal':
        # 两个主方向（比如道路方向），再加一点噪声
        yaw = rng.choice([0.0, np.pi/2], size=n) + rng.normal(0, 0.15, size=n)
    elif yaw_mode == 'near0':
        yaw = rng.normal(0, 0.2, size=n)
    else:
        raise ValueError("Invalid yaw_mode: {}".format(yaw_mode))
    
    # ---- 2.5D ----
    cz = rng.uniform(-2.0, 2.0, size=n)
    h  = rng.uniform(1.2, 2.2, size=n)

    # ---- score sampling ----
    a, b = score_shape
    score = rng.beta(a, b, size=n)

    # --- inject corner cases ----
    # edge: push centers close to ROI boundary
    m = rng.random(n) < p_edge
    # chose which side: x=±R or y=±R
    side = rng.integers(0, 4, size=n)
    eps = rng.uniform(0.0, 0.5, size=n)
    cx[m & (side==0)] = R - eps[m & (side==0)]
    cx[m & (side==1)] = -R + eps[m & (side==1)]
    cy[m & (side==2)] = R + eps[m & (side==2)]
    cy[m & (side==3)] = -R + eps[m & (side==3)]

    # tiny: set w,l to very small values
    m = rng.random(n) < p_tiny
    w[m] = rng.uniform(1e-4, 5e-3, size=np.sum(m))
    l[m] = rng.uniform(1e-4, 5e-3, size=np.sum(m))

    # huge: set w,l to very large values
    m = rng.random(n) < p_huge
    w[m] = rng.uniform(30.0, 120.0, size=np.sum(m))
    l[m] = rng.uniform(30.0, 120.0, size=np.sum(m))

    boxes = np.stack([cx, cy, cz, l, w, h, yaw, score], axis=1).astype(np.float32)
    return boxes

if __name__ == "__main__":

    boxes = generate_boxes_rotated(
        n=1000,
        R=50.0,
        w_range=(1.6, 2.2),
        l_range=(3.5, 5.0),
        yaw_mode='near0',
        score_shape=(2.0, 5.0),
        clusters=10,
        clusters_std=2.0,
        p_edge=0.05,
        p_tiny=0.01,
        p_huge=0.01,
        seed=42
    )

    # import torch
    # from detectron2.layers import nms_rotated

    # def nms_3d_cpu(boxes, iou_threshold=0.1):
    #     # boxes: [N, 8] (cx, cy, cz, l, w, h, yaw, score)
    #     # 使用 BEV 的 NMS
    #     boxes_bev = boxes[:, [0, 1, 3, 4, 6, 7]]  # (cx, cy, l, w, yaw, score)
    #     boxes_bev[:, 4] = boxes_bev[:, 4] * 180.0 / np.pi  # rad -> deg
        
    #     boxes_tensor = torch.from_numpy(boxes_bev[:, :5]).float()
    #     scores_tensor = torch.from_numpy(boxes_bev[:, 5]).float()

    #     keep_indices = nms_rotated(
    #         boxes_tensor,
    #         scores_tensor,
    #         iou_threshold
    #     ).cpu().numpy()
    #     return keep_indices

    # start_time = time.time()
    # keep_indices = nms_3d_cpu(boxes, iou_threshold=0.3)
    # end_time = time.time()

    # print("Generated {} boxes.".format(len(boxes)))
    # print("Kept {} boxes after NMS.".format(len(keep_indices)))

    np.save("rotated_boxes.npy", boxes)
    # np.save("rotated_keep_ref.npy", keep_indices)