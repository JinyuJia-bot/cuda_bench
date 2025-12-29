###  1. 用 Nsight Systesms 看时间开销具体在哪个kernel上
    - nsys profile + Nsight Systems GUI: 确认耗时是否都在 nms_cuda 上，有没有隐藏同步、memcpy、GPU等待
    - nsys profile -t cuda,nvtx -o nms_sys --force_overwrite true ./bench_nms ../script/rotated_boxes.npy 100 0.5
    - launch 间隔: CPU 是否来不及喂 GPU （大量间隔）
    - 结论： A GPU kernel 绝对主导 -> 继续 kernel micro profiling
            B launch/CPU overhead 明显 （1000 boxes 的 n^2 可能 kernel 也大，但也可能 launch 太频繁） -> 考虑融合/减少调用次数

### 2. 定位到 kernel 内部： Nsight Compute 看是算耗时还是访存耗时
    - 工具： ncn --set full 或者 GUI
    - sudo /opt/nvidia/nsight-compute/2022.2.1/ncu --set full --kernel-name nms_cuda -o nms_ncu --force-overwrite ./bench_nms ../script/rotated_boxes.npy 100 0.5
    - ncu --set full --kernel-name nms_cuda -o nms_ncu --force-overwrite ./bench_nms ../script/rotated_boxes.npy 100 0.5
    - 目的： 判断瓶颈类型：1. 算术（SFU/数学函数）2. 内存（global/shared）3. 分支/发散  4. occupancy 被寄存器压死
    - 1. SM Busy / Achieved Occupancy
    - 2. Register per thread & Local Memory(spill)  
        -- (ptax info + ncu 的 local load/store spill 出现基本上等于 devIOU太重)
    - 3. Warp execution efficiency / Branch efficiency 
        -- rotated IOU 里 大量的 early return 
        -- 交点 cnt 不同导致的循环次数不同->典型发散
    - 4. Memory: global load efficiency / L2 hit / DRAM throughput 
        -- cur_box 从 global 读多次
        -- block_boxes shared load 的 bank conflict
    - 5. SFU / special function cos/sin/atan2 
        -- 走 SFU 或者 libdevice 路径
        -- ncu里能看到 SFU utilization / pipe utiliaztion
        -- 指令集里 special math 占比高

### 3. 验证猜想，设计对照实验
    - 1. 把 devIoU 替换成轻量版本
        - devIoU 直接 return false
    - 2. 禁用 atan2排序（给定一个假面积）
        - 如果时间大幅度下降 -> atan2 + 冒泡
    - 3. 禁用 check_box2d 的 trig
        - check_box2d 每次都 cos/sin，改为预计算的如果时间下降（trig重复计算是关键）
    - 4. 只炮一个 block / 固定数据分布
        - 让 block 非常稀疏（几乎不相交） vs 高度重叠 （几乎全部相交）
        - 如果稀疏场景块，说明 early reject 有效果/分支路径不同
        - 如果稠密场景更慢，说明求交面积部分耗时多
