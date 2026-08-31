# ONCE-3DLanes 数据准备、训练与评估

本文说明 LaneLoom 当前的 ONCE-3DLanes 支持范围，以及如何得到论文表格中的
F1、Precision、Recall 和 CDE 指标。

参考实现：

- [ONCE-3DLanes 官方数据与评估器](https://github.com/once-3dlanes/once_3dlanes_benchmark)
- [Anchor3DLane++ ONCE 配置](https://github.com/tusen-ai/Anchor3DLane/tree/anchor3dlane%2B%2B/configs_v2/once)
- [Anchor3DLane++ ONCEDataset](https://github.com/tusen-ai/Anchor3DLane/blob/anchor3dlane%2B%2B/mmseg/datasets/lane_datasets/once.py)
- [Anchor3DLane++ ONCE 评估器](https://github.com/tusen-ai/Anchor3DLane/blob/anchor3dlane%2B%2B/mmseg/datasets/tools/eval_once.py)

## 1. 当前实现状态

ONCE 与 OpenLane 使用完全独立的配置和数据集模块：

```text
datasets/once/dataset.py
configs/_base_/datasets/once_datasets.py
configs/_base_/evaluators/once_official.py
configs/main/laneloom_once_image_lidar_Temporal.py
evaluation/once_official_metric.py
tools/test_once.py
```

现有 `OpenLaneDataset`、`openlane_datasets.py`、
OpenLane 主配置现位于 `configs/main/laneloom_openlane_image_lidar_Temporal.py`；
`tools/test.py` 的训练评估方式不变。

当前 ONCE 实现支持：

- `cam01` 图像和 `lidar_roof` 点云联合输入；
- 使用 ONCE `cam_to_velo` 把 LiDAR 与官方相机坐标车道对齐；
- ONCE 相机坐标 `(x, y, z)` 与 LaneLoom 地面坐标 `(x, z, -y)` 双向转换；
- 官方 `val` 上计算 F1、Precision、Recall、CDE；
- 导出官方要求的 `pred_dir/<sequence>/cam01/<frame>.json`。

当前暂时只做单帧训练。ONCE 车道标签没有 LaneLoom 时序 GT 关联所需的
lane track ID，因此配置明确设置 `temporal_sampling_distance=False`，不会把
OpenLane 的时序 GT 补全逻辑错误地套到 ONCE 上。

## 2. 数据准备

先按 ONCE-3DLanes 官方仓库的下载说明获取标注、`cam01` 图像、
`lidar_roof` 点云和 info 压缩包。将原始文件放成下面的形式：

```text
once_3dlanes/
├── ONCE_3DLanes.zip
├── 3D_images/
│   ├── raw_cam01_p0.tar
│   ├── raw_cam01_p1.tar
│   ├── val_cam01.tar
│   └── test_cam01.tar
├── 3D_lidars/
│   ├── raw_lidar_p0.tar.partaa ...
│   ├── raw_lidar_p1.tar.partaa ...
│   ├── val_lidar.tar
│   └── test_lidar.tar
└── 3D_infos/
    ├── raw_small_infos.tar
    ├── raw_medium_infos.tar
    ├── val_infos.tar
    └── test_infos.tar
```

然后执行：

```bash
cd /workspace/LaneLoom

python tools/prepare_once_laneloom.py \
  --once-root /path/to/once_3dlanes
```

脚本会：

- 解压官方标注；
- 直接流式读取 LiDAR 分卷，无需先合并出一个巨大 tar；
- 只展开 p0+p1 训练帧以及完整 `val`/`test`；
- 生成 `LaneLoom_splits/{train_p0p1,val,test}.txt`；
- 核对每张图像、点云、info 和可用标注是否存在。

已完成的大压缩包会通过
`LaneLoom_splits/.prepare_markers/` 记录，中断后可重新执行。
`--overwrite` 会强制重新扫描和替换已展开文件，一般不需要使用。

### 本机已准备的数据

当前数据根目录为：

```text
/workspace/LaneLoom/data/once_3dlanes
```

已经准备好的数据量：

| split | 样本数 | 用途 |
|---|---:|---|
| `train_p0p1` | 40,291 | 当前 LaneLoom 训练 |
| `val` | 3,325 | 本地定量评估 |
| `test` | 7,930 | 无公开 GT，不能在本地计算分数 |

目录结构如下：

```text
once_3dlanes/
├── ONCE_3DLanes/
│   ├── train/<sequence>/cam01/<frame>.json
│   ├── val/<sequence>/cam01/<frame>.json
│   └── list/{train,val,test}.txt
├── 3D_images/data/<sequence>/cam01/<frame>.jpg
├── 3D_lidars/data/<sequence>/lidar_roof/<frame>.bin
├── 3D_infos/data/<sequence>/<sequence>.json
├── LaneLoom_splits/{train_p0p1,val,test}.txt
└── pred_dir/
```

单条 split 记录：

```text
/000027/cam01/1616100800400.jpg
```

对应：

```text
图像:  3D_images/data/000027/cam01/1616100800400.jpg
LiDAR: 3D_lidars/data/000027/lidar_roof/1616100800400.bin
info:  3D_infos/data/000027/000027.json
标签:  ONCE_3DLanes/val/000027/cam01/1616100800400.json
```

本机上述 51,546 个图像/点云样本已经展开完成，当前训练不需要
再次解压。可只做完整性检查：

```bash
python tools/prepare_once_laneloom.py \
  --once-root /workspace/LaneLoom/data/once_3dlanes \
  --verify-only
```

## 3. 训练

单卡快速检查：

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0 \
python tools/train.py \
  configs/main/laneloom_once_image_lidar_Temporal.py \
  --max-epochs 1
```

多卡正式训练：

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29503 \
  tools/train.py \
  configs/main/laneloom_once_image_lidar_Temporal.py \
  --launcher pytorch
```

实验默认写入：

```text
work_dirs/laneloom_once_image_lidar/exp_date_YYYYMMDD_HHMMSS/
```

不要直接加载 OpenLane 最终 checkpoint 继续训练：ONCE 只有一个 lane 类，模型
分类头形状与 OpenLane 的 21 类不同。如需迁移图像或 LiDAR backbone，需要另做
只加载兼容权重的迁移流程。

## 4. 评估并得到论文式分数

```bash
cd /workspace/LaneLoom

PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=29504 \
  tools/test_once.py \
  configs/main/laneloom_once_image_lidar_Temporal.py \
  work_dirs/laneloom_once_image_lidar/exp_date_YYYYMMDD_HHMMSS/checkpoints/best_ONCEOfficial_F1_epoch_XX.pth \
  --launcher pytorch
```

输出包括：

```text
threshold=0.5 F1=xx.xx% P=xx.xx% R=xx.xx% CDE=0.xxxxm
best threshold=... F1=... P=... R=... CDE=...m
```

并写入：

```text
once_evaluation/once_evaluation_results.json
once_evaluation/pred_dir/<sequence>/cam01/<frame>.json
```

主结果固定报告 `score_threshold=0.5`，避免只展示测试后挑出的最好阈值；同时保留
官方 0.10 到 0.95 的阈值扫描和 `best_*` 字段，便于与 Anchor3DLane++ 的输出
逐行核对。匹配的 CD 阈值为论文 Table IV 使用的 `0.3 m`。

如果还希望用官方仓库二次校验导出的 JSON，可执行：

```bash
git clone https://github.com/once-3dlanes/once_3dlanes_benchmark.git
cd once_3dlanes_benchmark

python eval.py \
  --gt_path /workspace/LaneLoom/data/once_3dlanes/ONCE_3DLanes/val \
  --pred_path /path/to/once_evaluation/pred_dir
```

官方脚本是 Python 3.6/NumPy 旧环境代码，在当前 NumPy 1.26 中使用了已移除的
`np.float`，还依赖仓库自带的 `jarvis` wheel 和 `munkres`。LaneLoom 的
`ONCEOfficialMetric` 保留其匹配与 CDE 行为，但改用当前依赖可运行的实现。

## 5. 为什么当前分数不能直接与图中 81.25 比较

图中的 Anchor3DLane++ 结果使用完整 ONCE-3DLanes 训练集：201,856 个训练帧。
当前 LaneLoom 只使用 p0+p1 的 40,291 帧，约为完整训练集的 20%。此外，图中
Anchor3DLane++ 是纯相机模型；LaneLoom 配置使用图像 + LiDAR。因此当前实验可以
比较同一 `val` 和同一评估协议下的指标，但不是相同训练数据量、相同输入模态的
严格复现。

若要做论文级公平对比，需要：

1. 下载并展开 raw p2-p9 对应的 LiDAR；本机目前只有 p0、p1 LiDAR 分卷；
2. 为官方 `ONCE_3DLanes/list/train.txt` 的全部 201,856 帧准备图像、LiDAR 和 info；
3. 将 `once_datasets.py` 中 `train_split='train_p0p1'` 改为
   `train_split='train'`；
4. 保持官方 `val`、0.3 m CD 阈值、输入分辨率和置信度报告方式不变；
5. 另建纯相机 ONCE 配置，才可与 Anchor3DLane++ 的纯相机行做严格同模态比较。

`train_cam01.tar`、`train_lidar.tar` 只有约 9,977 帧，并不是 201,856 帧的完整
ONCE-3DLanes 训练输入，不能用它们代替 raw p0-p9。

## 6. 可学习的外部脚本

Anchor3DLane++ 仓库确实包含完整 ONCE 路径，最值得参考的是：

- `tools/convert_datasets/once.py`：将官方 JSON 转成 Anchor3DLane 的 dense cache；
- `mmseg/datasets/lane_datasets/once.py`：读取图像、转换相机/地面坐标、导出预测；
- `mmseg/datasets/tools/eval_once.py`：ONCE F1/P/R/CDE；
- `mmseg/apis/test_once.py`：推理、保存 JSON、调用 evaluator；
- `configs_v2/once/anchor3dlane++_r18.py` 和
  `anchor3dlane++_r50x2.py`：训练与测试参数。

这些文件适合学习协议，但不能直接复制进 LaneLoom：Anchor3DLane++ 的 ONCE 配置
只加载相机，不读取 LiDAR；其 `once.py --generate` 在当前分支还有参数调用问题，
且整套代码依赖 mmsegmentation 0.26/Python 3.7 的旧运行环境。
