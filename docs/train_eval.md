# 训练与评估

先进入仓库并配置 Python 路径：

```bash
cd /path/to/LaneLoom
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
```

主配置：

| 数据集 | 配置 |
|---|---|
| OpenLane 单帧 | `configs/main/laneloom_openlane_image_lidar.py` |
| OpenLane 时序 | `configs/main/laneloom_openlane_image_lidar_temporal.py` |
| ONCE-3DLanes p0+p1 | `configs/main/laneloom_once_image_lidar_temporal.py` |

## 训练

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  configs/main/laneloom_openlane_image_lidar_temporal.py
```

双卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 --master_port=29501 \
  tools/train.py \
  configs/main/laneloom_openlane_image_lidar_temporal.py \
  --launcher pytorch
```

训练 ONCE 时将配置替换为
`configs/main/laneloom_once_image_lidar_temporal.py`。输出保存在：

```text
work_dirs/<experiment>/exp_date_YYYYMMDD_HHMMSS/
├── checkpoints/
├── logs/
└── Tensorboard/
```

单轮训练或加载已有权重：

```bash
python tools/train.py CONFIG --max-epochs 1
python tools/train.py CONFIG --load-from CHECKPOINT
```

`--load-from` 只加载模型权重，并以新优化器开始一次新实验。

## 评估 OpenLane

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/main/laneloom_openlane_image_lidar_temporal.py \
  /path/to/checkpoint.pth \
  --case validation
```

双卡评估：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 --master_port=29502 \
  tools/test.py \
  configs/main/laneloom_openlane_image_lidar_temporal.py \
  /path/to/checkpoint.pth \
  --case validation \
  --launcher pytorch
```

`--case` 可选 `validation`、`test`、`all`，也可指定
`curve_case`、`intersection_case` 等单个官方场景。评估结果默认写入 checkpoint
所在实验的 `evaluation/evaluation_results.json`。

## 评估 ONCE-3DLanes

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_once.py \
  configs/main/laneloom_once_image_lidar_temporal.py \
  /path/to/checkpoint.pth
```

结果写入 `once_evaluation/once_evaluation_results.json`，官方格式预测写入
`once_evaluation/pred_dir/`。

## TensorBoard

```bash
tensorboard --logdir work_dirs --host 0.0.0.0 --port 6006
```
