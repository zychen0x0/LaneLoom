## 训练


```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  configs/main/laneloom_openlane_image_lidar.py --launcher pytorch
```


```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  configs/main/laneloom_openlane_image_lidar_temporal.py \
  --launcher pytorch \
  --load-from /path/to/best_OpenLaneTemporal_F1_epoch.pth


CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 tools/train.py \
  configs/main/laneloom_once_image_lidar_temporal.py \
  --launcher pytorch \
  --load-from /path/to/best_ONCEOfficial_F1_epoch.pth
```

`--load-from` 加载模型权重并新建优化器。每次训练的输出目录如下。

```text
work_dirs/<experiment>/exp_date_YYYYMMDD_HHMMSS/
├── checkpoints/
├── logs/
└── Tensorboard/
```

## 评估 OpenLane

单帧和多帧都用 `tools/test.py` 进行官方评估。按 checkpoint 选择对应的配置。

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/main/laneloom_openlane_image_lidar.py \
  /path/to/best_OpenLaneOfficial_F1_epoch_24.pth \
  --case validation
```

双卡评估多帧权重。

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 --master_port=29502 \
  tools/test.py \
  configs/main/laneloom_openlane_image_lidar_temporal.py \
  /path/to/best_OpenLaneTemporal_F1_epoch_20.pth \
  --case validation \
  --launcher pytorch
```

`--case` 可选 `validation`、`test`、`all`，也可指定
`curve_case`、`intersection_case` 等单个官方场景。官方评估使用当前帧 GT，默认置信度阈值为 0.50。结果默认写入 checkpoint 所在实验的 `evaluation/evaluation_results.json`。

## 评估 ONCE-3DLanes

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_once.py \
  configs/main/laneloom_once_image_lidar_temporal.py \
  /path/to/best_ONCEOfficial_F1_epoch_17.pth
```

结果写入 `once_evaluation/once_evaluation_results.json`，官方格式预测写入
`once_evaluation/pred_dir/`。


## TensorBoard

```bash
tensorboard --logdir work_dirs --host 0.0.0.0 --port 6006
```
