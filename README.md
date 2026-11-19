# LaneLoom

## 论文实验配置

| 实验 | 配置 | 输入 |
|---|---|---|
| OpenLane 单帧 | [laneloom_openlane_image_lidar.py](configs/main/laneloom_openlane_image_lidar.py) | 当前帧前视图像和 LiDAR |
| OpenLane 多帧 | [laneloom_openlane_image_lidar_temporal.py](configs/main/laneloom_openlane_image_lidar_temporal.py) | 当前帧及两帧历史的前视图像和 LiDAR |
| ONCE-3DLanes 多帧 | [laneloom_once_image_lidar_temporal.py](configs/main/laneloom_once_image_lidar_temporal.py) | 当前帧及两帧历史的 cam01 前视图像和 LiDAR |

[训练与评估](docs/train_eval.md)列出了每个入口对应的训练日志、初始化权重、批量大小和评估口径。

## 仓库目录

- `configs/main/` 存放上表的三个训练和评估入口。三个入口都只从 `configs/_base_/` 继承配置。
- `configs/_base_/` 存放共享的数据集、模型、训练计划和评估器配置。
- [历史实验与诊断结果](configs/ablations/README.md) 记录早期实验的训练和评估口径。
- `work_dirs/` 存放本地生成的训练日志、权重和评估结果，由 Git 忽略。

开始运行前，请按[环境安装](docs/installation.md)和[数据准备](docs/data_preparation.md)配置工作目录。
