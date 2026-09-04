# 数据增强可视化

这里放置数据增强相关的手动可视化工具，不再把生成图片的逻辑混在 pytest
测试实现中。目前支持 OpenLane；输出目录预留了数据集层级，后续加入 ONCE 等
数据集时不会混在一起。

## 文件说明

- `visualize_network_input.py`：运行当前 OpenLane 训练 pipeline，展示 resize 原图
  与模型实际收到的时序图像，并输出所选增强方案和 tensor 信息。
- `visualize_named_recipes.py`：把同一张图分别应用 `original`、`brightness`、
  `fog`、`blur`、`occlusion`、`random`，生成带名称的对比图。
- `_common.py`：两个命令共用的配置加载、样本定位、画布和输出目录工具，通常
  不直接运行。

## 使用方法

在仓库根目录执行：

```bash
python vis/data_aug/visualize_network_input.py \
  data/openlane/images/training/segment-16625429321676352815_1543_860_1563_860_with_camera_labels/150809235044180100.jpg
```

如果不传图片，`visualize_network_input.py` 会从训练集选择一个具有完整时序帧的
样本。使用 `--seed 0` 可以稳定复现样本选择和随机增强。

查看同一图片上的全部命名增强：

```bash
python vis/data_aug/visualize_named_recipes.py \
  data/openlane/images/training/segment-16625429321676352815_1543_860_1563_860_with_camera_labels/150809235044180100.jpg \
  --seed 20260726
```

两个命令都支持 `--output-root` 指定新的输出根目录。

## 输出位置

默认按“数据集 / scene / sample”创建目录：

```text
vis/data_aug/<dataset>/<scene>/<sample>/
├── 01_network_input.jpg
├── 02_resized_vs_network_input.jpg
├── 03_named_recipes.jpg
├── network_input_report.json
└── named_recipes_report.json
```

例如：

```text
vis/data_aug/openlane/
└── segment-16625429321676352815_1543_860_1563_860_with_camera_labels/
    └── 150809235044/
        ├── 01_network_input.jpg
        ├── 02_resized_vs_network_input.jpg
        └── 03_named_recipes.jpg
```

两个脚本可以单独运行，因此目录中只会出现当前命令生成的图片和报告。重复运行
同一个 sample 会更新同名输出，不会影响其他数据集、scene 或 sample。
