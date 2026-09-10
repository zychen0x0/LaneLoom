# GT label 可视化

这个目录把 ONCE-3DLanes 和 OpenLane 的 GT 可视化统一放在 `vis/` 下。
两个入口都接收一张数据集原图，自动查找对应 JSON 标签，并输出相机视角叠加图和
地面坐标系鸟瞰图（BEV）。原数据和 `tests/` 中的数据集回归测试不会被修改。

## 文件说明

- `visualize_once_gt.py`：ONCE-3DLanes 命令行入口，支持公开的 `train` 和
  `val` 标签。
- `visualize_openlane_gt.py`：OpenLane 命令行入口。
- `visualize_gt.py`：两个入口共用的标签查找、投影、绘制和输出逻辑，通常不需要
  直接运行。

## 使用方法

在仓库根目录运行：

```bash
python vis/gt_label/visualize_once_gt.py \
  data/once_3dlanes/3D_images/000027/cam01/1616101257899.jpg
```

```bash
python vis/gt_label/visualize_openlane_gt.py \
  data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600.jpg
```

数据不在仓库的 `data/` 目录时，先把 `LANELOOM_DATA_ROOT` 指向同时包含
`openlane/` 和 `once_3dlanes/` 的目录：

```bash
LANELOOM_DATA_ROOT=/path/to/data \
python vis/gt_label/visualize_once_gt.py \
  /path/to/data/once_3dlanes/3D_images/000027/cam01/1616101257899.jpg
```

使用 `--output-root` 可以更改输出根目录：

```bash
python vis/gt_label/visualize_once_gt.py IMAGE \
  --output-root /tmp/laneloom_gt
```

## 输出结构

默认按“数据集 / scene / sample”创建目录，不同数据集和样本不会互相覆盖：

```text
vis/gt_label/
├── once/
│   └── 000027/
│       └── 1616101257899/
│           ├── 01_gt_overlay.jpg
│           └── 02_gt_bev.jpg
└── openlane/
    └── segment-...-with_camera_labels/
        └── 151026502946/
            ├── 01_gt_overlay.jpg
            └── 02_gt_bev.jpg
```

- `01_gt_overlay.jpg`：把标签中的车道线投影到原始相机图像。
- `02_gt_bev.jpg`：使用 LaneLoom 的地面坐标约定绘制 GT；横向为 `x`，向前为
  `y`，显示范围沿用对应数据集配置中的 `point_cloud_range`。

命令执行成功后还会在终端输出 JSON，其中包含源图、源标签、车道数量和所有输出
文件路径，便于脚本化处理。
