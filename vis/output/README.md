# 模型输出可视化

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `infer_openlane_sample.py` | 对一个 OpenLane 样本执行模型推理，输出原图、预测叠加图、BEV 和 3D 视图。 |

## 使用方法

在仓库根目录执行：

```bash
PYTHONPATH=/workspace CUDA_VISIBLE_DEVICES=0 \
python vis/output/infer_openlane_sample.py \
  --image data/openlane/images/validation/segment-12358364923781697038_2232_990_2252_990_with_camera_labels/150707937605591000.jpg \
  --checkpoint work_dirs/laneloom_openlane_image_lidar/exp_date_20260803_081030/checkpoints/best_OpenLaneTemporal_F1_epoch_44.pth
```

默认按“数据集 / 场景 / sample”创建目录：

```text
vis/output/openlane/<scene>/<frame>/
├── 01_original.jpg
├── 02_lane_overlay.jpg
├── 03_bev.png
├── 03_bev.svg
├── 04_3d.png
└── 04_3d.svg
```

用 `--output-root <目录>` 可以修改输出根目录，目录内仍会自动创建
`openlane/<scene>/<frame>`。旧命令中的 `--output-dir <目录>` 仍然可用，
它表示将文件直接写入指定目录。

`03_bev.png` 和 `04_3d.png` 使用 600 DPI，并同时输出适合论文编辑的 SVG。
