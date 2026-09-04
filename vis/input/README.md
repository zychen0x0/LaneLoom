# 数据可视化

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `visualize_sample.py` | 可视化一个 OpenLane 或 ONCE sample，输出原图、GT 叠加图、数据增强图和着色点云。 |
| `export_ipm.py` | 将 OpenLane 相机图像导出为标定或手工四点 IPM。 |
| `export_depth_anything_v2.py` | 使用 Depth Anything V2 Small 导出相对深度图。 |
| `export_uv_gt_mask.py` | 将 OpenLane UV 标注导出为二值 PNG 和 SVG mask。 |
| `export_aligned_pointpainting.py` | 将历史 PointPainting 点云对齐到当前帧并绘图。 |
| `export_pointpainting_front.py` | 导出 PointPainting 前后对比的正视图。 |
| `export_lidar_bev.py` | 将 PLY 点云导出为俯视 BEV 图。 |

每个脚本都可以用 `python vis/input/<脚本名> --help` 查看完整参数。三个点云
辅助脚本的最简命令模板如下：

```bash
python vis/input/export_aligned_pointpainting.py <painted.ply> \
  --pose-info <scene.pkl> \
  --current-frame-id <frame-id> \
  --output <output.png>

python vis/input/export_pointpainting_front.py <painted.ply> \
  --output-dir <output-dir>

python vis/input/export_lidar_bev.py <points.ply> \
  --output-dir <output-dir>
```

## 基础用法

`visualize_sample.py` 会自动识别 OpenLane/ONCE，并按“数据集 / 场景 /
sample”组织结果：

```bash
python vis/input/visualize_sample.py \
  /workspace/LaneLoom/data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600.jpg
```

输出位置：

```text
vis/input/<dataset>/<scene>/<frame>/
├── 01_original.jpg
├── 02_gt_overlay.jpg
├── 03_augmentations.jpg
└── 04_points.ply
```

# 导出论文示意图用的 RGB IPM

推荐使用当前帧 JSON 中的内外参生成平面 IPM。默认窄幅版本显示车体左右
`[-4.5 m, 4.5 m]`、前方 `[10 m, 60 m]`，远处在上、近处在下：


python vis/input/export_ipm.py \
  data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600.jpg


# 输出位置：


vis/input/openlane/<scene>/<frame>/05_ipm_calibrated.png


上面的默认范围适合窄幅流程图，但会裁掉当前帧位于左侧约 `x=-7.7 m`
和 `x=-9.3 m` 的道路边界。要输出当前帧更完整的道路区域，使用：


python vis/input/export_ipm.py \
  data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600.jpg \
  --x-range -10 6 \
  --y-range 7.5 80 \
  --size 1920 3200 \
  --dpi 600 \
  --output vis/input/openlane/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600/05_ipm_calibrated_full_road.png


这个论文版本为 1920×3200 像素并写入 600 DPI 元数据。深度图保持原始
1920×1280 像素（约 81 mm 宽时为 600 DPI），避免用插值制造并不存在的细节。
UV GT mask 推荐按 4 倍尺寸重新栅格化，并同时输出真正的矢量 SVG：


python vis/input/export_depth_anything_v2.py \
  data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600.jpg \
  --dpi 600

python vis/input/export_uv_gt_mask.py \
  data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600.jpg \
  --size 3840 2560 \
  --line-width 20 \
  --dpi 600


完整道路版本的底部会有少量黑色三角区。这些 BEV 单元超出了相机在近处的
横向视野，没有可采样的 RGB 像素。天空也不属于地面平面；树木、墙面和路灯等
非地面物体只会被拉伸，不能像原图一样完整保留。

如果只想人工调整论文框架图，也可以使用四点法进行视觉上的梯形矫正。下面的点已经从
1536×1024 坐标按 1.25 倍换算到当前 1920×1280 原图：

```bash
python vis/input/export_ipm.py \
  data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600.jpg \
  --mode manual \
  --src 737.5 712.5 1062.5 712.5 1562.5 1262.5 275 1262.5 \
  --size 350 700
```

四点顺序必须是左上、右上、右下、左下。这组人工点只保证梯形矫正，不保证对应
地面上的米制矩形，因此生成结果不能称为严格 BEV。实验或正文中若声称投影到真实
BEV 空间，应使用标定法，并在图注中注明地面平面假设和米制范围。


# sample

当前帧
data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502946490600.jpg


5 米历史帧（实际位移约 6.326 m，frame 94）
data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502906493100.jpg


10 米历史帧（实际位移约 11.068 m，frame 91）
data/openlane/images/validation/segment-5373876050695013404_3817_170_3837_170_with_camera_labels/151026502876496800.jpg
