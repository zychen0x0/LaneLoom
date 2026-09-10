# 数据准备

LaneLoom 支持 OpenLane 和 ONCE-3DLanes。默认数据根目录为 `data/`；数据放在
其他磁盘时设置：

```bash
export LANELOOM_DATA_ROOT=/path/to/laneloom_data
```

## OpenLane

### 1. 下载

- 从 [OpenLane](https://github.com/OpenDriveLab/OpenLane) 下载 `images` 和
  `lane3d_1000`。
- 从 [Waymo Open Dataset](https://waymo.com/open/download/) 下载与 OpenLane
  对应的 training 和 validation TFRecord。

### 2. 转换 Waymo LiDAR

转换在独立的 OpenPCDet 环境中完成。先按其
[安装说明](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)
安装 OpenPCDet 和 Waymo SDK，再按
[Waymo 数据准备说明](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#waymo-open-dataset)
放置 TFRecord，并在 OpenPCDet 根目录执行：

```bash
python -m pcdet.datasets.waymo.waymo_dataset \
  --func create_waymo_infos \
  --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml
```

LaneLoom 直接读取 OpenPCDet 生成的逐帧 `.npy` 和逐序列 `.pkl`，无需再转换。

### 3. 建立目录

```bash
cd /path/to/LaneLoom
LANELOOM_DATA_DIR="${LANELOOM_DATA_ROOT:-$PWD/data}"
mkdir -p "$LANELOOM_DATA_DIR/openlane" "$LANELOOM_DATA_DIR/waymo"
ln -s /path/to/OpenLane/images "$LANELOOM_DATA_DIR/openlane/images"
ln -s /path/to/OpenLane/lane3d_1000 "$LANELOOM_DATA_DIR/openlane/lane3d_1000"
ln -s /path/to/OpenPCDet/data/waymo/waymo_processed_data_v0_5_0 \
  "$LANELOOM_DATA_DIR/waymo/waymo_processed_data"
```

最终目录：

```text
$LANELOOM_DATA_DIR/
├── openlane/
│   ├── images/{training,validation}/...
│   └── lane3d_1000/{training,validation,test}/...
└── waymo/waymo_processed_data/
    └── segment-*/
        ├── 0000.npy
        └── segment-*.pkl
```

## ONCE-3DLanes

### 1. 下载

从 [ONCE-3DLanes](https://github.com/once-3dlanes/once_3dlanes_benchmark)
下载标注、`cam01` 图像、`lidar_roof` 点云和 info 文件，保持压缩包结构：

```text
once_3dlanes/
├── ONCE_3DLanes.zip
├── 3D_images/{raw_cam01_p0.tar,raw_cam01_p1.tar,val_cam01.tar,test_cam01.tar}
├── 3D_lidars/{raw_lidar_p0.tar.part*,raw_lidar_p1.tar.part*,val_lidar.tar,test_lidar.tar}
└── 3D_infos/{raw_small_infos.tar,raw_medium_infos.tar,val_infos.tar,test_infos.tar}
```

当前主配置使用 p0+p1 训练子集。

### 2. 一键转换并校验

```bash
cd /path/to/LaneLoom
LANELOOM_DATA_DIR="${LANELOOM_DATA_ROOT:-$PWD/data}"
python tools/prepare_once_laneloom.py \
  --once-root "$LANELOOM_DATA_DIR/once_3dlanes"
```

脚本会按官方划分解压所需文件并生成 `LaneLoom_splits`。中断后可直接重跑。

最终目录：

```text
$LANELOOM_DATA_DIR/once_3dlanes/
├── ONCE_3DLanes/{train,val,list}/...
├── 3D_images/<sequence>/cam01/*.jpg
├── 3D_lidars/<sequence>/lidar_roof/*.bin
├── 3D_infos/<sequence>/<sequence>.json
└── LaneLoom_splits/{train_p0p1,val,test}.txt
```

只检查已准备的数据：

```bash
LANELOOM_DATA_DIR="${LANELOOM_DATA_ROOT:-$PWD/data}"
python tools/prepare_once_laneloom.py \
  --once-root "$LANELOOM_DATA_DIR/once_3dlanes" \
  --verify-only
```
