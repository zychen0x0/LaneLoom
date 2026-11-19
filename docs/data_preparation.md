# 数据准备

按所用配置准备 **OpenLane** 或 **ONCE-3DLanes**。默认数据根目录是 LaneLoom 仓库下的 `data/`。以下命令从 LaneLoom 仓库根目录执行，Waymo 转换步骤除外。

```bash
# 数据放在其他磁盘时，先设置 export LANELOOM_DATA_ROOT=/path/to/data
DATA_DIR="${LANELOOM_DATA_ROOT:-$PWD/data}"
```

## OpenLane

1. 下载 OpenLane-V1 的前视图像 `images` 和 v1.2 标注 `lane3d_1000`，解压到 `$DATA_DIR/openlane/`。[OpenLane 下载说明](https://github.com/OpenDriveLab/OpenLane/blob/main/data/README.md)
2. 下载 Waymo Perception v1.4.3 的 training 和 validation TFRecord（798 个训练序列和 202 个验证序列），放到 `OpenPCDet/data/waymo/raw_data/`。所需文件名对应 OpenPCDet 的 `data/waymo/ImageSets/train.txt` 和 `val.txt`。[Waymo 下载](https://waymo.com/open/download/) · [OpenPCDet 数据说明](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#waymo-open-dataset)
3. 在独立环境中安装 OpenPCDet 和 Waymo SDK，然后进入 OpenPCDet 仓库根目录执行转换。[OpenPCDet 安装](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) · [Waymo SDK 安装](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md#waymo-open-dataset)

   ```bash
   python -m pcdet.datasets.waymo.waymo_dataset \
     --func create_waymo_infos \
     --cfg_file tools/cfgs/dataset_configs/waymo_dataset_multiframe.yaml
   ```

4. 将生成的 `OpenPCDet/data/waymo/waymo_processed_data_v0_5_0/` 移到 `$DATA_DIR/waymo/` 并改名为 `waymo_processed_data`，或在同一位置创建指向它的软链接。确认目录如下。

   ```text
   $DATA_DIR/
   ├── openlane/
   │   ├── images/{training,validation}/...
   │   └── lane3d_1000/{training,validation,test}/...
   └── waymo/waymo_processed_data/<segment>/
       ├── 0000.npy
       └── <segment>.pkl
   ```

## ONCE-3DLanes

主配置使用官方完整训练划分。下载车道标注 ZIP，以及 ONCE 的 `cam01` 图像、LiDAR 和序列 info。按下列目录放置原始压缩包，保留全部 LiDAR 分卷。[ONCE-3DLanes 标注](https://drive.google.com/file/d/16_Tw0K55yR-3sJf8-lBbymGHcq0H7Zjx/view) · [ONCE 图像、LiDAR 和 info](https://once-for-auto-driving.github.io/download.html#downloads)

```text
$DATA_DIR/once_3dlanes/
├── ONCE_3DLanes.zip
├── 3D_images/
│   ├── raw_cam01_p0.tar 至 raw_cam01_p9.tar
│   └── val_cam01.tar、test_cam01.tar
├── 3D_lidars/
│   ├── raw_lidar_p0.tar.part* 至 raw_lidar_p9.tar.part*（各组全部分卷）
│   └── val_lidar.tar、test_lidar.tar
└── 3D_infos/
    ├── raw_small_infos.tar、raw_medium_infos.tar、raw_large_infos.tar
    └── val_infos.tar、test_infos.tar
```

运行以下命令。脚本会读取分卷、解压所需帧并检查文件，默认在输出磁盘预留 20 GiB。中断后可运行同一条命令继续。

```bash
python tools/prepare_once_laneloom.py --full-train --once-root "$DATA_DIR/once_3dlanes"
```

看到 `ONCE preparation verification passed.` 即表示文件检查通过。解压后的目录如下。

```text
$DATA_DIR/once_3dlanes/
├── list/{train,val,test}.txt
├── {train,val}/<sequence>/cam01/<frame>.json
├── 3D_images/<sequence>/cam01/<frame>.jpg
├── 3D_lidars/<sequence>/lidar_roof/<frame>.bin
└── 3D_infos/<sequence>/<sequence>.json
```

数据就绪后，按对应配置启动实验。[训练与评估](train_eval.md)
