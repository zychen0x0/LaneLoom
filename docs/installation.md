# 安装

已验证环境：Ubuntu 22.04、Python 3.10、CUDA 11.8、PyTorch 2.1.0。

## 1. 创建环境

```bash
git clone https://github.com/zychen0x0/LaneLoom.git
cd LaneLoom

conda create -n laneloom python=3.10 -y
conda activate laneloom
python -m pip install --upgrade pip
```

已有源码时从 `conda create` 开始即可。

## 2. 安装依赖

```bash
python -m pip install \
  torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118

python -m pip install -U openmim
mim install "mmengine==0.10.7" "mmcv==2.1.0" \
  "mmdet==3.2.0" "mmdet3d==1.4.0"

python -m pip install -r requirements.txt
```

## 3. 配置项目路径

每个新终端执行一次：

```bash
cd /path/to/LaneLoom
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
```

检查安装：

```bash
python -c "import torch, mmcv, mmdet3d, LaneLoom; print(torch.__version__, mmcv.__version__, mmdet3d.__version__)"
python tools/train.py --help
```

正常情况下第一条命令输出的版本以 `2.1.0`、`2.1.0`、`1.4.0` 开头。
