# LaneLoom 补充实验操作手册 v2（省算力版）

这一版把原来"每个实验完整训 50 轮、约 8 天"改成三档：今天就能出的推理分析、四分之一数据的训练、不跑的实验。

## 一、替换代码包

1. 删掉仓库根目录下原来的 `LaneLoom/ablation_extras/` 文件夹。如果里面已经有 `return_back/`，先把它挪到别处保存。
2. 解压新的 `ablation_extras.zip`，把得到的 `ablation_extras` 文件夹放回仓库根目录，和 `configs`、`tools` 同一级。
3. 自检，在仓库根目录执行：

```bash
cd /path/to/LaneLoom
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"
python -m pytest ablation_extras/tests -q
```

最后一行应该是 `10 passed`。不是的话，把整段输出发回来。

这一版改了三处：

- 训练中的验证从每轮一次改成每 10 轮一次。原配置每轮都在完整验证集上跑一遍多帧推理，这部分占了不少训练时间。
- 训练完评测最后一轮的 checkpoint，不再从验证集上挑最好的一轮。
- 新增 `QUARTER=1` 模式：50 轮的学习率曲线不变，训练时每 4 帧取 1 帧，迭代次数约为原来的四分之一；评测仍用完整验证集和六个场景。

## 二、第一档：今天就跑，不用训练

用现有的 65.9 多帧 checkpoint 和 63.1 单帧 checkpoint。三项加起来几个小时。

**E6：每帧有几种车道形状，只用 CPU，约 30 分钟。**

```bash
bash ablation_extras/run_analysis.sh e6 configs/main/laneloom_openlane_image_lidar_temporal.py none train
```

**E4：初始查询覆盖率，约 1 小时。** 这次脚本会同时算"同样 101 个横向位置上的直线"的覆盖率。直线正好就是 e1 的初始查询，所以 CQG 和直线的覆盖率对比不用训练就能拿到。

```bash
bash ablation_extras/run_analysis.sh e4 configs/main/laneloom_openlane_image_lidar_temporal.py /path/to/65.9的checkpoint.pth full
```

**E3：测速，每次约 10 分钟。** 多帧和单帧在同一张卡上各测一次：

```bash
bash ablation_extras/run_analysis.sh e3 configs/main/laneloom_openlane_image_lidar_temporal.py /path/to/多帧checkpoint.pth mf
bash ablation_extras/run_analysis.sh e3 configs/main/laneloom_openlane_image_lidar.py /path/to/单帧checkpoint.pth sf
```

实验室如果有 V100，请在 V100 上再各测一次，tag 写成 `mf_v100` 和 `sf_v100`。DV-3DLane 的速度是在 V100 上测的，同卡的数可以直接比。

每条命令最后会打印 `SUCCESS` 和一个路径。把 `ablation_extras/return_back/` 下面这几样打包发回来：

- `e6_shape_modes_train.json`、`e6_shape_modes_train.md`
- `e4_q0coverage_full` 整个文件夹
- `e3_fps_mf.json`、`e3_fps_mf.md`、`e3_fps_sf.json`、`e3_fps_sf.md`
- 三个 `*_console.txt`

## 三、第二档：四分之一数据训练，四个实验

只跑 e0、e1、e2a、e2b。四个实验用同一套缩减设置，彼此可以直接比较；e0 是这套设置下的完整模型，作为对照，不替代表里的 65.9。

先冒烟一次，确认新模式能跑通（约 10 分钟）：

```bash
SMOKE=1 bash ablation_extras/run_experiment.sh e0
```

打印 `SUCCESS` 后开始正式跑。有几台机器就并行几个，每个实验占 2 张卡，端口各不相同。一台机器 4 张卡时：

```bash
QUARTER=1 GPUS=0,1 PORT=29611 nohup bash ablation_extras/run_experiment.sh e0  > e0q.out 2>&1 &
QUARTER=1 GPUS=2,3 PORT=29612 nohup bash ablation_extras/run_experiment.sh e1  > e1q.out 2>&1 &
```

这两个跑完，或者另一台机器空出来，再跑：

```bash
QUARTER=1 GPUS=0,1 PORT=29613 nohup bash ablation_extras/run_experiment.sh e2a > e2aq.out 2>&1 &
QUARTER=1 GPUS=2,3 PORT=29614 nohup bash ablation_extras/run_experiment.sh e2b > e2bq.out 2>&1 &
```

优先级是 e1 > e0 > e2a = e2b。只有一台 2 卡机器的话，先跑 e0 和 e1。

训练开始约半小时后，执行 `tail -n 3 e0q.out`，日志行里有 `eta` 字段，把预计剩余时间告诉我。

| 实验 | 改了什么 | 回应什么 |
|---|---|---|
| e0 | 完整模型 | 同一缩减设置下的对照行 |
| e1 | CQG 模板固定成直线 | CQG 的直接消融 |
| e2a | 只用历史图像，点云只用当前帧 | 拆开时序增益里图像的部分 |
| e2b | 只用累积点云，只采当前帧图像 | 拆开时序增益里点云的部分 |

每个跑完，把 `ablation_extras/return_back/e*_quarter_return` 文件夹打包发回来。

## 四、第三档：不跑

e2c（时序卷积换平均）、e5（三层解码器）、完整 50 轮版本都不跑。审稿意见真的问到再说。
