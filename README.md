# piper_openpi使用文档

## 项目概览

本项目旨在利用piper机械臂采集示教数据、使用pi0.5进行微调，并最终在piper机械臂上进行测试。

## Quick Start

TODO

## 项目结构
```
piper_openpi/
├── openpi/                     # openpi源代码
├── can_multi_activate.sh       # 机械臂激活脚本
├── piper_test.py               # 机械臂测试脚本
├── multi-cam.py                # 相机测试脚本
├── piper_data_collect.py       # 数据采集脚本
├── align.py                    # 机械臂数据与图像数据对齐脚本
├── convert.py                  # 将对齐后的数据转换为Lerobot格式，以便openpi训练
├── piper_dual_replay.py        # 数据集复播脚本
├── infer_piper_dual_replay.py  # 模型在训练集上推理脚本
├── infer_piper_dual.py         # 模型在真机上实时推理脚本
├── utils.py                    # 包含模型加载、模型推理、机械臂执行等工具函数
└── temp.txt                    # 记录常用指令
```

## 详细功能与使用示例

### can_multi_activate.sh

在运行piper_data_collect.py前，必须先执行此脚本以激活机械臂（可能需要sudo权限）
```bash
bash can_multi_activate.sh
```

### piper_test.py

若希望确认机械臂是否成功连接并激活，可运行
```bash
python piper_test.py
```
并观察机械臂是否如预期移动。

### multi-cam.py

若希望确认相机是否成功连接，可运行
```bash
python multi-cam.py
```

该脚本通过realsense接口获取相机传感器的数据，并通过opencv将多个图像拼接并显示在屏幕上。

### piper_data_collect.py

机械臂和相机数据采集脚本，使用multiprocessing库实现了2个线程异步采集机械臂数据和相机数据。使用样例：

```bash
python piper_data_collect.py --task_name pick_block --start_episode 0 --fps 30
```

- task_name: 采集的数据将会存放在/home/tengenx2204/workspace/mozihao/<task_name>下
- start_episode: 可通过<start_episode>从断点恢复采集
- fps: 指定了相机的采集频率

机械臂数据和相机数据分别保存为两份.h5文件

### align.py

机械臂和相机数据对齐脚本。

- 通过数据中记录的timestamps，二分查询距离当前图像帧最近的机械臂数据并保留
- 将图像降采样至openpi需要的224x224，保存为JPEG格式，存储在当前episode的/frames文件夹下
- 将下一帧的state记录为当前帧的action
- 112行（可选）降采样50%数据

```bash
python align.py --data_dir /home/tengenx2204/workspace/mozihao/Data/pick_block --start_episode 0
```

- data_dir: 指定数据集路径
- start_episode: 从<start_episode>开始转换

## convert.py

将align后的数据集转换为Lerobot格式。

```bash
python convert.py
```

请在convert_dataset()函数的开头修改<original_data_dir>为你希望转换的数据集路径，并将<repo_id>修改为你希望转换后存储数据集的文件夹名。

