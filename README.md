# ⚽️ Football Video AI Analyzer

简体中文｜[English](README_EN.md)

该项目当前处于研究开发阶段，已实现足球比赛视频中的场地检测（即相机标定）、人球识别和追踪。最终目标为威胁度量化和整体战术评估，由此自动生成集锦和赛后分析

<div align="center">
<figure>
    <img src="docs/main.png" alt="Overview">
    <figcaption><em>总体效果</em></figcaption>
</figure>
</div>

## 流程

### I. 球场检测 (相机标定)

`core/broadtrack_calib`（C++ 核心在 `core/BroadTrack`）

使用 [BroadTrack](https://arxiv.org/abs/2412.01721) 的架构：
- 主要使用 Line Segmentation Model，辅助使用Keypoint Model，来检测球场;
- 通过光流法、pan 网格搜索 + 非线性优化、相机锚点优化相机标定;
- 通过 IoU 评估检测结果。

追踪循环通过 pybind11 编译为 Python 扩展：Ceres 光束法平差、line-IoU 评分与
Lucas-Kanade 光流关联都留在 C++，两个检测网络则复用进程中已有的 Python `torch`
（不重复链接第二份 libtorch）。`BroadTrackCalib` 可直接替换原先的 `PnLCalib`：
返回字段、坐标系与绘图接口都保持一致。详见
[`core/BroadTrack/README.md`](core/BroadTrack/README.md)。
![Pitch Detection](docs/pnl.png)

### II. 球员识别跟踪

`core/player_tracker.py`

![Player Tracker in BEV](docs/player_track_bev.png)

1. 通过 Object Detection Model 进行球员/裁判分类

2. 用 BoT-SORT 进行跟踪，会自动处理 GMC, ReId, Kalman Filter

(3.) 计划通过位置特征 (目前仅依靠x、y坐标) 识别守门员候选人

### III. 足球识别跟踪

`core/ball_tracker.py`

<div align="center">
<figure>
    <img src="docs/ball_tracker.png" alt="ball_tracker">
    <figcaption><em>如图，红色为模型检测，蓝色为滤波处理后结果</em></figcaption>
</figure>
</div>

1. 通过 Camera Calibration 得到的P，将所有足球放到BEV坐标系中
2. 计算所有检测点距离之前球轨迹的 Mahalanobis Distance
3. 使用 Chi Square 判断检测是否有效，选出最优的检测
4. 用 Kalman Filter 预测，用于平滑化轨迹以及防止阻挡带来的问题

---

## 第一次启动？

> 测试比赛视频已内置于 `input_vids/` 中

1. 准备足球和球员的检测模型：从 [roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) 
下载数据集并训练（仅测试过 YOLO11）。将训练好的权重文件放入 `models` 文件夹，并在 `main.py` 中更改模型路径
1. 从 [github](https://github.com/evs-broadcast/BroadTrack/tree/main/models) 下载 BroadTrack 的两个 TorchScript 检测模型
—— `nbjw_keypoint_model.pt` 与 `tvcalib_model.pt`。放入 `models/`
1. 创建环境并构建标定扩展：
   ```bash
   conda env create -f environment.yml && conda activate football
   python scripts/build_minimal_opencv.py   # 只需一次，约 20 分钟
   python scripts/build_minimal_ceres.py    # 只需一次，约 5 分钟
   python scripts/build_native.py
   ```
   两个 `build_minimal_*` 会在本地编译 OpenCV 和 Ceres，而不是用现成的包 ——
   现有各家的这两个库都会往进程里再拉一份 OpenMP，而 PyTorch 的 wheel 已经自带
   一份，两者相遇时解释器会在导入阶段直接 abort。详见
   [`core/BroadTrack/README.md`](core/BroadTrack/README.md)。
   Linux 上还需要 RapidJSON 与 Boost 头文件，以及 Ceres 依赖的 `metis`。

1. 在 main.py 中修改几个 weight 路径和 `VIDEO_PATH` 为你的文件路径
1. 运行 main.py，弹出 cam视角、bev视角 窗口

---

## TODO

更详细的问题列表位于 [problems.md](docs/problems.md)

- [ ] 一键下载模型脚本
- [ ] 自动生成守门员处理球集锦
- [ ] 威胁度评分（结合位置、球速、对方球员密度等特征）
- [ ] 守门员姿态分析、运动速度轨迹分析
- [ ] 沉浸式看球（？）VR/gaming

---

## 参考与鸣谢

### 球场检测
- [SoccerNet Calibration](https://github.com/SoccerNet/sn-calibration)
- [BroadTrack](https://github.com/evs-broadcast/BroadTrack): 在其原版基础上进行优化并采用代码
   ```bibtex
   @inproceedings{Magera2025BroadTrack,
     title = {BroadTrack: Broadcast Camera Tracking for Soccer},
     author = {Magera, Floriane and Hoyoux, Thomas and Barnich, Olivier and Van Droogenbroeck, Marc},
     booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
     month = {February},
     year = {2025},
     address = {Tucson, Arizona, USA}
   }
   ```

### 数据处理与分析
- [SoccermaticsForPython](https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython)
- [Friends of Tracking](https://www.youtube.com/@friendsoftracking755): 极有帮助的学习资源
- [wyscout](https://apidocs.wyscout.com?version=3)

---

## 🤝 贡献与联系

欢迎提交 issues 或 PR，亦可通过 hungryhenry101@outlook.com 进行联系