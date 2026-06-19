# ⚽️ Football Video AI Analyzer

简体中文｜[English](README_EN.md)

目前该项目为研究/学习级产品，用于跟踪足球比赛中的人和球，并识别出守门员、计算威胁程度，用于最终自动生成守门员处理球集锦（未实现）。

<div align="center">
<figure>
    <img src="docs/main.png" alt="Overview">
    <figcaption><em>总体效果</em></figcaption>
</figure>
</div>

## 功能

### I. 白线识别与拟合标准球场

`core/pitch_detection`

1. 使用 Segmentation Model 进行球场线识别及分类（后续计划不再依赖模型分类，可能尝试 Hough Transform）
![Pitch Detection](docs/pitch_det.png)

2. 根据预设的球场线信息，计算 Homography，拟合标准球场（顺便得出 Bird's Eye View）
![Birds' Eye View](docs/bev.png)

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

1. 通过前面得到的H，将所有足球放到BEV坐标系中
2. 计算所有检测点距离之前球轨迹的 Mahalanobis Distance
3. 使用 Chi Square 判断检测是否有效，选出最优的检测
4. 用 Kalman Filter 预测，用于平滑化轨迹以及防止阻挡带来的问题

---

## TODO

![project plan](docs/project_plan.jpg)

- [ ] 自动生成守门员处理球集锦
- [ ] 威胁度评分（结合位置、球速、对方球员密度等特征）
- [ ] 守门员姿态分析、运动速度轨迹分析
- [ ] 沉浸式看球（？）VR/gaming

---

## 使用的技术与依赖

- 检测模型：[football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbcYOLOv11m)
- 跟踪器：ByteTrack（配置文件 `bytetrack.yaml`）
- 可视化：OpenCV
- 语言：Python 3.10

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 如何启动？

> 模型文件 [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbcYOLOv11m)，自行训练或在 [蓝奏云](https://wwbcc.lanzoup.com/iDH023f1y7zg) 下载模型
> 测试比赛视频已内置于 `input_vids/` 中

1. 将模型文件放入 `models/`, 将比赛视频放入 `input_vids/` 
2. 在 main.py 中修改 `model` 和 `VIDEO_PATH` 为你的文件路径
3. 运行 main.py

4. 处理完成后：
   - 跟踪日志会保存到 `output/track_log.csv`
   - 可运行 `draw_2d.py` 查看可视化结果，会保存至 `output/track_trajectories.png`

---

## 🤝 贡献与联系

欢迎提交 issues 或 PR，亦可通过 hungryhenry101@outlook.com 进行联系