# ⚽️ Football Video AI Analyzer

[简体中文](README.md) ｜ English

The project is currently under Research & Development. 
Pitch registration (camera calibration), detection and tracking of balls and players are implemented but require optimising.
The ultimate goal is to automatically generate highlights and tactical analysis by quantification and evaluation of certain metrics, such as threat models

<div align="center">
<figure>
    <img src="docs/main.png" alt="Overview">
    <figcaption><em>Overall Visual Output</em></figcaption>
</figure>
</div>

## Features

### I. Pitch Registration (Camera Calibration)

`core/pnl`

Using the architecture of [PnLCalib](https://arxiv.org/abs/2404.08401):
1. Detect the keypoints with a KP model, then use a line detection model to assist with pitch fitting
2. Calibrate the camera and optimise the result with FramebyFrameCalib
![Pitch Detection](docs/pnl.png)

### II. Player Detection and Tracking

`core/player_tracker.py`

![Player Tracker in BEV](docs/player_track_bev.png)

1. Uses an Object Detection Model to classify players/referees.

2. Tracks objects using trackers (e.g. ByteTrack), automatically handling CMC (Camera Motion Compensation), ReID, and Kalman Filtering.

3. Plans to identify goalkeeper candidates based on spatial features (currently only using x, y coordinates).

### III. Ball Detection and Tracking

`core/ball_tracker.py`

<div align="center">
<figure>
    <img src="docs/ball_tracker.png" alt="ball_tracker">
    <figcaption><em>Red denotes raw model detections, Blue denotes filtered trajectory.</em></figcaption>
</figure>
</div>

1. Maps all ball detections to the Bird's Eye View (BEV) coordinate system using the homography matrix H.
2. Calculates the Mahalanobis Distance from all detected points to the previous ball trajectory.
3. Uses Chi-Square testing to validate detections and select the best candidate.
4. Smooths the trajectory and handles occlusions using a Kalman Filter.

---

## Quick Start

> The test video clips are under `input_vids/`.

1. Prepare detection model for ball & player：Download dataset from [roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) 
and train it (YOLO11 is used in demonstration). Put the trained weights file to `weights` folder, and edit the path of weight file in `main.py`
1. Download keypoint model and line detection model from [PnLCalib](https://github.com/mguti97/PnLCalib/releases) (tested with SV_kp and SV_lines), and place them in `weights/`
1. Install dependencies (only tested with python 3.10):
   ```bash
   pip install -r requirements.txt
   ```

1. Update the weight paths and `VIDEO_PATH` in `main.py` with your file paths.
1. Run `main.py`: View of camera and BEV will show up
---

## TODO

Detailed problem list here:  [problems.md](docs/problems.md)

- [ ] Automatically generate goalkeeper highlight reels
- [ ] Threat level rating (combining location, ball speed, opponent density, etc.)
- [ ] Goalkeeper pose analysis and velocity trajectory analysis
- [ ] Immersive viewing (?) VR/gaming

---

## References

### Pitch Detection
- SoccerNet Calibration: https://github.com/SoccerNet/sn-calibration
- PnLCalib: https://arxiv.org/abs/2404.08401

### Data Processing & Analytics
- SoccermaticsForPython: https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython
- Friends of Tracking: https://www.youtube.com/@friendsoftracking755
- wyscout: https://apidocs.wyscout.com?version=3

---

## 🤝 Contribution & Contact

Contributions via issues or PRs are welcome! You can also reach out via hungryhenry101@outlook.com.
