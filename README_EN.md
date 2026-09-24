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

## Pipeline

### I. Pitch Registration (Camera Calibration)

`core/broadtrack_calib` (C++ core in `core/BroadTrack`)

Uses the architecture of [BroadTrack](https://arxiv.org/abs/2412.01721):
- Detects the pitch with Line Segmentation Model mainly and Keypoint Model in assistance;
- Optimises the camera calibration with optical flow, pan grid-search + parabolic refinement, camera anchor;
- Evaluate the result by IoU score.

The tracking loop is compiled to a Python extension with pybind11, so the Ceres
bundle adjustment, the line-IoU score and the Lucas-Kanade association stay in
C++ while the two detectors run on the Python `torch` already in the process.
`BroadTrackCalib` is a drop-in for the former `PnLCalib`: same result keys, same
coordinate frame, same drawing helpers. See
[`core/BroadTrack/README.md`](core/BroadTrack/README.md).
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

## First Launch

> The test video clips are under `input_vids/`.

1. Prepare detection model for ball & player：Download dataset from [roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) 
and train it (YOLO11 is used in demonstration). Put the trained weights file to the `models` folder, and edit the path of weight file in `main.py`
1. Download BroadTrack's two TorchScript detectors — `nbjw_keypoint_model.pt` and `tvcalib_model.pt` from 
[github](https://github.com/evs-broadcast/BroadTrack/tree/main/models). Place them in `models/`
1. Create the environment and build the calibration extension:
   ```bash
   conda env create -f environment.yml && conda activate football
   python scripts/build_minimal_opencv.py   # once, ~20 min
   python scripts/build_minimal_ceres.py    # once, ~5 min
   python scripts/build_native.py
   ```
   The two `build_minimal_*` steps compile OpenCV and Ceres locally rather than
   using the packaged ones, because every packaged build of both pulls a second
   OpenMP runtime into the process and PyTorch's wheel already ships one — the
   interpreter then aborts on import. See
   [`core/BroadTrack/README.md`](core/BroadTrack/README.md). On Linux
   you also need RapidJSON and Boost headers, plus `metis` for Ceres.

1. Update the weight paths and `VIDEO_PATH` in `main.py` with your file paths.
1. Run `main.py`: View of camera and BEV will show up
---

## TODO

Detailed problem list here:  [problems.md](docs/problems.md)

- [ ] Model downloading script
- [ ] Automatically generate goalkeeper highlight reels
- [ ] Threat level rating (combining location, ball speed, opponent density, etc.)
- [ ] Goalkeeper pose analysis and velocity trajectory analysis
- [ ] Immersive viewing (?) VR/gaming

---

## References & Acknowledgments

### Pitch Registration
- [SoccerNet Calibration](https://github.com/SoccerNet/sn-calibration)
- [BroadTrack](https://github.com/evs-broadcast/BroadTrack): Adapted code from this project
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

### Data Processing & Analytics
- [SoccermaticsForPython](https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython)
- [Friends of Tracking](https://www.youtube.com/@friendsoftracking755): helpful learning resources
- [wyscout](https://apidocs.wyscout.com?version=3)

---

## 🤝 Contribution & Contact

Contributions via issues or PRs are welcome! You can also reach out via hungryhenry101@outlook.com.
