# Highway-Risk-Early-Warning-System-MVP-
ر# Highway Risk Early Warning System (MVP)

This project is an MVP for a **video-based early warning system** for highway traffic risk.

It takes **road camera video**, extracts **vehicle trajectories**, builds **time-series traffic indicators**, and trains a **Random Forest model** to estimate the probability of an accident occurring within the next 5 minutes.

The goal is to support **Road Security / Traffic Patrols** with **proactive alerts** instead of only reacting *after* the accident.

---

## 1. Project Pipeline

1. **Video → Vehicle Tracks (Detection + Tracking)**  
   - Script: `extract_tracks_from_video.py`  
   - Uses YOLOv8 (Ultralytics) + DeepSORT Realtime  
   - Output: `data/interim/tracks_cam01.csv`  
   - Each row = one vehicle (track) at one frame:  
     - `track_id, frame_idx, time_sec, x_center, y_center, width, height, class_id, confidence`

2. **Tracks → Time-Series per Minute**  
   - Script: `build_time_series_from_tracks.py`  
   - Config: `camera_config.json`  
   - Output: `data/processed/timeseries_cam01.csv`  
   - For each time window (e.g. 60 seconds):  
     - `window_start_sec` – start time of the window (sec)  
     - `flow_count` (veh/min) – vehicles crossing a counting line  
     - `speed_mean_mps` / `speed_mean_kmh` – average speed  
     - `occupancy_mean` – ratio of road occupancy in ROI

3. **Add Accident Labels + Features (Proactive)**  
   - Manual labels: `data/accidents_cam01.csv`  
     - Format:
       ```text
       accident_id,accident_time_sec,description,severity
       1,1800,"rear-end collision","medium"
       2,2700,"lane-change conflict","low"
       ```  
   - Script: `build_ts_features.py`  
   - Output: `data/processed/risk_windows_cam01.csv`  
   - Adds:
     - `risk_label` = 1 if an accident occurs within the next 5 minutes  
       (configurable via `LEAD_TIME_SEC`)  
     - Rolling features (3-window): mean, std, CV, deltas for:
       - `flow_t`, `speed_t`, `occ_t` (flow, speed, occupancy)

4. **Train Random Forest Model**  
   - Script: `train_rf_model_from_history.py`  
   - Input: `data/processed/risk_windows_cam01.csv`  
   - Output: `models/rf_risk_model_cam01.pkl`  
   - Task: binary classification:
     - **1 = risky window** (accident expected within next 5 minutes)  
     - **0 = non-risky window**

5. **Risk Prediction for a Given Time Window**  
   - Script: `predict_latest_risk.py`  
   - Reads the features for one time window  
   - Loads the trained model and outputs:
     - `flow_t`, `speed_t`, `occ_t`  
     - `risk_prob` (probability of an accident in the next 5 minutes)  
     - `risk_class` = Low / Medium / High  

---

## 2. Files Overview

- `extract_tracks_from_video.py`  
  Video → YOLOv8 + DeepSORT → `tracks_cam01.csv`.  
  Detects vehicles per frame and assigns a persistent `track_id` to each vehicle.

- `build_time_series_from_tracks.py`  
  Tracks → per-minute time-series indicators (`flow`, `speed`, `occupancy`).  
  Uses `camera_config.json` to define ROI, counting line, and meters-per-pixel.

- `camera_config.json`  
  Camera calibration and ROI configuration:
  - `meters_per_pixel_y` – convert pixel movement to meters  
  - `roi` – region of interest of the road  
  - `counting_line_y` – line used to count vehicles  
  - `time_window_sec` – time window size (e.g. 60 seconds)

- `accidents_cam01.csv`  
  Historical accident labels:
  - `accident_time_sec` = time (sec) from start of video where an accident occurred  
  - Used only for **training**, not for real-time prediction.

- `build_ts_features.py`  
  - Combines `timeseries_cam01.csv` + `accidents_cam01.csv`.  
  - Adds `risk_label` (proactive: “accident within next 5 minutes?”).  
  - Adds rolling statistics + deltas for flow, speed, and occupancy.  
  - Output: `risk_windows_cam01.csv` – ready for ML training.

- `train_rf_model_from_history.py`  
  - Trains a Random Forest classifier on `risk_windows_cam01.csv`.  
  - Saves the model to `models/rf_risk_model_cam01.pkl`.

- `predict_latest_risk.py`  
  - Loads `rf_risk_model_cam01.pkl`.  
  - Reads one time window (latest or a specific `TARGET_WINDOW_SEC`).  
  - Outputs `risk_prob` and `risk_class` for that window.

- `inspect_sharp_changes.py` (optional tool for analysis)  
  - Helps explore windows with sharp changes in speed/flow.  
  - Useful for understanding unusual patterns (e.g. sudden drops in speed).

---

## 3. Data Folders

- `data/raw/`  
  - **Not pushed to GitHub**  
  - Contains the original camera videos (large/private), e.g.:
    - `data/raw/highway_cam01.mp4`

- `data/interim/`  
  - Intermediate outputs:
    - `tracks_cam01.csv` from detection + tracking.

- `data/processed/`  
  - Processed and ML-ready data:
    - `timeseries_cam01.csv` – per-minute traffic indicators.  
    - `risk_windows_cam01.csv` – enriched with labels and features.

- `models/`  
  - Trained models:
    - `rf_risk_model_cam01.pkl`

---

## 4. How to Run (Locally)

### 4.1 Create and activate a conda environment

```bash
conda create -n warning_env python=3.10
conda activate warning_env
