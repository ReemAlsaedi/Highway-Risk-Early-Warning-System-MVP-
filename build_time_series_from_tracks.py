"""

"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


CONFIG_PATH = "camera_config.json"
TRACKS_CSV = "data/interim/tracks_cam01.csv"
OUTPUT_TS_CSV = "data/processed/timeseries_cam01.csv"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def inside_roi(row, roi: dict) -> bool:
    
    return (
        roi["x_min"] <= row["x_center"] <= roi["x_max"] and
        roi["y_min"] <= row["y_center"] <= roi["y_max"]
    )


def compute_flow_events(df: pd.DataFrame, counting_line_y: float) -> pd.DataFrame:
   
    flow_events = []

    # ترتيب حسب المركبة والزمن
    df = df.sort_values(["track_id", "time_sec"])
    grouped = df.groupby("track_id")

    for track_id, g in grouped:
        g = g.reset_index(drop=True)
        crossed = False

        for i in range(1, len(g)):
            prev_y = g.loc[i - 1, "y_center"]
            curr_y = g.loc[i, "y_center"]
            prev_t = g.loc[i - 1, "time_sec"]
            curr_t = g.loc[i, "time_sec"]

            # 
            if (not crossed) and (prev_y > counting_line_y >= curr_y):
                
                flow_events.append({
                    "track_id": track_id,
                    "cross_time_sec": curr_t
                })
                crossed = True  # 

    return pd.DataFrame(flow_events)


def compute_instant_speeds(df: pd.DataFrame, meters_per_pixel_y: float) -> pd.DataFrame:
   
    speed_records = []

    df = df.sort_values(["track_id", "time_sec"])
    grouped = df.groupby("track_id")

    for track_id, g in grouped:
        g = g.reset_index(drop=True)

        for i in range(1, len(g)):
            prev_y = g.loc[i - 1, "y_center"]
            curr_y = g.loc[i, "y_center"]
            prev_t = g.loc[i - 1, "time_sec"]
            curr_t = g.loc[i, "time_sec"]

            dt = curr_t - prev_t
            if dt <= 0:
                continue

            dy_pixels = curr_y - prev_y
            dy_meters = abs(dy_pixels) * meters_per_pixel_y
            speed_mps = dy_meters / dt

            if 0 < speed_mps < 60:
                speed_records.append({
                    "track_id": track_id,
                    "time_sec": curr_t,
                    "speed_mps": speed_mps
                })

    return pd.DataFrame(speed_records)


def compute_frame_occupancy(df: pd.DataFrame, roi: dict) -> pd.DataFrame:

    roi_area = (roi["x_max"] - roi["x_min"]) * (roi["y_max"] - roi["y_min"])
    if roi_area <= 0:
        raise ValueError("ROI area must be positive")

    df = df.copy()
    df = df[
        (df["x_center"] >= roi["x_min"]) &
        (df["x_center"] <= roi["x_max"]) &
        (df["y_center"] >= roi["y_min"]) &
        (df["y_center"] <= roi["y_max"])
    ]

    df["bbox_area"] = df["width"] * df["height"]

    grouped = df.groupby("frame_idx")["bbox_area"].sum().reset_index()
    grouped["occupancy_ratio"] = grouped["bbox_area"] / roi_area

    time_lookup = df.groupby("frame_idx")["time_sec"].mean().reset_index()

    occ_df = pd.merge(grouped, time_lookup, on="frame_idx", how="left")

    return occ_df[["frame_idx", "time_sec", "occupancy_ratio"]]


def aggregate_to_time_windows(
    flow_events: pd.DataFrame,
    speeds: pd.DataFrame,
    occupancy_frames: pd.DataFrame,
    time_window_sec: int
) -> pd.DataFrame:
    """
    تجميع Flow / Speed / Occupancy إلى نوافذ زمنية (مثلاً كل 60 ثانية).
    """

    def assign_window(t):
        return int(t // time_window_sec) * time_window_sec

    # 1) flow per window
    if len(flow_events) > 0:
        flow_events["window_start_sec"] = flow_events["cross_time_sec"].apply(assign_window)
        flow_agg = (
            flow_events.groupby("window_start_sec")["track_id"]
            .nunique()
            .reset_index()
            .rename(columns={"track_id": "flow_count"})
        )
    else:
        flow_agg = pd.DataFrame(columns=["window_start_sec", "flow_count"])

    # 2) speed per window
    if len(speeds) > 0:
        speeds["window_start_sec"] = speeds["time_sec"].apply(assign_window)
        speed_agg = (
            speeds.groupby("window_start_sec")["speed_mps"]
            .mean()
            .reset_index()
            .rename(columns={"speed_mps": "speed_mean_mps"})
        )
    else:
        speed_agg = pd.DataFrame(columns=["window_start_sec", "speed_mean_mps"])

    # 3) occupancy per window
    if len(occupancy_frames) > 0:
        occupancy_frames["window_start_sec"] = occupancy_frames["time_sec"].apply(assign_window)
        occ_agg = (
            occupancy_frames.groupby("window_start_sec")["occupancy_ratio"]
            .mean()
            .reset_index()
            .rename(columns={"occupancy_ratio": "occupancy_mean"})
        )
    else:
        occ_agg = pd.DataFrame(columns=["window_start_sec", "occupancy_mean"])

    # دمج الكل
    ts = pd.merge(flow_agg, speed_agg, on="window_start_sec", how="outer")
    ts = pd.merge(ts, occ_agg, on="window_start_sec", how="outer")

    ts = ts.sort_values("window_start_sec").reset_index(drop=True)

    return ts


def main():
    config = load_config(CONFIG_PATH)
    roi = config["roi"]
    meters_per_pixel_y = config["meters_per_pixel_y"]
    counting_line_y = config["counting_line_y"]
    time_window_sec = config["time_window_sec"]

    df_tracks = pd.read_csv(TRACKS_CSV)

    df_tracks = df_tracks[df_tracks.apply(lambda r: inside_roi(r, roi), axis=1)].copy()

    #  Flow events 
    flow_events = compute_flow_events(df_tracks, counting_line_y)

    #  السرعات اللحظية
    speeds = compute_instant_speeds(df_tracks, meters_per_pixel_y)

    # Occupancy 
    occupancy_frames = compute_frame_occupancy(df_tracks, roi)

 
    ts = aggregate_to_time_windows(flow_events, speeds, occupancy_frames, time_window_sec)

    ts["speed_mean_kmh"] = ts["speed_mean_mps"] * 3.6

    # save
    Path(OUTPUT_TS_CSV).parent.mkdir(parents=True, exist_ok=True)
    ts.to_csv(OUTPUT_TS_CSV, index=False)
    print(f"Saved time-series to: {OUTPUT_TS_CSV}")


if __name__ == "__main__":
    main()
