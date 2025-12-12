"""
Script 1: استخراج مسارات المركبات من الفيديو باستخدام YOLO + DeepSORT
النتيجة: ملف CSV فيه كل track عبر الزمن (إحداثيات البوكس ووقت كل إطار)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



VIDEO_PATH = "data/raw/highway_cam01.mp4"   
OUTPUT_TRACKS_CSV = "C:/Users/reema/Desktop/data/interim/tracks_cam01.csv"
CAMERA_ID = "cam_01"


VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck 


def main():
    # 1)  YOLO

    model = YOLO("yolov8n.pt")

    # 2)  DeepSORT
    tracker = DeepSort(
        max_age=30,             
        n_init=3,                
        nms_max_overlap=1.0,
        max_cosine_distance=0.3
    )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")

    records = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        time_sec = frame_idx / fps

        # 3)  YOLO 
        results = model(frame, verbose=False)[0]

        #  detections  DeepSORT
        detections_for_tracker = []
        boxes_xyxy = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        classes = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

        for box, cls_id, conf in zip(boxes_xyxy, classes, confs):
            if int(cls_id) not in VEHICLE_CLASS_IDS:
                continue

            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            detections_for_tracker.append(
                ([x1, y1, w, h], float(conf), int(cls_id))
            )

        # 4)  tracker
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cx = (l + r) / 2.0
            cy = (t + b) / 2.0
            w = r - l
            h = b - t

            records.append({
                "camera_id": CAMERA_ID,
                "video_path": VIDEO_PATH,
                "frame_idx": frame_idx,
                "time_sec": time_sec,
                "track_id": track_id,
                "x_center": cx,
                "y_center": cy,
                "width": w,
                "height": h
            })

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()

    df_tracks = pd.DataFrame(records)
    Path(OUTPUT_TRACKS_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_tracks.to_csv(OUTPUT_TRACKS_CSV, index=False)
    print(f"Saved tracks to: {OUTPUT_TRACKS_CSV}")


if __name__ == "__main__":
    main()
