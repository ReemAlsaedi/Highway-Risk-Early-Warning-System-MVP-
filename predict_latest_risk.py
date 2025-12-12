"""
predict_latest_risk.py

الآن بدل ما يختار آخر نافذة زمنية،
نختار نافذة معيّنة حسب الوقت بالثواني (مثلاً t = 180 sec).
"""

import pandas as pd
import joblib
from pathlib import Path

DATA_CSV = "data/processed/risk_windows_cam01.csv"
MODEL_PATH = "models/rf_risk_model_cam01.pkl"

# 👈 هنا نحدد أي نافذة نبغى نحللها
TARGET_WINDOW_SEC = 180   # 3 


def classify_risk(prob: float) -> str:
    if prob >= 0.8:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    else:
        return "Low"


def main():
    # 1) قراءة الداتا
    if not Path(DATA_CSV).exists():
        raise FileNotFoundError(f"لم أجد ملف: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    df = df.sort_values("window_start_sec").reset_index(drop=True)

    # 2) تحميل المودل
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"لم أجد ملف المودل: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    # 3) تجهيز الـ features 
    exclude_cols = ["risk_label"]
    if "window_start_sec" in df.columns:
        exclude_cols.append("window_start_sec")

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    print("عدد الـ features التي يتوقعها المودل:", len(feature_cols))
    print("بعضها:", feature_cols[:8])

    # 4) نختار النافذة المطلوبة حسب TARGET_WINDOW_SEC
    #    
    if "window_start_sec" not in df.columns:
        raise ValueError("عمود window_start_sec غير موجود في الملف!")

    # نحسب الفرق المطلق بين كل نافذة و TARGET_WINDOW_SEC
    idx = (df["window_start_sec"] - TARGET_WINDOW_SEC).abs().idxmin()
    row = df.loc[[idx]]  # نخليها DataFrame (صف واحد)

    actual_t = row["window_start_sec"].iloc[0]
    mm = int(actual_t // 60)
    ss = int(actual_t % 60)

    X = row[feature_cols]

    # 5) التنبؤ
    prob = float(model.predict_proba(X)[:, 1][0])
    risk_class = classify_risk(prob)

    # 
    flow_t = row.get("flow_t", row.get("flow_count")).iloc[0]
    if "speed_t" in row.columns:
        speed_t = row["speed_t"].iloc[0]
    elif "speed_mean_kmh" in row.columns:
        speed_t = row["speed_mean_kmh"].iloc[0]
    else:
        speed_t = float("nan")

    if "occ_t" in row.columns:
        occ_t = row["occ_t"].iloc[0]
    else:
        occ_t = row.get("occupancy_mean", pd.Series([float("nan")])).iloc[0]

    print("\n=== Risk Assessment at specific window ===")
    print(f"Actual window_start = {actual_t} sec (~ {mm:02d}:{ss:02d})")
    print(f"flow_t (veh/min)    = {flow_t:.2f}")
    print(f"speed_t (km/h)      = {speed_t:.2f}")
    print(f"occ_t (occupancy)   = {occ_t:.3f}")
    print(f"risk_prob           = {prob:.3f}")
    print(f"risk_class          = {risk_class}")


if __name__ == "__main__":
    main()
