"""
build_ts_features.py

الغرض:
- قراءة timeseries_cam01.csv (إشارات الطريق لكل نافذة زمنية)
- قراءة accidents_cam01.csv (أوقات الحوادث بالثواني)
- إضافة:
    * risk_label  (0/1) استباقي: هل سيحدث حادث خلال 5 دقائق بعد بداية النافذة؟
    * مجموعة من الـ features مثل:
        - flow_t, speed_t, occ_t
        - rolling mean / std / cv لآخر 3 نوافذ
        - deltas (فرق عن النافذة السابقة)

الناتج:
- ملف: data/processed/risk_windows_cam01.csv
  جاهز للتدريب.
"""

import pandas as pd
from pathlib import Path

#  إذا لزم)
TS_CSV = "data/processed/timeseries_cam01.csv"
ACC_CSV = "data/accidents_cam01.csv"
OUTPUT_CSV = "data/processed/risk_windows_cam01.csv"

# إعدادات الزمن
WINDOW_SIZE_SEC = 60      
LEAD_TIME_SEC = 300       # 5 الحادث (تقدرين تغيرينه)


# --------- 1)  risk_label  ---------

def label_risky_windows(df_ts: pd.DataFrame,
                        df_acc: pd.DataFrame,
                        lead_time_sec: int = LEAD_TIME_SEC) -> pd.DataFrame:
    """
    لكل نافذة زمنية window_start_sec = t_window
    نبحث عن حوادث accident_time_sec بحيث:
        t_window <= accident_time_sec <= t_window + lead_time_sec

    لو فيه حادث في هالفترة → risk_label = 1
    غير كذا → risk_label = 0
    """
    df_ts = df_ts.copy()
    df_ts = df_ts.sort_values("window_start_sec").reset_index(drop=True)

    accident_times = df_acc["accident_time_sec"].values

    def label_row(row):
        t0 = row["window_start_sec"]
        t1 = t0 + lead_time_sec
        is_risky = ((accident_times >= t0) & (accident_times <= t1)).any()
        return 1 if is_risky else 0

    df_ts["risk_label"] = df_ts.apply(label_row, axis=1)
    return df_ts


# --------- 2) Features ---------

def build_features(df_labeled: pd.DataFrame) -> pd.DataFrame:
    """
    يبني features شبيهة اللي شفتيها قبل (flow_mean_3, speed_mean_3, ...).
    """

    df = df_labeled.copy()
    df = df.sort_values("window_start_sec").reset_index(drop=True)


    df["flow_t"] = df["flow_count"]


    if "speed_mean_kmh" in df.columns:
        df["speed_t"] = df["speed_mean_kmh"]
    elif "speed_mean_mps" in df.columns:
        df["speed_t"] = df["speed_mean_mps"] * 3.6
    else:
        raise ValueError("لم أجد عمود السرعة (speed_mean_kmh أو speed_mean_mps) في timeseries_cam01.csv")

    df["occ_t"] = df["occupancy_mean"]

    window = 3

    def add_rolling_stats(base_col: str, prefix: str):
        """يبني mean, std, cv, delta لعمود معين"""
        # المتوسط
        df[f"{prefix}_mean_{window}"] = (
            df[base_col].rolling(window=window, min_periods=1).mean()
        )

        # الانحراف المعياري
        df[f"{prefix}_std_{window}"] = (
            df[base_col].rolling(window=window, min_periods=1).std()
        )

        # معامل التشتت 
        mean_col = df[f"{prefix}_mean_{window}"]
        std_col = df[f"{prefix}_std_{window}"]
        df[f"{prefix}_cv_{window}"] = std_col / (mean_col.replace(0, pd.NA))

        # delta 
        df[f"{prefix}_delta_1"] = df[base_col].diff(1)

    # flow, speed, occ
    add_rolling_stats("flow_t", "flow")
    add_rolling_stats("speed_t", "speed")
    add_rolling_stats("occ_t", "occ")

    # نعالج NaN في أول الصفوف
    df = df.fillna(0.0)

    return df


def main():
    # 1) قراءة البيانات
    df_ts = pd.read_csv(TS_CSV)
    df_acc = pd.read_csv(ACC_CSV)

    # 2) إضافة الـ risk_label الاستباقية
    df_labeled = label_risky_windows(df_ts, df_acc, LEAD_TIME_SEC)

    # 3) بناء features غنية
    df_feat = build_features(df_labeled)

    # 4) حفظ الناتج
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved ML dataset with features to: {OUTPUT_CSV}")
    print("أول 5 صفوف:")
    print(df_feat.head())


if __name__ == "__main__":
    main()
