"""
train_rf_from_risk_windows.py

الغرض:
- قراءة data/processed/risk_windows_cam01.csv
- اختيار الأعمدة المميزة (features)
- تدريب RandomForestClassifier على risk_label
- طباعة تقرير بسيط + أهم الـ features
"""

import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_CSV = "data/processed/risk_windows_cam01.csv"
MODEL_PATH = "models/rf_risk_model_cam01.pkl"


def main():
    df = pd.read_csv(DATA_CSV)

    # نتأكد يوجد risk_label
    if "risk_label" not in df.columns:
        raise ValueError("عمود risk_label غير موجود في dataset. تأكدي من build_ts_features.py")

    # نحدد العمود الهدف
    y = df["risk_label"]

    # نختار الـ features: كل الأعمدة الرقمية ما عدا window_start_sec و risk_label
    exclude_cols = ["risk_label"]
    if "window_start_sec" in df.columns:
        exclude_cols.append("window_start_sec")

    feature_cols = [
        c for c in df.columns
        if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols]

    print("عدد العيّنات:", len(df))
    print("عدد الـ features:", len(feature_cols))
    print("أسماء بعض الـ features:", feature_cols[:10])

    # تقسيم Train / Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # نموذج Random Forest
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"  # مهم لو البيانات غير متوا
    )

    clf.fit(X_train, y_train)

    # تقييم بسيط
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    # أهم الـ features
    importances = clf.feature_importances_
    feat_imp = sorted(
        zip(feature_cols, importances),
        key=lambda t: t[1],
        reverse=True
    )

    print("\n=== Top 15 Feature Importances ===")
    for name, imp in feat_imp[:15]:
        print(f"{name:25s}  {imp:.4f}")

    # حفظ المودل
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"\nSaved Random Forest model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
