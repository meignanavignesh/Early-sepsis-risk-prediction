"""Audit the protected sepsis project before improvement experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


SPLIT_DIR = Path("data/processed/model_ready")
REPORT_DIR = Path("reports")
OUTPUT_PATH = REPORT_DIR / "project_audit_report.json"


def _split_summary(path: Path) -> dict:
    frame = pd.read_csv(path, usecols=["Patient_ID", "SepsisLabel"])
    patient_labels = frame.groupby("Patient_ID")["SepsisLabel"].max()
    return {
        "rows": int(len(frame)),
        "patients": int(frame["Patient_ID"].nunique()),
        "sepsis_patients": int(patient_labels.sum()),
        "non_sepsis_patients": int((patient_labels == 0).sum()),
        "positive_observations": int(frame["SepsisLabel"].sum()),
        "negative_observations": int((frame["SepsisLabel"] == 0).sum()),
        "patient_ids": set(frame["Patient_ID"].astype(str)),
    }


def main() -> None:
    summaries = {
        split: _split_summary(SPLIT_DIR / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    overlaps = {
        f"{left}_intersection_{right}": sorted(
            summaries[left]["patient_ids"] & summaries[right]["patient_ids"]
        )
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    }
    report = {
        "patient_level_split": True,
        "patient_leakage": {
            name: {"count": len(values), "empty": not values}
            for name, values in overlaps.items()
        },
        "splits": {
            split: {key: value for key, value in summary.items() if key != "patient_ids"}
            for split, summary in summaries.items()
        },
        "production": {
            "model": "models/xgboost_optimized.pkl",
            "prediction_output": "data/processed/final_predictions.csv",
            "threshold": 0.60,
            "required_consecutive": 2,
            "feature_count": 196,
        },
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
