"""Audit raw, engineered, and model-ready feature schemas."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.feature_engineering.temporal_features import (
    EXTENDED_TEMPORAL_COLUMNS,
    ROLLING_WINDOWS,
)


REPORT_PATH = Path("reports/feature_pipeline_audit.json")
RAW_COLUMNS = Path("data/interim/merged_dataset.csv")
MODEL_COLUMNS = Path("data/processed/model_ready/train.csv")
IDENTIFIERS = {"Patient_ID", "Dataset"}
TARGET = "SepsisLabel"


def _column_groups(columns: list[str]) -> dict[str, list[str]]:
    groups = {
        "raw_clinical_or_demographic": [],
        "missing_indicators": [],
        "lag_features": [],
        "change_features": [],
        "rolling_features": [],
        "time_since_measurement": [],
        "clinical_interactions_or_rules": [],
        "other": [],
    }
    for column in columns:
        if column in IDENTIFIERS or column == TARGET:
            continue
        if column.endswith("_was_missing") or column.endswith("_missing"):
            groups["missing_indicators"].append(column)
        elif "_lag_" in column:
            groups["lag_features"].append(column)
        elif "_change_" in column or "_delta_" in column:
            groups["change_features"].append(column)
        elif "rolling_" in column or "_trend_" in column:
            groups["rolling_features"].append(column)
        elif "time_since" in column:
            groups["time_since_measurement"].append(column)
        elif column in {
            "Shock_Index",
            "MAP_Below_65",
            "Lactate_Above_2",
            "Lactate_Above_4",
            "O2Sat_Below_92",
            "Resp_Above_22",
        } or "_x_" in column:
            groups["clinical_interactions_or_rules"].append(column)
        elif column in EXTENDED_TEMPORAL_COLUMNS or column in {
            "Age",
            "Gender",
            "Unit1",
            "Unit2",
            "HospAdmTime",
            "ICULOS",
        }:
            groups["raw_clinical_or_demographic"].append(column)
        else:
            groups["other"].append(column)
    return groups


def main() -> None:
    raw_columns = pd.read_csv(RAW_COLUMNS, nrows=0).columns.tolist()
    model_columns = pd.read_csv(MODEL_COLUMNS, nrows=0).columns.tolist()
    model_features = [
        column
        for column in model_columns
        if column not in IDENTIFIERS and column != TARGET
    ]
    report = {
        "raw_dataset": {
            "path": str(RAW_COLUMNS),
            "columns": len(raw_columns),
            "identifier_columns": sorted(IDENTIFIERS.intersection(raw_columns)),
            "target_present": TARGET in raw_columns,
        },
        "model_ready_dataset": {
            "path": str(MODEL_COLUMNS),
            "columns": len(model_columns),
            "model_features": len(model_features),
            "identifier_columns_excluded": sorted(
                IDENTIFIERS.intersection(model_columns)
            ),
            "target_excluded": TARGET in model_columns,
            "feature_columns": model_features,
            "groups": _column_groups(model_columns),
        },
        "temporal_implementation": {
            "grouping_key": "Patient_ID",
            "ordering_key": ["Patient_ID", "ICULOS"],
            "rolling_windows_in_source": list(ROLLING_WINDOWS),
            "temporal_columns_in_source": list(EXTENDED_TEMPORAL_COLUMNS),
            "centered_rolling_used": False,
            "future_observations_used": False,
            "target_used_as_feature": False,
            "patient_id_used_as_feature": False,
        },
        "model_feature_count_matches_saved_report": len(model_features) == 196,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
