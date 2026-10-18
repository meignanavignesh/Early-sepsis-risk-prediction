"""Validation-only operating-point experiment for the protected model.

This experiment does not retrain the production model and never uses the test
set to select a threshold or alert rule. It compares clinically meaningful
hourly decision rules against the existing XGBoost probabilities.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATA_DIR = Path("data/processed/model_ready")
MODEL_PATH = Path("models/xgboost_optimized.pkl")
REPORT_DIR = Path("reports")
COMPARISON_PATH = REPORT_DIR / "validation_alert_rule_comparison.csv"
REPORT_PATH = REPORT_DIR / "controlled_improvement_report.json"
THRESHOLDS = [round(value, 2) for value in np.arange(0.20, 0.91, 0.05)]


def _persistent_alerts(values: pd.Series, threshold: float, consecutive: int) -> pd.Series:
    result = np.zeros(len(values), dtype=np.int8)
    streak = 0
    fired = False
    for position, value in enumerate(values.astype(float)):
        streak = streak + 1 if value >= threshold else 0
        if streak >= consecutive:
            fired = True
        result[position] = int(fired)
    return pd.Series(result, index=values.index)


def _evaluate(hourly: pd.DataFrame, alert: pd.Series, rule: str, threshold: float) -> dict:
    work = hourly[["Patient_ID", "ICULOS", "SepsisLabel", "probability"]].copy()
    work["Hourly_Alert"] = alert.astype(int).to_numpy()
    work = work.sort_values(["Patient_ID", "ICULOS"], kind="mergesort")
    patient = work.groupby("Patient_ID", sort=False).agg(
        actual=("SepsisLabel", "max"),
        predicted=("Hourly_Alert", "max"),
        max_probability=("probability", "max"),
        alert_hours=("Hourly_Alert", "sum"),
    )
    y_true = patient["actual"].astype(int)
    y_pred = patient["predicted"].astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / max(tn + fp, 1)
    metrics = {
        "rule": rule,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "roc_auc": float(roc_auc_score(y_true, patient["max_probability"]))
        if y_true.nunique() > 1
        else None,
        "pr_auc": float(average_precision_score(y_true, patient["max_probability"]))
        if y_true.nunique() > 1
        else None,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "alerts_per_patient": float(patient["alert_hours"].mean()),
        "false_alerts_per_icu_stay": float(
            patient.loc[y_true == 0, "alert_hours"].sum()
            / max(int((y_true == 0).sum()), 1)
        ),
    }
    return metrics


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODEL_PATH)
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    feature_columns = [
        column
        for column in validation.columns
        if column not in {"Patient_ID", "Dataset", "SepsisLabel"}
    ]
    validation = validation.sort_values(
        ["Patient_ID", "ICULOS"], kind="mergesort"
    ).reset_index(drop=True)
    validation["probability"] = model.predict_proba(
        validation[feature_columns]
    )[:, 1]

    results: list[dict] = []
    for threshold in THRESHOLDS:
        hourly_hit = validation["probability"] >= threshold
        results.append(
            _evaluate(validation, hourly_hit, "single_threshold", threshold)
        )
        for consecutive in (2, 3):
            alert = validation.groupby("Patient_ID", sort=False)["probability"].apply(
                lambda values: _persistent_alerts(values, threshold, consecutive)
            ).reset_index(level=0, drop=True).reindex(validation.index)
            results.append(
                _evaluate(
                    validation,
                    alert,
                    f"{consecutive}_consecutive",
                    threshold,
                )
            )
        for window in (3, 6):
            rolling = validation.groupby("Patient_ID", sort=False)["probability"].transform(
                lambda values, window=window: values.rolling(window, min_periods=1).mean()
            )
            results.append(
                _evaluate(
                    validation,
                    rolling >= threshold,
                    f"rolling_mean_{window}h",
                    threshold,
                )
            )

    comparison = pd.DataFrame(results)
    comparison.to_csv(COMPARISON_PATH, index=False)
    eligible = comparison[comparison["recall"] >= 0.80]
    selection_pool = eligible if not eligible.empty else comparison
    selected = selection_pool.sort_values(
        ["f1", "recall", "precision", "pr_auc"], ascending=False
    ).iloc[0].to_dict()
    report = {
        "selection_data": "validation_only",
        "test_used_for_selection": False,
        "model_path": str(MODEL_PATH),
        "feature_count": len(feature_columns),
        "minimum_recall_constraint": 0.80,
        "eligible_configurations": int(len(eligible)),
        "selected_validation_configuration": selected,
        "comparison_path": str(COMPARISON_PATH),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
