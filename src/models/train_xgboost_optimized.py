"""Tune XGBoost for patient-level sepsis performance.

This keeps the existing xgboost.pkl unchanged.  Candidate models are selected
on validation data using patient-level F1 while requiring a minimum recall.
The test set is used only once, after model and threshold selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.evaluation.hourly_rules import (
    AlertRule,
    aggregate_patient_alerts,
    apply_alert_rule,
    evaluate_patient_predictions,
)
from src.utils.experiment_tracker import ExperimentTracker


INPUT_DIR = Path("data/processed/model_ready")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
TARGET = "SepsisLabel"
IDENTIFIERS = {"Patient_ID", "Dataset", TARGET}
MIN_RECALL = 0.80


def best_patient_threshold(frame: pd.DataFrame) -> dict:
    """Find the best patient threshold while preserving target recall."""

    rows = []
    for threshold in [x / 100 for x in range(10, 91, 5)]:
        hourly = frame.copy()
        hourly["Hourly_Alert"] = (
            hourly["Sepsis_Probability"] >= threshold
        ).astype(int)
        patient = aggregate_patient_alerts(hourly)
        metrics = evaluate_patient_predictions(patient)
        metrics["threshold"] = threshold
        metrics["f1"] = metrics["f1_score"]
        rows.append(metrics)

    eligible = [row for row in rows if row["recall"] >= MIN_RECALL]
    pool = eligible or rows
    return max(pool, key=lambda row: (row["f1"], row["precision"]))


def row_metrics(y_true: pd.Series, probabilities: pd.Series) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(INPUT_DIR / "train.csv")
    validation = pd.read_csv(INPUT_DIR / "validation.csv")
    feature_columns = [column for column in train.columns if column not in IDENTIFIERS]

    X_train, y_train = train[feature_columns], train[TARGET]
    X_validation, y_validation = validation[feature_columns], validation[TARGET]
    class_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    candidates = {
        "baseline": {
            "scale_pos_weight": class_weight,
            "max_depth": 6,
            "min_child_weight": 2,
            "gamma": 0,
            "reg_lambda": 1,
        },
        "precision_balanced": {
            "scale_pos_weight": class_weight * 0.50,
            "max_depth": 6,
            "min_child_weight": 4,
            "gamma": 0.05,
            "reg_lambda": 3,
        },
        "regularized": {
            "scale_pos_weight": class_weight * 0.75,
            "max_depth": 4,
            "min_child_weight": 6,
            "gamma": 0.10,
            "reg_lambda": 5,
        },
        "recall_balanced": {
            "scale_pos_weight": class_weight * 1.25,
            "max_depth": 5,
            "min_child_weight": 3,
            "gamma": 0,
            "reg_lambda": 2,
        },
    }

    results = []
    fitted = {}
    for name, params in candidates.items():
        print(f"Training candidate: {name}")
        model = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            **params,
        )
        model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)], verbose=False)
        probabilities = pd.Series(model.predict_proba(X_validation)[:, 1])
        hourly = validation[["Patient_ID", "ICULOS", TARGET]].copy()
        hourly["Sepsis_Probability"] = probabilities.to_numpy()
        patient_best = best_patient_threshold(hourly)
        row = row_metrics(y_validation, probabilities)
        row.update({"candidate": name, "patient_best": patient_best})
        results.append(row)
        fitted[name] = model
        print(
            f"  row f1={row['f1']:.4f}, row recall={row['recall']:.4f}, "
            f"patient f1={patient_best['f1']:.4f}, "
            f"patient recall={patient_best['recall']:.4f}"
        )

    selected = max(
        results,
        key=lambda row: (
            row["patient_best"]["f1"],
            row["patient_best"]["recall"],
            row["pr_auc"],
        ),
    )
    selected_name = selected["candidate"]
    selected_model = fitted[selected_name]
    model_path = MODEL_DIR / "xgboost_optimized.pkl"
    joblib.dump(selected_model, model_path)

    report = {
        "selection_metric": "best patient-level F1 with recall constraint",
        "minimum_recall": MIN_RECALL,
        "selected_candidate": selected_name,
        "selected_model_path": str(model_path),
        "feature_count": len(feature_columns),
        "class_weight_baseline": class_weight,
        "candidates": results,
    }
    report_path = REPORT_DIR / "xgboost_optimized_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "candidate": row["candidate"],
                "row_accuracy": row["accuracy"],
                "row_precision": row["precision"],
                "row_recall": row["recall"],
                "row_f1": row["f1"],
                "row_pr_auc": row["pr_auc"],
                "patient_threshold": row["patient_best"]["threshold"],
                "patient_precision": row["patient_best"]["precision"],
                "patient_recall": row["patient_best"]["recall"],
                "patient_f1": row["patient_best"]["f1"],
            }
            for row in results
        ]
    ).to_csv(REPORT_DIR / "xgboost_optimized_candidates.csv", index=False)

    ExperimentTracker().log_run(
        model_name="xgboost_optimized",
        dataset_paths={"train": str(INPUT_DIR / "train.csv"), "validation": str(INPUT_DIR / "validation.csv")},
        feature_count=len(feature_columns),
        parameters={"candidate_count": len(candidates), "minimum_recall": MIN_RECALL},
        metrics={"selected_candidate": selected_name, "patient_f1": selected["patient_best"]["f1"], "patient_recall": selected["patient_best"]["recall"]},
        artifacts={"model": str(model_path), "report": str(report_path)},
    )

    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
