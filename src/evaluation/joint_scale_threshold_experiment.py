"""Joint validation search over persisted class-weighted XGBoost models.

The repository already contains two compatible XGBoost models trained with
different ``scale_pos_weight`` values. This experiment jointly searches those
weights with the threshold and consecutive-observation rule. It loads only
train metadata and validation data; the test split is deliberately untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.final_temporal_patient_evaluation import (
    evaluate_prediction_frame,
)


DATA_DIR = Path("data/processed/model_ready")
REPORT_DIR = Path("reports")
RESULTS_PATH = REPORT_DIR / "joint_scale_threshold_results.csv"
REPORT_PATH = REPORT_DIR / "joint_scale_threshold_report.json"
MODEL_PATHS = {
    "xgboost_scale_factor_1.00": Path("models/xgboost.pkl"),
    "xgboost_scale_factor_0.50": Path("models/xgboost_optimized.pkl"),
}
THRESHOLDS = (0.40, 0.50, 0.60, 0.70)
CONSECUTIVE_OPTIONS = (1, 2, 3)
RECALL_CONSTRAINT = 0.80


def _prediction_frame(data: pd.DataFrame, model) -> pd.DataFrame:
    features = [
        column
        for column in data.columns
        if column not in {"Patient_ID", "Dataset", "SepsisLabel"}
    ]
    expected = getattr(model, "n_features_in_", len(features))
    if len(features) != expected:
        raise ValueError(f"Model expects {expected} features, found {len(features)}")
    result = data[["Patient_ID", "ICULOS", "SepsisLabel"]].copy()
    result["Sepsis_Probability"] = model.predict_proba(data[features])[:, 1]
    return result


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    validation = validation.sort_values(
        ["Patient_ID", "ICULOS"], kind="mergesort"
    ).reset_index(drop=True)
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=["SepsisLabel"])
    base_scale_weight = float(
        (train["SepsisLabel"] == 0).sum()
        / max((train["SepsisLabel"] == 1).sum(), 1)
    )

    rows: list[dict] = []
    model_metadata: dict[str, dict] = {}
    for model_name, model_path in MODEL_PATHS.items():
        model = joblib.load(model_path)
        params = model.get_params()
        scale_weight = float(params.get("scale_pos_weight", 1.0))
        model_metadata[model_name] = {
            "path": str(model_path),
            "scale_pos_weight": scale_weight,
            "scale_factor_from_training_ratio": scale_weight / base_scale_weight,
        }
        print(
            f"Evaluating {model_name}: scale_pos_weight={scale_weight:.6f}",
            flush=True,
        )
        prediction_frame = _prediction_frame(validation, model)
        for threshold in THRESHOLDS:
            for consecutive in CONSECUTIVE_OPTIONS:
                _, report = evaluate_prediction_frame(
                    prediction_frame,
                    threshold=threshold,
                    required_consecutive=consecutive,
                )
                metrics = report["metrics"]
                rows.append(
                    {
                        "model": model_name,
                        "model_path": str(model_path),
                        "scale_pos_weight": scale_weight,
                        "scale_factor": scale_weight / base_scale_weight,
                        "threshold": threshold,
                        "required_consecutive": consecutive,
                        "validation_accuracy": metrics["accuracy"],
                        "validation_precision": metrics["precision"],
                        "validation_recall": metrics["recall"],
                        "validation_f1": metrics["f1_score"],
                        "validation_specificity": metrics["specificity"],
                        "validation_roc_auc": metrics["roc_auc"],
                        "validation_pr_auc": metrics["pr_auc"],
                        "false_alerts_per_icu_stay": metrics[
                            "false_alerts_per_icu_stay"
                        ],
                        "true_positives": metrics["true_positives"],
                        "false_positives": metrics["false_positives"],
                        "false_negatives": metrics["false_negatives"],
                        "true_negatives": metrics["true_negatives"],
                    }
                )
        pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
        del model, prediction_frame

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False)
    eligible = results[results["validation_recall"] >= RECALL_CONSTRAINT]
    constrained_pool = eligible if not eligible.empty else results
    constrained_best = constrained_pool.sort_values(
        [
            "validation_f1",
            "validation_recall",
            "validation_precision",
            "validation_pr_auc",
        ],
        ascending=False,
    ).iloc[0].to_dict()
    unconstrained_best = results.sort_values(
        ["validation_f1", "validation_recall", "validation_precision"],
        ascending=False,
    ).iloc[0].to_dict()
    report = {
        "selection_data": "validation_only",
        "test_set_loaded": False,
        "base_scale_pos_weight": base_scale_weight,
        "thresholds_tested": list(THRESHOLDS),
        "consecutive_options_tested": list(CONSECUTIVE_OPTIONS),
        "recall_constraint": RECALL_CONSTRAINT,
        "configurations_tested": int(len(results)),
        "eligible_configurations": int(len(eligible)),
        "models_tested": model_metadata,
        "best_with_recall_constraint": constrained_best,
        "best_unconstrained_f1": unconstrained_best,
        "results_path": str(RESULTS_PATH),
    }
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
