"""Combine completed validation measurements into the joint-search report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPORT_DIR = Path("reports")
RESULTS_PATH = REPORT_DIR / "joint_scale_threshold_results.csv"
REPORT_PATH = REPORT_DIR / "joint_scale_threshold_report.json"
BASE_RATIO = 54.543762781186096


def main() -> None:
    # The scale=1.00 rows were measured by the joint experiment. The scale=0.50
    # rows come from the earlier validation-only rule sweep using the persisted
    # optimized production model. Both use the same patient-level evaluator.
    scale_one = pd.read_csv(RESULTS_PATH)
    scale_half = pd.read_csv(REPORT_DIR / "validation_alert_rule_comparison.csv")
    scale_half = scale_half[
        scale_half["rule"].isin(
            ["single_threshold", "2_consecutive", "3_consecutive"]
        )
        & scale_half["threshold"].astype(float).isin([0.40, 0.50, 0.60, 0.70])
    ].copy()
    scale_half = scale_half.rename(
        columns={
            "rule": "rule_name",
            "precision": "validation_precision",
            "recall": "validation_recall",
            "f1": "validation_f1",
            "accuracy": "validation_accuracy",
            "specificity": "validation_specificity",
            "roc_auc": "validation_roc_auc",
            "pr_auc": "validation_pr_auc",
        }
    )
    scale_half["model"] = "xgboost_scale_factor_0.50"
    scale_half["model_path"] = "models/xgboost_optimized.pkl"
    scale_half["scale_pos_weight"] = BASE_RATIO * 0.50
    scale_half["scale_factor"] = 0.50
    scale_half["required_consecutive"] = scale_half["rule_name"].map(
        {"single_threshold": 1, "2_consecutive": 2, "3_consecutive": 3}
    )
    scale_half["false_alerts_per_icu_stay"] = scale_half[
        "false_alerts_per_icu_stay"
    ]
    scale_half["true_positives"] = scale_half["true_positives"]
    scale_half["false_positives"] = scale_half["false_positives"]
    scale_half["false_negatives"] = scale_half["false_negatives"]
    scale_half["true_negatives"] = scale_half["true_negatives"]
    scale_half = scale_half.drop(columns=["rule_name"])
    combined = pd.concat([scale_one, scale_half], ignore_index=True, sort=False)
    combined.to_csv(RESULTS_PATH, index=False)

    eligible = combined[combined["validation_recall"].astype(float) >= 0.80]
    constrained_pool = eligible if not eligible.empty else combined
    constrained = constrained_pool.sort_values(
        ["validation_f1", "validation_recall", "validation_precision"],
        ascending=False,
    ).iloc[0].to_dict()
    unconstrained = combined.sort_values(
        ["validation_f1", "validation_recall", "validation_precision"],
        ascending=False,
    ).iloc[0].to_dict()
    report = {
        "selection_data": "validation_only",
        "test_set_loaded": False,
        "models_compared": {
            "xgboost_scale_factor_1.00": "models/xgboost.pkl",
            "xgboost_scale_factor_0.50": "models/xgboost_optimized.pkl",
        },
        "base_scale_pos_weight": BASE_RATIO,
        "recall_constraint": 0.80,
        "configurations_compared": int(len(combined)),
        "eligible_configurations": int(len(eligible)),
        "best_with_recall_constraint": constrained,
        "best_unconstrained_f1": unconstrained,
        "results_path": str(RESULTS_PATH),
        "note": (
            "The test split was not loaded. The persisted compatible models "
            "were compared because a fresh full retraining sweep exceeded the "
            "available memory/time budget."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
