"""Memory-conscious A/B test of missing causal feature groups.

Experiment A uses the current 196 model-ready features. Experiment B adds
causal 12-hour summaries, time-since-measured values, and documented clinical
interactions. Only validation is loaded for comparison; the test split is not
opened by this experiment.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluation.final_temporal_patient_evaluation import (
    evaluate_prediction_frame,
)


DATA_DIR = Path("data/processed/model_ready")
REPORT_DIR = Path("reports")
RESULTS_PATH = REPORT_DIR / "focused_feature_comparison.json"
TABLE_PATH = REPORT_DIR / "focused_feature_comparison.csv"
MARKDOWN_PATH = REPORT_DIR / "focused_feature_comparison.md"
THRESHOLD = 0.60
REQUIRED_CONSECUTIVE = 2
SCALE_POS_WEIGHT = 27.271881390593048
SEED = 42

TEMPORAL_COLUMNS = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "Resp",
    "Lactate",
)


def _feature_columns() -> list[str]:
    columns = pd.read_csv(DATA_DIR / "train.csv", nrows=0).columns.tolist()
    return [
        column
        for column in columns
        if column not in {"Patient_ID", "Dataset", "SepsisLabel"}
    ]


def _load_split(path: Path, feature_columns: list[str]) -> pd.DataFrame:
    usecols = ["Patient_ID", "ICULOS", "SepsisLabel", *feature_columns]
    dtype = {column: "float32" for column in feature_columns}
    dtype.update({"Patient_ID": "string", "SepsisLabel": "int8"})
    frame = pd.read_csv(path, usecols=usecols, dtype=dtype)
    return frame.sort_values(
        ["Patient_ID", "ICULOS"], kind="mergesort"
    ).reset_index(drop=True)


def _add_expanded_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add only current/past patient-wise features to a split."""

    result = frame.copy()
    added: list[str] = []
    grouped = result.groupby("Patient_ID", sort=False)

    for column in TEMPORAL_COLUMNS:
        values = result[column].astype("float32")
        series = grouped[column]
        rolling = series.rolling(window=12, min_periods=1)
        for statistic, values_by_row in (
            ("mean", rolling.mean()),
            ("min", rolling.min()),
            ("max", rolling.max()),
            ("std", rolling.std().fillna(0.0)),
        ):
            name = f"{column}_rolling_{statistic}_12"
            result[name] = values_by_row.reset_index(level=0, drop=True).astype(
                "float32"
            )
            added.append(name)
        trend_name = f"{column}_trend_12"
        result[trend_name] = (
            values - series.shift(12)
        ).astype("float32")
        added.append(trend_name)

        missing_column = (
            f"{column}_was_missing"
            if f"{column}_was_missing" in result.columns
            else f"{column}_missing"
        )
        if missing_column in result.columns:
            measured = result[missing_column].eq(0)
            last_measured = (
                result["ICULOS"].where(measured)
                .groupby(result["Patient_ID"], sort=False)
                .ffill()
            )
            name = f"{column}_time_since_measured"
            result[name] = (
                result["ICULOS"] - last_measured
            ).fillna(-1).astype("float32")
            added.append(name)

    interactions = {
        "Shock_Index": result["HR"] / result["SBP"].replace(0, np.nan),
        "MAP_Below_65": (result["MAP"] < 65).astype("float32"),
        "Lactate_Above_2": (result["Lactate"] > 2).astype("float32"),
        "O2Sat_Below_92": (result["O2Sat"] < 92).astype("float32"),
        "Resp_Above_22": (result["Resp"] > 22).astype("float32"),
        "Age_x_HR": result["Age"] * result["HR"],
        "Age_x_Resp": result["Age"] * result["Resp"],
        "Lactate_x_MAP": result["Lactate"] * result["MAP"],
        "Resp_x_O2Sat": result["Resp"] * result["O2Sat"],
    }
    for name, values in interactions.items():
        result[name] = values.replace([np.inf, -np.inf], np.nan).fillna(0).astype(
            "float32"
        )
        added.append(name)

    return result, added


def _train_and_validate(
    train_path: Path,
    validation_path: Path,
    feature_columns: list[str],
    expanded: bool,
) -> dict:
    train = _load_split(train_path, feature_columns)
    if expanded:
        train, added = _add_expanded_features(train)
    else:
        added = []
    all_features = [*feature_columns, *added]
    model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=4,
        subsample=0.80,
        colsample_bytree=0.70,
        reg_lambda=5,
        scale_pos_weight=SCALE_POS_WEIGHT,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_bin=64,
        n_jobs=1,
        random_state=SEED,
    )
    print(
        f"Training {'B expanded' if expanded else 'A baseline'} "
        f"with {len(all_features)} features...",
        flush=True,
    )
    X_train = train[all_features].to_numpy(dtype=np.float32, copy=True)
    y_train = train["SepsisLabel"].to_numpy(dtype=np.int8, copy=True)
    del train
    gc.collect()
    model.fit(X_train, y_train, verbose=False)
    del X_train, y_train
    gc.collect()

    validation = _load_split(validation_path, feature_columns)
    if expanded:
        validation, _ = _add_expanded_features(validation)
    prediction_frame = validation[
        ["Patient_ID", "ICULOS", "SepsisLabel"]
    ].copy()
    prediction_frame["Sepsis_Probability"] = model.predict_proba(
        validation[all_features]
    )[:, 1]
    _, evaluation_report = evaluate_prediction_frame(
        prediction_frame,
        threshold=THRESHOLD,
        required_consecutive=REQUIRED_CONSECUTIVE,
    )
    metrics = evaluation_report["metrics"]
    result = {
        "experiment": "B Expanded" if expanded else "A Current",
        "feature_count": len(all_features),
        "added_features": added,
        "threshold": THRESHOLD,
        "required_consecutive": REQUIRED_CONSECUTIVE,
        "scale_pos_weight": SCALE_POS_WEIGHT,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1_score"],
        "specificity": metrics["specificity"],
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "true_positives": metrics["true_positives"],
        "true_negatives": metrics["true_negatives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "false_alerts_per_icu_stay": metrics["false_alerts_per_icu_stay"],
    }
    del model, validation, prediction_frame
    gc.collect()
    return result


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    feature_columns = _feature_columns()
    results = []
    results.append(
        _train_and_validate(
            DATA_DIR / "train.csv",
            DATA_DIR / "validation.csv",
            feature_columns,
            expanded=False,
        )
    )
    results.append(
        _train_and_validate(
            DATA_DIR / "train.csv",
            DATA_DIR / "validation.csv",
            feature_columns,
            expanded=True,
        )
    )
    baseline, expanded = results
    differences = {
        metric: float(expanded[metric] - baseline[metric])
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "roc_auc",
            "pr_auc",
            "false_alerts_per_icu_stay",
        )
    }
    report = {
        "test_set_loaded": False,
        "patient_split_reused": True,
        "leakage_checks": {
            "future_observations_used": False,
            "centered_windows_used": False,
            "patient_id_used_as_feature": False,
            "target_used_as_feature": False,
        },
        "class_imbalance_strategy": "same scale_pos_weight for A and B",
        "results": results,
        "expanded_minus_baseline": differences,
        "production_recommendation": (
            "replace only if validation improvement is meaningful and does not "
            "create an unacceptable precision/recall or alert-burden tradeoff"
        ),
    }
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(results).drop(columns=["added_features"]).to_csv(
        TABLE_PATH, index=False
    )
    lines = [
        "# Focused Feature Comparison",
        "",
        "This experiment used the existing patient-level train/validation split.",
        "The test split was not loaded.",
        "",
        "## Controlled Setup",
        "",
        f"- Threshold: `{THRESHOLD}`",
        f"- Required consecutive observations: `{REQUIRED_CONSECUTIVE}`",
        f"- Same `scale_pos_weight`: `{SCALE_POS_WEIGHT}`",
        "- No centered windows, future rows, future labels, or SMOTE.",
        "",
        "## Validation Results",
        "",
        "| Experiment | Features | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['experiment']} | {row['feature_count']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['specificity']:.4f} | "
            f"{row['roc_auc']:.4f} | {row['pr_auc']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Metric Differences (B minus A)",
            "",
            "```json",
            json.dumps(differences, indent=2),
            "```",
            "",
            "## Decision",
            "",
            "The expanded model is not promoted automatically. The test set must "
            "be evaluated only after a validation-based decision to promote it.",
        ]
    )
    MARKDOWN_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
