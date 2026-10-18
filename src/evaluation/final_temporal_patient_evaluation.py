"""Locked patient-level evaluation using the deployed alert rule."""

from __future__ import annotations

import json
from pathlib import Path

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


PREDICTIONS_PATH = Path("data/processed/final_predictions.csv")
REPORT_PATH = Path("reports/final_temporal_patient_evaluation_report.json")
PATIENT_OUTPUT_PATH = Path(
    "data/processed/final_temporal_patient_predictions.csv"
)

# These values were selected using validation data only.
FINAL_THRESHOLD = 0.60
REQUIRED_CONSECUTIVE = 2


def has_consecutive_positive(
    probabilities: list[float],
    threshold: float,
    required_consecutive: int,
) -> int:
    """Return one when a sequence contains a sustained threshold crossing."""

    consecutive = 0
    for probability in probabilities:
        consecutive = consecutive + 1 if probability >= threshold else 0
        if consecutive >= required_consecutive:
            return 1
    return 0


def _alert_flags(
    probabilities: list[float],
    threshold: float,
    required_consecutive: int,
) -> list[int]:
    """Return persistent alert flags for each hourly observation."""

    flags: list[int] = []
    consecutive = 0
    fired = False
    for probability in probabilities:
        consecutive = consecutive + 1 if probability >= threshold else 0
        if consecutive >= required_consecutive:
            fired = True
        flags.append(int(fired))
    return flags


def evaluate_prediction_frame(
    predictions: pd.DataFrame,
    threshold: float = FINAL_THRESHOLD,
    required_consecutive: int = REQUIRED_CONSECUTIVE,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate canonical hourly predictions at patient level."""

    required = {"Patient_ID", "ICULOS", "SepsisLabel", "Sepsis_Probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            f"Prediction output is missing columns: {sorted(missing)}"
        )

    ordered = predictions[
        ["Patient_ID", "ICULOS", "SepsisLabel", "Sepsis_Probability"]
    ].copy()
    ordered["Sepsis_Probability"] = ordered["Sepsis_Probability"].astype(float)
    ordered = ordered.sort_values(
        ["Patient_ID", "ICULOS"], kind="mergesort"
    ).reset_index(drop=True)

    patient_rows: list[dict] = []
    for patient_id, group in ordered.groupby("Patient_ID", sort=False):
        probabilities = group["Sepsis_Probability"].tolist()
        flags = _alert_flags(probabilities, threshold, required_consecutive)
        first_alert_position = next(
            (index for index, value in enumerate(flags) if value), None
        )
        actual_label = int(group["SepsisLabel"].astype(int).max())
        onset_rows = group[group["SepsisLabel"].astype(int) == 1]
        onset_hour = (
            float(onset_rows["ICULOS"].iloc[0])
            if not onset_rows.empty
            else None
        )
        first_alert_hour = (
            float(group["ICULOS"].iloc[first_alert_position])
            if first_alert_position is not None
            else None
        )
        patient_rows.append(
            {
                "Patient_ID": patient_id,
                "Actual_Patient_Label": actual_label,
                "Predicted_Patient_Label": int(max(flags, default=0)),
                "Max_Probability": float(max(probabilities, default=0.0)),
                "Mean_Probability": float(
                    sum(probabilities) / max(len(probabilities), 1)
                ),
                "Alert_Hours": int(sum(flags)),
                "Total_Observations": int(len(group)),
                "First_Alert_Hour": first_alert_hour,
                "Sepsis_Onset_Hour": onset_hour,
                "Lead_Time_Hours": (
                    onset_hour - first_alert_hour
                    if onset_hour is not None and first_alert_hour is not None
                    else None
                ),
            }
        )

    patient_data = pd.DataFrame(patient_rows)
    y_true = patient_data["Actual_Patient_Label"].astype(int)
    y_pred = patient_data["Predicted_Patient_Label"].astype(int)
    scores = patient_data["Max_Probability"].astype(float)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    non_sepsis = patient_data["Actual_Patient_Label"].eq(0)
    detected_sepsis = patient_data[
        (patient_data["Actual_Patient_Label"] == 1)
        & patient_data["Lead_Time_Hours"].notna()
    ]
    lead_times = detected_sepsis["Lead_Time_Hours"].astype(float)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "roc_auc": float(roc_auc_score(y_true, scores))
        if y_true.nunique() > 1
        else None,
        "pr_auc": float(average_precision_score(y_true, scores))
        if y_true.nunique() > 1
        else None,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "alerts_per_patient": float(patient_data["Alert_Hours"].mean()),
        "false_alerts_per_icu_stay": float(
            patient_data.loc[non_sepsis, "Alert_Hours"].sum()
            / max(int(non_sepsis.sum()), 1)
        ),
        "detected_before_onset": int((lead_times > 0).sum()),
        "detected_at_least_3h_before": int((lead_times >= 3).sum()),
        "detected_at_least_6h_before": int((lead_times >= 6).sum()),
        "detected_at_least_12h_before": int((lead_times >= 12).sum()),
        "median_lead_time_hours": float(lead_times.median())
        if not lead_times.empty
        else None,
    }
    report = {
        "evaluation_type": "patient_level_temporal_test",
        "model_output": str(PREDICTIONS_PATH),
        "patients": int(len(patient_data)),
        "threshold": float(threshold),
        "required_consecutive": int(required_consecutive),
        "decision_rule": (
            f"At least {required_consecutive} consecutive ICU observations "
            f"with Sepsis_Probability >= {threshold:.2f}"
        ),
        "test_set_used_for_optimization": False,
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
    }
    return patient_data, report


class FinalTemporalPatientEvaluator:
    """Evaluate the frozen production output without changing the model."""

    def __init__(
        self,
        predictions_path: str | Path = PREDICTIONS_PATH,
        report_path: str | Path = REPORT_PATH,
        patient_output_path: str | Path = PATIENT_OUTPUT_PATH,
        threshold: float = FINAL_THRESHOLD,
        required_consecutive: int = REQUIRED_CONSECUTIVE,
    ) -> None:
        self.predictions_path = Path(predictions_path)
        self.report_path = Path(report_path)
        self.patient_output_path = Path(patient_output_path)
        self.threshold = threshold
        self.required_consecutive = required_consecutive

    def run(self) -> dict:
        if not self.predictions_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {self.predictions_path}"
            )
        predictions = pd.read_csv(self.predictions_path)
        patient_data, report = evaluate_prediction_frame(
            predictions,
            threshold=self.threshold,
            required_consecutive=self.required_consecutive,
        )
        self.patient_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        patient_data.to_csv(self.patient_output_path, index=False)
        self.report_path.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return report


def main() -> None:
    FinalTemporalPatientEvaluator().run()


if __name__ == "__main__":
    main()
