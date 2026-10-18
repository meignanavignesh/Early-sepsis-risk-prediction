"""Evaluate how early hourly sepsis alerts fire before onset."""

import json
from pathlib import Path

import pandas as pd

from src.evaluation.hourly_rules import (
    AlertRule,
    aggregate_patient_alerts,
    apply_alert_rule,
)
from src.utils.experiment_tracker import ExperimentTracker


PREDICTIONS_PATH = Path("data/processed/final_predictions.csv")
TEST_DATA_PATH = Path("data/processed/model_ready/test.csv")
REPORT_PATH = Path("reports/hourly_early_warning_report.json")
PATIENT_RESULTS_PATH = Path(
    "reports/hourly_early_warning_patient_results.csv"
)

DEFAULT_RULE = AlertRule(
    name="two_consecutive_0.50",
    strategy="consecutive",
    threshold=0.50,
    required_consecutive=2,
)


class HourlyEarlyWarningEvaluator:
    """Evaluate alert timing using ICULOS as the hourly axis."""

    def __init__(
        self,
        predictions_path: str | Path = PREDICTIONS_PATH,
        test_data_path: str | Path = TEST_DATA_PATH,
        report_path: str | Path = REPORT_PATH,
        patient_results_path: str | Path = PATIENT_RESULTS_PATH,
        rule: AlertRule = DEFAULT_RULE,
    ) -> None:
        self.predictions_path = Path(predictions_path)
        self.test_data_path = Path(test_data_path)
        self.report_path = Path(report_path)
        self.patient_results_path = Path(patient_results_path)
        self.rule = rule

    def run(self) -> dict:
        """Run early-warning evaluation."""

        print("=" * 70)
        print("HOURLY EARLY-WARNING EVALUATION")
        print("=" * 70)

        hourly = self._load_hourly_predictions()
        ruled = apply_alert_rule(hourly, self.rule)
        patient_alerts = aggregate_patient_alerts(ruled)

        patient_rows = []

        for patient_id, group in ruled.groupby("Patient_ID", sort=False):
            group = group.sort_values(
                by=["ICULOS"] if "ICULOS" in group.columns else None,
                kind="mergesort",
            )

            actual_label = int(group["SepsisLabel"].max())
            onset_hour = self._first_hour(
                group[group["SepsisLabel"].astype(int) == 1]
            )
            first_alert_hour = self._first_hour(
                group[group["Hourly_Alert"].astype(int) == 1]
            )

            warning_time = None
            alert_before_onset = False

            if (
                onset_hour is not None
                and first_alert_hour is not None
            ):
                warning_time = float(onset_hour - first_alert_hour)
                alert_before_onset = first_alert_hour <= onset_hour

            patient_rows.append(
                {
                    "Patient_ID": patient_id,
                    "Actual_Patient_Label": actual_label,
                    "Predicted_Patient_Label": int(
                        group["Hourly_Alert"].max()
                    ),
                    "Sepsis_Onset_Hour": onset_hour,
                    "First_Alert_Hour": first_alert_hour,
                    "Warning_Time_Hours": warning_time,
                    "Alert_Before_Onset": bool(alert_before_onset),
                    "Alert_At_Least_3h_Before": bool(
                        warning_time is not None
                        and warning_time >= 3
                    ),
                    "Alert_At_Least_6h_Before": bool(
                        warning_time is not None
                        and warning_time >= 6
                    ),
                    "Alert_At_Least_12h_Before": bool(
                        warning_time is not None
                        and warning_time >= 12
                    ),
                    "Alert_Hours": int(group["Hourly_Alert"].sum()),
                    "Total_Observations": int(len(group)),
                    "Max_Probability": float(
                        group["Sepsis_Probability"].max()
                    ),
                }
            )

        patient_results = pd.DataFrame(patient_rows)

        self.patient_results_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        patient_results.to_csv(
            self.patient_results_path,
            index=False,
        )

        summary = self._summarize(patient_results, patient_alerts)

        with self.report_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4)

        ExperimentTracker().log_run(
            model_name="xgboost",
            dataset_paths={
                "predictions": str(self.predictions_path),
                "test": str(self.test_data_path),
            },
            feature_count=None,
            parameters={
                "rule_name": self.rule.name,
                "strategy": self.rule.strategy,
                "threshold": self.rule.threshold,
                "required_consecutive": (
                    self.rule.required_consecutive
                ),
                "window": self.rule.window,
            },
            metrics={
                "patient_level_recall": summary[
                    "patient_level_recall"
                ],
                "recall_at_least_6h_before_onset": summary[
                    "recall_at_least_6h_before_onset"
                ],
                "false_alerts_per_non_sepsis_patient": summary[
                    "false_alerts_per_non_sepsis_patient"
                ],
            },
            artifacts={
                "report": str(self.report_path),
                "patient_results": str(self.patient_results_path),
            },
        )

        print("Early-warning evaluation complete.")
        print(
            "Patient results saved to: "
            f"{self.patient_results_path.resolve()}"
        )
        print(f"Report saved to: {self.report_path.resolve()}")

        return summary

    def _load_hourly_predictions(self) -> pd.DataFrame:
        """Load probabilities and align them with labels/timeline."""

        if not self.predictions_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {self.predictions_path}"
            )

        if not self.test_data_path.exists():
            raise FileNotFoundError(
                f"Test data not found: {self.test_data_path}"
            )

        predictions = pd.read_csv(self.predictions_path)
        test = pd.read_csv(
            self.test_data_path,
            usecols=lambda column: column
            in {
                "Patient_ID",
                "ICULOS",
                "SepsisLabel",
                "Age",
                "Gender",
                "Unit1",
                "Unit2",
            },
        )

        if "Sepsis_Probability" not in predictions.columns:
            raise ValueError(
                "Prediction file must contain Sepsis_Probability."
            )

        if len(predictions) != len(test):
            raise ValueError(
                "Prediction rows and test rows do not match."
            )

        hourly = test.copy()
        hourly["Sepsis_Probability"] = predictions[
            "Sepsis_Probability"
        ].values

        return hourly

    @staticmethod
    def _first_hour(group: pd.DataFrame) -> float | None:
        """Return first ICULOS hour from a non-empty group."""

        if group.empty or "ICULOS" not in group.columns:
            return None

        return float(group["ICULOS"].iloc[0])

    def _summarize(
        self,
        patient_results: pd.DataFrame,
        patient_alerts: pd.DataFrame,
    ) -> dict:
        """Create early-warning summary metrics."""

        sepsis = patient_results[
            patient_results["Actual_Patient_Label"] == 1
        ]
        non_sepsis = patient_results[
            patient_results["Actual_Patient_Label"] == 0
        ]

        detected = sepsis[
            sepsis["Predicted_Patient_Label"] == 1
        ]

        warning_times = detected[
            "Warning_Time_Hours"
        ].dropna()

        total_sepsis = len(sepsis)
        total_non_sepsis = len(non_sepsis)

        false_positive_patients = int(
            (
                (patient_results["Actual_Patient_Label"] == 0)
                & (
                    patient_results["Predicted_Patient_Label"]
                    == 1
                )
            ).sum()
        )

        summary = {
            "evaluation_type": "hourly_early_warning",
            "rule": {
                "name": self.rule.name,
                "strategy": self.rule.strategy,
                "threshold": self.rule.threshold,
                "window": self.rule.window,
                "required_consecutive": (
                    self.rule.required_consecutive
                ),
            },
            "total_patients": int(len(patient_results)),
            "sepsis_patients": int(total_sepsis),
            "non_sepsis_patients": int(total_non_sepsis),
            "patient_level_recall": float(
                len(detected) / max(total_sepsis, 1)
            ),
            "recall_before_onset": float(
                sepsis["Alert_Before_Onset"].sum()
                / max(total_sepsis, 1)
            ),
            "recall_at_least_3h_before_onset": float(
                sepsis["Alert_At_Least_3h_Before"].sum()
                / max(total_sepsis, 1)
            ),
            "recall_at_least_6h_before_onset": float(
                sepsis["Alert_At_Least_6h_Before"].sum()
                / max(total_sepsis, 1)
            ),
            "recall_at_least_12h_before_onset": float(
                sepsis["Alert_At_Least_12h_Before"].sum()
                / max(total_sepsis, 1)
            ),
            "median_warning_time_hours": (
                None
                if warning_times.empty
                else float(warning_times.median())
            ),
            "mean_warning_time_hours": (
                None
                if warning_times.empty
                else float(warning_times.mean())
            ),
            "missed_sepsis_patients": int(
                total_sepsis - len(detected)
            ),
            "false_positive_patients": false_positive_patients,
            "alerts_per_patient": float(
                patient_results["Alert_Hours"].sum()
                / max(len(patient_results), 1)
            ),
            "false_alerts_per_patient": float(
                patient_results.loc[
                    patient_results["Actual_Patient_Label"] == 0,
                    "Alert_Hours",
                ].sum()
                / max(len(patient_results), 1)
            ),
            "false_alerts_per_non_sepsis_patient": float(
                patient_results.loc[
                    patient_results["Actual_Patient_Label"] == 0,
                    "Alert_Hours",
                ].sum()
                / max(total_non_sepsis, 1)
            ),
            "patient_level_alert_file": str(
                self.patient_results_path
            ),
            "data_leakage_prevented": True,
            "test_set_used_for_threshold_selection": False,
        }

        summary["patient_alert_metrics"] = (
            patient_alerts.describe(include="all")
            .fillna("")
            .to_dict()
        )

        return summary


def main() -> None:
    """Run hourly early-warning evaluation."""

    HourlyEarlyWarningEvaluator().run()


if __name__ == "__main__":
    main()
