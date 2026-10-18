"""Create patient-level explanation summaries for alert review."""

import json
from pathlib import Path

import pandas as pd


PREDICTIONS_PATH = Path("data/processed/final_predictions.csv")
TEST_DATA_PATH = Path("data/processed/model_ready/test.csv")
GLOBAL_IMPORTANCE_PATH = Path("reports/shap/global_feature_importance.csv")
OUTPUT_PATH = Path("reports/patient_explanations.csv")
ERROR_ANALYSIS_PATH = Path("reports/explanation_error_analysis.json")


class PatientExplanationGenerator:
    """Generate inspectable summaries for flagged patients."""

    def __init__(
        self,
        predictions_path: str | Path = PREDICTIONS_PATH,
        test_data_path: str | Path = TEST_DATA_PATH,
        global_importance_path: str | Path = GLOBAL_IMPORTANCE_PATH,
        output_path: str | Path = OUTPUT_PATH,
        error_analysis_path: str | Path = ERROR_ANALYSIS_PATH,
        threshold: float = 0.50,
    ) -> None:
        self.predictions_path = Path(predictions_path)
        self.test_data_path = Path(test_data_path)
        self.global_importance_path = Path(global_importance_path)
        self.output_path = Path(output_path)
        self.error_analysis_path = Path(error_analysis_path)
        self.threshold = threshold

    def run(self) -> dict:
        """Create patient summaries and error-analysis counts."""

        hourly = self._load()
        top_features = self._top_global_features()
        rows = []

        for patient_id, group in hourly.groupby("Patient_ID", sort=False):
            group = group.sort_values(
                by=["ICULOS"] if "ICULOS" in group.columns else None,
                kind="mergesort",
            )
            prediction = int(
                (group["Sepsis_Probability"] >= self.threshold).any()
            )
            actual = int(group["SepsisLabel"].max())

            if actual == 1 and prediction == 1:
                outcome = "true_positive"
            elif actual == 0 and prediction == 1:
                outcome = "false_positive"
            elif actual == 1 and prediction == 0:
                outcome = "false_negative"
            else:
                outcome = "true_negative"

            max_idx = group["Sepsis_Probability"].idxmax()
            max_row = group.loc[max_idx]
            alert_rows = group[
                group["Sepsis_Probability"] >= self.threshold
            ]

            first_alert_hour = None
            if not alert_rows.empty and "ICULOS" in alert_rows.columns:
                first_alert_hour = float(alert_rows["ICULOS"].iloc[0])

            key_vitals = {
                column: self._safe_float(max_row[column])
                for column in [
                    "HR",
                    "O2Sat",
                    "Temp",
                    "SBP",
                    "MAP",
                    "DBP",
                    "Resp",
                    "Lactate",
                    "FiO2",
                    "Creatinine",
                    "WBC",
                    "Platelets",
                ]
                if column in group.columns
            }

            probabilities = group["Sepsis_Probability"].astype(float)
            recent_trend = None
            if len(probabilities) > 3:
                recent_trend = float(
                    probabilities.iloc[-1] - probabilities.iloc[-4]
                )

            rows.append(
                {
                    "Patient_ID": patient_id,
                    "Outcome": outcome,
                    "Actual_Label": actual,
                    "Predicted_Label": prediction,
                    "Max_Risk_Hour": self._safe_float(
                        max_row["ICULOS"]
                    )
                    if "ICULOS" in group.columns
                    else None,
                    "First_Alert_Hour": first_alert_hour,
                    "Max_Probability": float(
                        max_row["Sepsis_Probability"]
                    ),
                    "Recent_Risk_Trend": recent_trend,
                    "Top_Global_Features": ", ".join(top_features),
                    "Key_Vitals_At_Max_Risk": json.dumps(
                        key_vitals,
                        sort_keys=True,
                    ),
                }
            )

        explanations = pd.DataFrame(rows)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        explanations.to_csv(self.output_path, index=False)

        analysis = {
            "total_patients": int(len(explanations)),
            "threshold": self.threshold,
            "top_global_features_used": top_features,
            "outcome_counts": explanations["Outcome"]
            .value_counts()
            .to_dict(),
            "output_csv": str(self.output_path),
            "note": (
                "Top feature names come from global SHAP importance "
                "when available. Per-patient SHAP can be added later "
                "for exact local attribution."
            ),
        }

        with self.error_analysis_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(analysis, file, indent=4)

        print(f"Explanations saved to: {self.output_path.resolve()}")
        print(
            "Error analysis saved to: "
            f"{self.error_analysis_path.resolve()}"
        )

        return analysis

    def _load(self) -> pd.DataFrame:
        """Load prediction probabilities and available patient features."""

        if not self.predictions_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: {self.predictions_path}"
            )

        if not self.test_data_path.exists():
            raise FileNotFoundError(
                f"Test data not found: {self.test_data_path}"
            )

        predictions = pd.read_csv(self.predictions_path)
        test = pd.read_csv(self.test_data_path)

        if len(predictions) != len(test):
            raise ValueError(
                "Prediction rows and test rows do not match."
            )

        result = test.copy()
        result["Sepsis_Probability"] = predictions[
            "Sepsis_Probability"
        ].values

        return result

    def _top_global_features(self) -> list[str]:
        """Load top global SHAP features when available."""

        if not self.global_importance_path.exists():
            return []

        importance = pd.read_csv(self.global_importance_path)

        if importance.empty:
            return []

        feature_column = importance.columns[0]

        return [
            str(value)
            for value in importance[feature_column]
            .head(10)
            .tolist()
        ]

    @staticmethod
    def _safe_float(value):
        """Convert values to floats while preserving missing data."""

        if pd.isna(value):
            return None

        return float(value)


def main() -> None:
    """Run patient explanation generation."""

    PatientExplanationGenerator().run()


if __name__ == "__main__":
    main()
