"""Patient-level evaluation for the final sepsis prediction model."""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PREDICTIONS_PATH = Path(
    "data/processed/test_predictions.csv"
)

TEST_DATA_PATH = Path(
    "data/processed/model_ready/test.csv"
)

REPORT_PATH = Path(
    "reports/patient_level_evaluation_report.json"
)

THRESHOLD = 0.55


class PatientLevelEvaluator:
    """Evaluate sepsis predictions at the patient level."""

    def __init__(
        self,
        predictions_path: str | Path = PREDICTIONS_PATH,
        test_data_path: str | Path = TEST_DATA_PATH,
        report_path: str | Path = REPORT_PATH,
        threshold: float = THRESHOLD,
    ) -> None:

        self.predictions_path = Path(
            predictions_path
        )

        self.test_data_path = Path(
            test_data_path
        )

        self.report_path = Path(
            report_path
        )

        self.threshold = threshold

    def run(self) -> dict:
        """Perform patient-level evaluation."""

        print("=" * 65)
        print("PATIENT-LEVEL SEPSIS EVALUATION")
        print("=" * 65)

        # --------------------------------------------------
        # 1. Check files
        # --------------------------------------------------

        if not self.predictions_path.exists():
            raise FileNotFoundError(
                f"Prediction file not found: "
                f"{self.predictions_path}"
            )

        if not self.test_data_path.exists():
            raise FileNotFoundError(
                f"Test dataset not found: "
                f"{self.test_data_path}"
            )

        # --------------------------------------------------
        # 2. Load predictions
        # --------------------------------------------------

        print("Loading prediction results...")

        predictions = pd.read_csv(
            self.predictions_path
        )

        print(
            f"Prediction rows: "
            f"{len(predictions)}"
        )

        # --------------------------------------------------
        # 3. Load actual labels
        # --------------------------------------------------

        print("Loading actual test labels...")

        test_data = pd.read_csv(
            self.test_data_path,
            usecols=[
                "Patient_ID",
                "SepsisLabel",
            ],
        )

        print(
            f"Test label rows: "
            f"{len(test_data)}"
        )

        # --------------------------------------------------
        # 4. Verify row alignment
        # --------------------------------------------------

        if len(predictions) != len(test_data):
            raise ValueError(
                "Prediction rows and test-label rows "
                "do not match."
            )

        # --------------------------------------------------
        # 5. Attach actual labels
        # --------------------------------------------------

        evaluation_data = predictions[
            [
                "Patient_ID",
                "Sepsis_Probability",
                "Sepsis_Prediction",
            ]
        ].copy()

        evaluation_data[
            "Actual_SepsisLabel"
        ] = test_data[
            "SepsisLabel"
        ].values

        # --------------------------------------------------
        # 6. Aggregate to patient level
        # --------------------------------------------------

        patient_data = (
            evaluation_data
            .groupby(
                "Patient_ID",
                sort=False,
            )
            .agg(
                max_predicted_probability=(
                    "Sepsis_Probability",
                    "max",
                ),
                predicted_positive_observations=(
                    "Sepsis_Prediction",
                    "sum",
                ),
                actual_positive_observations=(
                    "Actual_SepsisLabel",
                    "sum",
                ),
                total_observations=(
                    "Sepsis_Prediction",
                    "count",
                ),
            )
            .reset_index()
        )

        # --------------------------------------------------
        # 7. Patient-level predictions
        # --------------------------------------------------

        patient_data[
            "Patient_Prediction"
        ] = (
            patient_data[
                "max_predicted_probability"
            ]
            >= self.threshold
        ).astype(int)

        # Actual patient label:
        # positive if at least one observation
        # has SepsisLabel = 1.

        patient_data[
            "Patient_Actual_Label"
        ] = (
            patient_data[
                "actual_positive_observations"
            ]
            > 0
        ).astype(int)

        # --------------------------------------------------
        # 8. Calculate metrics
        # --------------------------------------------------

        y_true = patient_data[
            "Patient_Actual_Label"
        ]

        y_pred = patient_data[
            "Patient_Prediction"
        ]

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        )

        true_negative = int(
            matrix[0, 0]
        )

        false_positive = int(
            matrix[0, 1]
        )

        false_negative = int(
            matrix[1, 0]
        )

        true_positive = int(
            matrix[1, 1]
        )

        # --------------------------------------------------
        # 9. Patient counts
        # --------------------------------------------------

        total_patients = len(
            patient_data
        )

        actual_sepsis_patients = int(
            y_true.sum()
        )

        actual_non_sepsis_patients = (
            total_patients
            - actual_sepsis_patients
        )

        predicted_sepsis_patients = int(
            y_pred.sum()
        )

        predicted_non_sepsis_patients = (
            total_patients
            - predicted_sepsis_patients
        )

        # --------------------------------------------------
        # 10. Save patient predictions
        # --------------------------------------------------

        patient_output_path = Path(
            "data/processed/"
            "patient_level_predictions.csv"
        )

        patient_output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        patient_data.to_csv(
            patient_output_path,
            index=False,
        )

        # --------------------------------------------------
        # 11. Create report
        # --------------------------------------------------

        report = {
            "evaluation_type": (
                "patient_level"
            ),
            "threshold": float(
                self.threshold
            ),
            "total_patients": int(
                total_patients
            ),
            "actual_sepsis_patients": int(
                actual_sepsis_patients
            ),
            "actual_non_sepsis_patients": int(
                actual_non_sepsis_patients
            ),
            "predicted_sepsis_patients": int(
                predicted_sepsis_patients
            ),
            "predicted_non_sepsis_patients": int(
                predicted_non_sepsis_patients
            ),
            "accuracy": float(
                accuracy
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1_score": float(
                f1
            ),
            "confusion_matrix": matrix.tolist(),
            "true_negatives": true_negative,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "true_positives": true_positive,
            "aggregation_method": (
                "maximum predicted probability "
                "across patient observations"
            ),
            "actual_patient_label_method": (
                "patient considered positive if "
                "at least one observation has "
                "SepsisLabel=1"
            ),
            "patient_leakage": False,
            "output": str(
                patient_output_path
            ),
        }

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self.report_path.open(
            "w",
            encoding="utf-8",
        ) as report_file:

            json.dump(
                report,
                report_file,
                indent=4,
            )

        # --------------------------------------------------
        # 12. Print results
        # --------------------------------------------------

        print()
        print("=" * 65)
        print("PATIENT-LEVEL EVALUATION COMPLETE")
        print("=" * 65)

        print(
            f"Total patients: "
            f"{total_patients}"
        )

        print(
            f"Actual sepsis patients: "
            f"{actual_sepsis_patients}"
        )

        print(
            f"Actual non-sepsis patients: "
            f"{actual_non_sepsis_patients}"
        )

        print(
            f"Predicted sepsis patients: "
            f"{predicted_sepsis_patients}"
        )

        print(
            f"Predicted non-sepsis patients: "
            f"{predicted_non_sepsis_patients}"
        )

        print()
        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Precision: "
            f"{precision:.4f}"
        )

        print(
            f"Recall: "
            f"{recall:.4f}"
        )

        print(
            f"F1-score: "
            f"{f1:.4f}"
        )

        print()
        print("Confusion matrix:")

        print(matrix)

        print()
        print(
            f"True positives: "
            f"{true_positive}"
        )

        print(
            f"False positives: "
            f"{false_positive}"
        )

        print(
            f"False negatives: "
            f"{false_negative}"
        )

        print(
            f"True negatives: "
            f"{true_negative}"
        )

        print()
        print(
            f"Patient predictions saved to: "
            f"{patient_output_path.resolve()}"
        )

        print(
            f"Report saved to: "
            f"{self.report_path.resolve()}"
        )

        return report


def main() -> None:
    """Run patient-level evaluation."""

    evaluator = PatientLevelEvaluator()

    evaluator.run()


if __name__ == "__main__":
    main()