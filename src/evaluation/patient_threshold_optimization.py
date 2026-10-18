"""Patient-level threshold optimization using the validation set."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


MODEL_PATH = Path(
    "models/xgboost.pkl"
)

VALIDATION_PATH = Path(
    "data/processed/model_ready/validation.csv"
)

REPORT_PATH = Path(
    "reports/patient_threshold_optimization_report.json"
)

THRESHOLDS = [
    round(x / 100, 2)
    for x in range(10, 91, 5)
]


class PatientThresholdOptimizer:
    """Find the best patient-level threshold on validation data."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        validation_path: str | Path = VALIDATION_PATH,
        report_path: str | Path = REPORT_PATH,
    ) -> None:

        self.model_path = Path(model_path)
        self.validation_path = Path(validation_path)
        self.report_path = Path(report_path)

    def run(self) -> dict:
        """Run patient-level threshold optimization."""

        print("=" * 70)
        print("PATIENT-LEVEL THRESHOLD OPTIMIZATION")
        print("=" * 70)

        # --------------------------------------------------
        # 1. Check files
        # --------------------------------------------------

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        if not self.validation_path.exists():
            raise FileNotFoundError(
                f"Validation dataset not found: "
                f"{self.validation_path}"
            )

        # --------------------------------------------------
        # 2. Load model
        # --------------------------------------------------

        print("Loading XGBoost model...")

        model = joblib.load(
            self.model_path
        )

        print(
            f"Model features: "
            f"{model.n_features_in_}"
        )

        # --------------------------------------------------
        # 3. Load validation data
        # --------------------------------------------------

        print("Loading validation dataset...")

        validation = pd.read_csv(
            self.validation_path
        )

        print(
            f"Validation shape: "
            f"{validation.shape}"
        )

        # --------------------------------------------------
        # 4. Identify model features
        # --------------------------------------------------

        excluded_columns = {
            "Patient_ID",
            "Dataset",
            "SepsisLabel",
        }

        feature_columns = [
            column
            for column in validation.columns
            if column not in excluded_columns
        ]

        if len(feature_columns) != model.n_features_in_:
            raise ValueError(
                "Feature count mismatch. "
                f"Model expects "
                f"{model.n_features_in_}, "
                f"but validation data contains "
                f"{len(feature_columns)} features."
            )

        X = validation[
            feature_columns
        ]

        # --------------------------------------------------
        # 5. Generate probabilities
        # --------------------------------------------------

        print(
            "Generating validation probabilities..."
        )

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        validation_predictions = validation[
            [
                "Patient_ID",
                "SepsisLabel",
            ]
        ].copy()

        validation_predictions[
            "Sepsis_Probability"
        ] = probabilities

        # --------------------------------------------------
        # 6. Aggregate observations to patients
        # --------------------------------------------------

        print(
            "Aggregating observations "
            "to patient level..."
        )

        patient_data = (
            validation_predictions
            .groupby(
                "Patient_ID",
                sort=False,
            )
            .agg(
                max_probability=(
                    "Sepsis_Probability",
                    "max",
                ),
                actual_positive_observations=(
                    "SepsisLabel",
                    "sum",
                ),
                total_observations=(
                    "SepsisLabel",
                    "count",
                ),
            )
            .reset_index()
        )

        patient_data[
            "actual_label"
        ] = (
            patient_data[
                "actual_positive_observations"
            ] > 0
        ).astype(int)

        print(
            f"Validation patients: "
            f"{len(patient_data)}"
        )

        print(
            "Validation sepsis patients: "
            f"{int(patient_data['actual_label'].sum())}"
        )

        # --------------------------------------------------
        # 7. Evaluate thresholds
        # --------------------------------------------------

        results = []

        for threshold in THRESHOLDS:

            predictions = (
                patient_data[
                    "max_probability"
                ] >= threshold
            ).astype(int)

            y_true = patient_data[
                "actual_label"
            ]

            accuracy = accuracy_score(
                y_true,
                predictions,
            )

            precision = precision_score(
                y_true,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_true,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_true,
                predictions,
                zero_division=0,
            )

            predicted_positive = int(
                predictions.sum()
            )

            false_positive = int(
                (
                    (y_true == 0)
                    & (predictions == 1)
                ).sum()
            )

            false_negative = int(
                (
                    (y_true == 1)
                    & (predictions == 0)
                ).sum()
            )

            results.append(
                {
                    "threshold": threshold,
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
                    "predicted_positive_patients": (
                        predicted_positive
                    ),
                    "false_positive_patients": (
                        false_positive
                    ),
                    "false_negative_patients": (
                        false_negative
                    ),
                }
            )

        results_dataframe = pd.DataFrame(
            results
        )

        # --------------------------------------------------
        # 8. Select threshold by F1
        # --------------------------------------------------

        best_row = (
            results_dataframe
            .sort_values(
                by=[
                    "f1_score",
                    "recall",
                    "precision",
                ],
                ascending=False,
            )
            .iloc[0]
        )

        best_threshold = float(
            best_row["threshold"]
        )

        # --------------------------------------------------
        # 9. Also find best threshold
        #    with recall >= 0.80
        # --------------------------------------------------

        recall_target = 0.80

        recall_constrained = (
            results_dataframe[
                results_dataframe["recall"]
                >= recall_target
            ]
            .sort_values(
                by=[
                    "precision",
                    "f1_score",
                    "recall",
                ],
                ascending=False,
            )
        )

        if not recall_constrained.empty:

            constrained_row = (
                recall_constrained.iloc[0]
            )

            constrained_threshold = float(
                constrained_row["threshold"]
            )

        else:

            constrained_threshold = None

        # --------------------------------------------------
        # 10. Save threshold table
        # --------------------------------------------------

        report_directory = (
            self.report_path.parent
        )

        report_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        threshold_table_path = (
            report_directory
            / "patient_threshold_results.csv"
        )

        results_dataframe.to_csv(
            threshold_table_path,
            index=False,
        )

        # --------------------------------------------------
        # 11. Create report
        # --------------------------------------------------

        report = {
            "evaluation_level": "patient",
            "dataset_used": "validation",
            "validation_patients": int(
                len(patient_data)
            ),
            "validation_sepsis_patients": int(
                patient_data[
                    "actual_label"
                ].sum()
            ),
            "thresholds_tested": [
                float(x)
                for x in THRESHOLDS
            ],
            "selection_metric": "F1-score",
            "selected_threshold": (
                best_threshold
            ),
            "selected_threshold_metrics": {
                "accuracy": float(
                    best_row["accuracy"]
                ),
                "precision": float(
                    best_row["precision"]
                ),
                "recall": float(
                    best_row["recall"]
                ),
                "f1_score": float(
                    best_row["f1_score"]
                ),
                "predicted_positive_patients": int(
                    best_row[
                        "predicted_positive_patients"
                    ]
                ),
                "false_positive_patients": int(
                    best_row[
                        "false_positive_patients"
                    ]
                ),
                "false_negative_patients": int(
                    best_row[
                        "false_negative_patients"
                    ]
                ),
            },
            "recall_target": recall_target,
            "best_threshold_with_recall_target": (
                constrained_threshold
            ),
            "test_set_used_for_selection": False,
            "data_leakage_prevented": True,
            "threshold_results": results,
            "threshold_table": str(
                threshold_table_path
            ),
        }

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
        print("=" * 70)
        print("PATIENT-LEVEL THRESHOLD OPTIMIZATION COMPLETE")
        print("=" * 70)

        print(
            f"Selected threshold: "
            f"{best_threshold:.2f}"
        )

        print(
            f"Validation accuracy: "
            f"{best_row['accuracy']:.4f}"
        )

        print(
            f"Validation precision: "
            f"{best_row['precision']:.4f}"
        )

        print(
            f"Validation recall: "
            f"{best_row['recall']:.4f}"
        )

        print(
            f"Validation F1-score: "
            f"{best_row['f1_score']:.4f}"
        )

        print()

        if constrained_threshold is not None:

            print(
                f"Best threshold with "
                f"recall >= {recall_target:.0%}: "
                f"{constrained_threshold:.2f}"
            )

        else:

            print(
                "No tested threshold achieved "
                f"recall >= {recall_target:.0%}."
            )

        print()

        print(
            f"Threshold table saved to: "
            f"{threshold_table_path.resolve()}"
        )

        print(
            f"Report saved to: "
            f"{self.report_path.resolve()}"
        )

        return report


def main() -> None:
    """Run patient-level threshold optimization."""

    optimizer = PatientThresholdOptimizer()

    optimizer.run()


if __name__ == "__main__":
    main()