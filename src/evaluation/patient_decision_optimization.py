"""Optimize patient-level sepsis decision rules using validation data only."""

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
    "reports/patient_decision_optimization_report.json"
)

THRESHOLDS = [
    round(x / 100, 2)
    for x in range(20, 91, 5)
]


def calculate_consecutive_positive(
    probabilities: list[float],
    threshold: float,
    required_consecutive: int,
) -> int:
    """Return 1 if required consecutive observations exceed threshold."""

    consecutive = 0

    for probability in probabilities:

        if probability >= threshold:
            consecutive += 1

            if consecutive >= required_consecutive:
                return 1

        else:
            consecutive = 0

    return 0


class PatientDecisionOptimizer:
    """Compare patient-level aggregation strategies."""

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
        """Optimize patient-level decision rules."""

        print("=" * 70)
        print("PATIENT-LEVEL DECISION OPTIMIZATION")
        print("=" * 70)

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
        # Load model
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
        # Load validation data
        # --------------------------------------------------

        print("Loading validation dataset...")

        dataframe = pd.read_csv(
            self.validation_path
        )

        print(
            f"Validation shape: "
            f"{dataframe.shape}"
        )

        # --------------------------------------------------
        # Identify features
        # --------------------------------------------------

        excluded = {
            "Patient_ID",
            "Dataset",
            "SepsisLabel",
        }

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in excluded
        ]

        if len(feature_columns) != model.n_features_in_:
            raise ValueError(
                "Feature count mismatch. "
                f"Model expects "
                f"{model.n_features_in_}, "
                f"but validation data has "
                f"{len(feature_columns)} features."
            )

        # --------------------------------------------------
        # Generate probabilities
        # --------------------------------------------------

        print(
            "Generating validation probabilities..."
        )

        X = dataframe[
            feature_columns
        ]

        dataframe[
            "Probability"
        ] = model.predict_proba(X)[:, 1]

        # --------------------------------------------------
        # Sort chronologically
        # --------------------------------------------------

        if "ICULOS" in dataframe.columns:
            dataframe = dataframe.sort_values(
                by=[
                    "Patient_ID",
                    "ICULOS",
                ],
                kind="mergesort",
            )
        else:
            dataframe = dataframe.sort_values(
                by=["Patient_ID"],
                kind="mergesort",
            )

        # --------------------------------------------------
        # Create patient-level summary
        # --------------------------------------------------

        print(
            "Creating patient-level summaries..."
        )

        patient_records = []

        for patient_id, group in dataframe.groupby(
            "Patient_ID",
            sort=False,
        ):

            probabilities = (
                group[
                    "Probability"
                ]
                .astype(float)
                .tolist()
            )

            labels = (
                group[
                    "SepsisLabel"
                ]
                .astype(int)
            )

            patient_records.append(
                {
                    "Patient_ID": patient_id,
                    "probabilities": probabilities,
                    "max_probability": max(
                        probabilities
                    ),
                    "positive_observations": 0,
                    "actual_label": int(
                        labels.max()
                    ),
                }
            )

        patients = pd.DataFrame(
            patient_records
        )

        print(
            f"Validation patients: "
            f"{len(patients)}"
        )

        # --------------------------------------------------
        # Evaluate strategies
        # --------------------------------------------------

        results = []

        for threshold in THRESHOLDS:

            # ----------------------------------------------
            # Strategy 1: Any observation
            # ----------------------------------------------

            any_prediction = (
                patients[
                    "max_probability"
                ]
                >= threshold
            ).astype(int)

            self._add_result(
                results,
                patients,
                any_prediction,
                threshold,
                "any_observation",
            )

            # ----------------------------------------------
            # Strategy 2: At least 2 positive observations
            # ----------------------------------------------

            positive_counts = []

            for probabilities in patients[
                "probabilities"
            ]:

                count = sum(
                    probability >= threshold
                    for probability in probabilities
                )

                positive_counts.append(
                    count
                )

            patients[
                "positive_count"
            ] = positive_counts

            for minimum_count in (
                2,
                3,
            ):

                prediction = (
                    patients[
                        "positive_count"
                    ]
                    >= minimum_count
                ).astype(int)

                self._add_result(
                    results,
                    patients,
                    prediction,
                    threshold,
                    (
                        f"at_least_"
                        f"{minimum_count}"
                        f"_positive_observations"
                    ),
                )

            # ----------------------------------------------
            # Strategy 3: Consecutive observations
            # ----------------------------------------------

            for consecutive_count in (
                2,
                3,
            ):

                prediction_values = []

                for probabilities in patients[
                    "probabilities"
                ]:

                    prediction = (
                        calculate_consecutive_positive(
                            probabilities,
                            threshold,
                            consecutive_count,
                        )
                    )

                    prediction_values.append(
                        prediction
                    )

                prediction_series = pd.Series(
                    prediction_values,
                    index=patients.index,
                )

                self._add_result(
                    results,
                    patients,
                    prediction_series,
                    threshold,
                    (
                        f"{consecutive_count}"
                        "_consecutive_observations"
                    ),
                )

        results_dataframe = pd.DataFrame(
            results
        )

        # --------------------------------------------------
        # Select best strategy
        # --------------------------------------------------

        best = (
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

        # Best strategy maintaining recall >= 80%
        recall_target = 0.80

        recall_constrained = (
            results_dataframe[
                results_dataframe[
                    "recall"
                ] >= recall_target
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

            best_recall_strategy = (
                recall_constrained.iloc[0]
            )

        else:

            best_recall_strategy = None

        # --------------------------------------------------
        # Save results
        # --------------------------------------------------

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        table_path = (
            self.report_path.parent
            / "patient_decision_results.csv"
        )

        results_dataframe.to_csv(
            table_path,
            index=False,
        )

        report = {
            "dataset": "validation",
            "validation_patients": int(
                len(patients)
            ),
            "optimization_metric": "F1-score",
            "recall_target": recall_target,
            "best_overall_strategy": (
                best.to_dict()
            ),
            "best_strategy_with_recall_at_least_80_percent": (
                (
                    best_recall_strategy.to_dict()
                    if best_recall_strategy
                    is not None
                    else None
                )
            ),
            "test_set_used": False,
            "data_leakage_prevented": True,
            "results_table": str(
                table_path
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
                default=float,
            )

        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print()
        print("=" * 70)
        print(
            "PATIENT-LEVEL DECISION OPTIMIZATION COMPLETE"
        )
        print("=" * 70)

        print()
        print("BEST OVERALL STRATEGY")

        print(
            f"Strategy: "
            f"{best['strategy']}"
        )

        print(
            f"Threshold: "
            f"{best['threshold']:.2f}"
        )

        print(
            f"Accuracy: "
            f"{best['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{best['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best['recall']:.4f}"
        )

        print(
            f"F1-score: "
            f"{best['f1_score']:.4f}"
        )

        print()
        print(
            "BEST STRATEGY WITH RECALL >= 80%"
        )

        if best_recall_strategy is not None:

            print(
                f"Strategy: "
                f"{best_recall_strategy['strategy']}"
            )

            print(
                f"Threshold: "
                f"{best_recall_strategy['threshold']:.2f}"
            )

            print(
                f"Accuracy: "
                f"{best_recall_strategy['accuracy']:.4f}"
            )

            print(
                f"Precision: "
                f"{best_recall_strategy['precision']:.4f}"
            )

            print(
                f"Recall: "
                f"{best_recall_strategy['recall']:.4f}"
            )

            print(
                f"F1-score: "
                f"{best_recall_strategy['f1_score']:.4f}"
            )

        else:

            print(
                "No strategy achieved recall >= 80%."
            )

        print()
        print(
            f"Results saved to: "
            f"{table_path.resolve()}"
        )

        print(
            f"Report saved to: "
            f"{self.report_path.resolve()}"
        )

        return report

    @staticmethod
    def _add_result(
        results: list[dict],
        patients: pd.DataFrame,
        predictions: pd.Series,
        threshold: float,
        strategy: str,
    ) -> None:
        """Add metrics for one decision strategy."""

        y_true = patients[
            "actual_label"
        ]

        predictions = pd.Series(
            predictions,
            index=patients.index,
        )

        results.append(
            {
                "strategy": strategy,
                "threshold": float(
                    threshold
                ),
                "accuracy": float(
                    accuracy_score(
                        y_true,
                        predictions,
                    )
                ),
                "precision": float(
                    precision_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
                "f1_score": float(
                    f1_score(
                        y_true,
                        predictions,
                        zero_division=0,
                    )
                ),
                "predicted_positive_patients": int(
                    predictions.sum()
                ),
                "false_positive_patients": int(
                    (
                        (y_true == 0)
                        & (predictions == 1)
                    ).sum()
                ),
                "false_negative_patients": int(
                    (
                        (y_true == 1)
                        & (predictions == 0)
                    ).sum()
                ),
            }
        )


def main() -> None:
    """Run optimization."""

    optimizer = PatientDecisionOptimizer()

    optimizer.run()


if __name__ == "__main__":
    main()