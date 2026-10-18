"""Optimize the XGBoost decision threshold using validation data only."""

import json
from pathlib import Path

import joblib
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


MODEL_PATH = Path("models/xgboost.pkl")

VALIDATION_PATH = Path(
    "data/processed/model_ready/validation.csv"
)

OUTPUT_REPORT = Path(
    "reports/threshold_optimization_report.json"
)

TARGET_COLUMN = "SepsisLabel"

IDENTIFIER_COLUMNS = [
    "Patient_ID",
    "Dataset",
]


# Thresholds to evaluate.
THRESHOLDS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


class ThresholdOptimizer:
    """Find a suitable classification threshold."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        validation_path: str | Path = VALIDATION_PATH,
        report_path: str | Path = OUTPUT_REPORT,
    ) -> None:
        self.model_path = Path(model_path)
        self.validation_path = Path(validation_path)
        self.report_path = Path(report_path)

    def run(self) -> dict:
        """Evaluate thresholds using validation data only."""

        print("Loading XGBoost model...")

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        model = joblib.load(
            self.model_path
        )

        print("Loading validation dataset...")

        if not self.validation_path.exists():
            raise FileNotFoundError(
                "Validation dataset not found: "
                f"{self.validation_path}"
            )

        validation = pd.read_csv(
            self.validation_path
        )

        if TARGET_COLUMN not in validation.columns:
            raise ValueError(
                f"Missing target column: "
                f"{TARGET_COLUMN}"
            )

        feature_columns = [
            column
            for column in validation.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        X_validation = validation[
            feature_columns
        ]

        y_validation = validation[
            TARGET_COLUMN
        ]

        print(
            f"Validation rows: "
            f"{len(validation)}"
        )

        print(
            f"Number of features: "
            f"{len(feature_columns)}"
        )

        print("=" * 65)
        print("GENERATING VALIDATION PROBABILITIES")
        print("=" * 65)

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        roc_auc = roc_auc_score(
            y_validation,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_validation,
            probabilities,
        )

        print(
            f"Validation ROC-AUC: "
            f"{roc_auc:.4f}"
        )

        print(
            f"Validation PR-AUC: "
            f"{pr_auc:.4f}"
        )

        results = []

        print("=" * 65)
        print("THRESHOLD EVALUATION")
        print("=" * 65)

        print(
            f"{'Threshold':<12}"
            f"{'Accuracy':<12}"
            f"{'Precision':<12}"
            f"{'Recall':<12}"
            f"{'F1':<12}"
        )

        for threshold in THRESHOLDS:

            predictions = (
                probabilities >= threshold
            ).astype(int)

            accuracy = accuracy_score(
                y_validation,
                predictions,
            )

            precision = precision_score(
                y_validation,
                predictions,
                zero_division=0,
            )

            recall = recall_score(
                y_validation,
                predictions,
                zero_division=0,
            )

            f1 = f1_score(
                y_validation,
                predictions,
                zero_division=0,
            )

            matrix = confusion_matrix(
                y_validation,
                predictions,
            )

            result = {
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
                "confusion_matrix": (
                    matrix.tolist()
                ),
            }

            results.append(result)

            print(
                f"{threshold:<12.2f}"
                f"{accuracy:<12.4f}"
                f"{precision:<12.4f}"
                f"{recall:<12.4f}"
                f"{f1:<12.4f}"
            )

        # --------------------------------------------------
        # Select threshold.
        #
        # For this project we prioritize F1 while requiring
        # recall to remain at least 50%.
        # --------------------------------------------------

        eligible_results = [
            result
            for result in results
            if result["recall"] >= 0.50
        ]

        if not eligible_results:
            eligible_results = results

        best_result = max(
            eligible_results,
            key=lambda result: (
                result["f1_score"],
                result["recall"],
            ),
        )

        selected_threshold = best_result[
            "threshold"
        ]

        print("=" * 65)
        print("SELECTED THRESHOLD")
        print("=" * 65)

        print(
            f"Threshold: "
            f"{selected_threshold:.2f}"
        )

        print(
            f"Accuracy: "
            f"{best_result['accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{best_result['precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{best_result['recall']:.4f}"
        )

        print(
            f"F1-score: "
            f"{best_result['f1_score']:.4f}"
        )

        report = {
            "model": "XGBoost",
            "validation_rows": int(
                len(validation)
            ),
            "number_of_features": len(
                feature_columns
            ),
            "validation_roc_auc": float(
                roc_auc
            ),
            "validation_pr_auc": float(
                pr_auc
            ),
            "thresholds_evaluated": THRESHOLDS,
            "selection_rule": (
                "Maximum F1-score among "
                "thresholds with recall >= 0.50"
            ),
            "selected_threshold": float(
                selected_threshold
            ),
            "selected_metrics": best_result,
            "all_threshold_results": results,
            "test_set_used": False,
            "data_leakage_prevented": True,
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

        print(
            "Report saved to: "
            f"{self.report_path.resolve()}"
        )

        return report


def main() -> None:
    """Run threshold optimization."""

    optimizer = ThresholdOptimizer()

    report = optimizer.run()

    print("=" * 65)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 65)

    print(
        "Selected threshold: "
        f"{report['selected_threshold']:.2f}"
    )


if __name__ == "__main__":
    main()