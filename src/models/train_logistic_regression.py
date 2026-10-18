"""Train and evaluate the Logistic Regression baseline."""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


INPUT_DIR = Path("data/processed/model_ready")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

TRAIN_PATH = INPUT_DIR / "train.csv"
VALIDATION_PATH = INPUT_DIR / "validation.csv"

MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"
REPORT_PATH = REPORT_DIR / "logistic_regression_report.json"

TARGET_COLUMN = "SepsisLabel"

IDENTIFIER_COLUMNS = [
    "Patient_ID",
    "Dataset",
]


class LogisticRegressionTrainer:
    """Train and evaluate the Logistic Regression baseline."""

    def __init__(
        self,
        train_path: str | Path = TRAIN_PATH,
        validation_path: str | Path = VALIDATION_PATH,
        model_path: str | Path = MODEL_PATH,
        report_path: str | Path = REPORT_PATH,
    ) -> None:
        self.train_path = Path(train_path)
        self.validation_path = Path(validation_path)
        self.model_path = Path(model_path)
        self.report_path = Path(report_path)

    def run(self) -> dict:
        """Train the model and evaluate it on validation data."""

        start_time = time.time()

        print("Loading training dataset...")
        train = pd.read_csv(self.train_path)

        print("Loading validation dataset...")
        validation = pd.read_csv(
            self.validation_path
        )

        self._validate_datasets(
            train,
            validation,
        )

        feature_columns = [
            column
            for column in train.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        print(
            f"Number of features: "
            f"{len(feature_columns)}"
        )

        X_train = train[feature_columns]
        y_train = train[TARGET_COLUMN]

        X_validation = validation[
            feature_columns
        ]
        y_validation = validation[
            TARGET_COLUMN
        ]

        print(
            f"Training rows: {len(X_train)}"
        )

        print(
            f"Validation rows: "
            f"{len(X_validation)}"
        )

        print("=" * 60)
        print("TRAINING LOGISTIC REGRESSION")
        print("=" * 60)

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
        )

        print("Training complete.")

        # --------------------------------------------------
        # Validation predictions
        # --------------------------------------------------

        predictions = model.predict(
            X_validation
        )

        probabilities = model.predict_proba(
            X_validation
        )[:, 1]

        metrics = self._calculate_metrics(
            y_validation,
            predictions,
            probabilities,
        )

        # --------------------------------------------------
        # Save model
        # --------------------------------------------------

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            self.model_path,
        )

        # --------------------------------------------------
        # Create report
        # --------------------------------------------------

        elapsed_time = time.time() - start_time

        report = {
            "model": "Logistic Regression",
            "training_rows": int(
                len(X_train)
            ),
            "validation_rows": int(
                len(X_validation)
            ),
            "number_of_features": len(
                feature_columns
            ),
            "features": feature_columns,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iterations": 1000,
            "validation_metrics": metrics,
            "training_time_seconds": round(
                elapsed_time,
                2,
            ),
            "model_path": str(
                self.model_path
            ),
            "data_leakage_prevented": True,
            "test_set_used": False,
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

        return report

    @staticmethod
    def _validate_datasets(
        train: pd.DataFrame,
        validation: pd.DataFrame,
    ) -> None:
        """Validate training and validation datasets."""

        if list(train.columns) != list(
            validation.columns
        ):
            raise ValueError(
                "Training and validation "
                "columns do not match."
            )

        required_columns = {
            "Patient_ID",
            "Dataset",
            "SepsisLabel",
        }

        missing = (
            required_columns
            - set(train.columns)
        )

        if missing:
            raise ValueError(
                f"Missing required columns: "
                f"{sorted(missing)}"
            )

        if train[TARGET_COLUMN].isna().any():
            raise ValueError(
                "Training contains missing "
                "SepsisLabel values."
            )

        if validation[TARGET_COLUMN].isna().any():
            raise ValueError(
                "Validation contains missing "
                "SepsisLabel values."
            )

    @staticmethod
    def _calculate_metrics(
        y_true: pd.Series,
        predictions,
        probabilities,
    ) -> dict:
        """Calculate classification metrics."""

        matrix = confusion_matrix(
            y_true,
            predictions,
        )

        report = classification_report(
            y_true,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        return {
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
            "roc_auc": float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            ),
            "pr_auc": float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            ),
            "confusion_matrix": (
                matrix.tolist()
            ),
            "classification_report": report,
        }


def main() -> None:
    """Run Logistic Regression training."""

    trainer = LogisticRegressionTrainer()

    report = trainer.run()

    metrics = report[
        "validation_metrics"
    ]

    print("=" * 60)
    print("LOGISTIC REGRESSION COMPLETE")
    print("=" * 60)

    print(
        f"Accuracy:  "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall:    "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1-score:  "
        f"{metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC:   "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:    "
        f"{metrics['pr_auc']:.4f}"
    )

    print(
        "Confusion matrix:"
    )

    print(
        metrics["confusion_matrix"]
    )

    print(
        f"Training time: "
        f"{report['training_time_seconds']} seconds"
    )

    print(
        f"Model saved to: "
        f"{Path(MODEL_PATH).resolve()}"
    )

    print(
        f"Report saved to: "
        f"{Path(REPORT_PATH).resolve()}"
    )


if __name__ == "__main__":
    main()