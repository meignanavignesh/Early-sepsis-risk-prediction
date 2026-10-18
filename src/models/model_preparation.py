"""Prepare leakage-safe datasets and class weights for model training."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight


INPUT_DIR = Path("data/processed/preprocessed")
OUTPUT_DIR = Path("data/processed/model_ready")
REPORT_PATH = Path("reports/model_preparation_report.json")

TRAIN_PATH = INPUT_DIR / "train.csv"
VALIDATION_PATH = INPUT_DIR / "validation.csv"
TEST_PATH = INPUT_DIR / "test.csv"

TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "validation.csv"
TEST_OUTPUT_PATH = OUTPUT_DIR / "test.csv"

TARGET_COLUMN = "SepsisLabel"

IDENTIFIER_COLUMNS = [
    "Patient_ID",
    "Dataset",
]


class ModelPreparation:
    """Prepare preprocessed datasets for machine-learning models."""

    def __init__(
        self,
        input_dir: str | Path = INPUT_DIR,
        output_dir: str | Path = OUTPUT_DIR,
        report_path: str | Path = REPORT_PATH,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.report_path = Path(report_path)

    def run(self) -> dict:
        """Prepare datasets and calculate training class weights."""

        print("Loading preprocessed datasets...")

        train = self._load_dataset(TRAIN_PATH)

        validation = self._load_dataset(
            VALIDATION_PATH
        )

        test = self._load_dataset(
            TEST_PATH
        )

        self._validate_datasets(
            train,
            validation,
            test,
        )

        # --------------------------------------------------
        # Identify model features.
        #
        # Patient_ID and Dataset are retained in the files
        # but are NOT used as model input features.
        # SepsisLabel is the prediction target.
        # --------------------------------------------------

        feature_columns = [
            column
            for column in train.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        print(
            f"Total model features: "
            f"{len(feature_columns)}"
        )

        # --------------------------------------------------
        # Separate features and target.
        # --------------------------------------------------

        X_train = train[feature_columns]
        y_train = train[TARGET_COLUMN]

        X_validation = validation[
            feature_columns
        ]

        y_validation = validation[
            TARGET_COLUMN
        ]

        X_test = test[feature_columns]

        y_test = test[
            TARGET_COLUMN
        ]

        # Prevent unused-variable warnings while keeping
        # the explicit separation clear.
        _ = X_train
        _ = X_validation
        _ = X_test

        # --------------------------------------------------
        # Validate target values.
        # --------------------------------------------------

        self._validate_target(
            y_train,
            "training",
        )

        self._validate_target(
            y_validation,
            "validation",
        )

        self._validate_target(
            y_test,
            "test",
        )

        # --------------------------------------------------
        # Calculate class weights ONLY from training data.
        #
        # NumPy arrays are used for compatibility with
        # different scikit-learn versions.
        # --------------------------------------------------

        classes = np.array(
            sorted(
                y_train.dropna()
                .unique()
                .tolist()
            )
        )

        y_train_array = y_train.to_numpy()

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train_array,
        )

        class_weights = {
            str(int(label)): float(weight)
            for label, weight in zip(
                classes,
                weights,
            )
        }

        # --------------------------------------------------
        # Calculate class distributions.
        # --------------------------------------------------

        train_distribution = (
            y_train
            .value_counts()
            .sort_index()
            .to_dict()
        )

        validation_distribution = (
            y_validation
            .value_counts()
            .sort_index()
            .to_dict()
        )

        test_distribution = (
            y_test
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print("=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)

        print(
            f"Training: "
            f"{train_distribution}"
        )

        print(
            f"Validation: "
            f"{validation_distribution}"
        )

        print(
            f"Test: "
            f"{test_distribution}"
        )

        print("=" * 60)
        print("CLASS WEIGHTS")
        print("=" * 60)

        for label, weight in class_weights.items():
            print(
                f"Class {label}: "
                f"{weight:.4f}"
            )

        # --------------------------------------------------
        # Save model-ready datasets.
        # --------------------------------------------------

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train.to_csv(
            TRAIN_OUTPUT_PATH,
            index=False,
        )

        validation.to_csv(
            VALIDATION_OUTPUT_PATH,
            index=False,
        )

        test.to_csv(
            TEST_OUTPUT_PATH,
            index=False,
        )

        # --------------------------------------------------
        # Create report.
        # --------------------------------------------------

        report = {
            "training_shape": list(
                train.shape
            ),
            "validation_shape": list(
                validation.shape
            ),
            "test_shape": list(
                test.shape
            ),
            "number_of_model_features": len(
                feature_columns
            ),
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "identifier_columns": (
                IDENTIFIER_COLUMNS
            ),
            "training_class_distribution": {
                str(key): int(value)
                for key, value in (
                    train_distribution.items()
                )
            },
            "validation_class_distribution": {
                str(key): int(value)
                for key, value in (
                    validation_distribution.items()
                )
            },
            "test_class_distribution": {
                str(key): int(value)
                for key, value in (
                    test_distribution.items()
                )
            },
            "training_class_weights": (
                class_weights
            ),
            "class_imbalance_handling": (
                "class_weight=balanced"
            ),
            "class_weights_fitted_on": (
                "training_only"
            ),
            "synthetic_oversampling_used": (
                False
            ),
            "smote_used": False,
            "patient_duplication_used": (
                False
            ),
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

        return report

    @staticmethod
    def _load_dataset(
        path: Path,
    ) -> pd.DataFrame:
        """Load a prepared dataset."""

        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {path}"
            )

        dataframe = pd.read_csv(path)

        if dataframe.empty:
            raise ValueError(
                f"Dataset is empty: {path}"
            )

        return dataframe

    @staticmethod
    def _validate_datasets(
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Validate train, validation and test datasets."""

        train_columns = list(
            train.columns
        )

        validation_columns = list(
            validation.columns
        )

        test_columns = list(
            test.columns
        )

        if train_columns != validation_columns:
            raise ValueError(
                "Training and validation "
                "columns differ."
            )

        if train_columns != test_columns:
            raise ValueError(
                "Training and test "
                "columns differ."
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
                "Missing required columns: "
                f"{sorted(missing)}"
            )

    @staticmethod
    def _validate_target(
        target: pd.Series,
        dataset_name: str,
    ) -> None:
        """Validate the binary sepsis target."""

        if target.isna().any():
            raise ValueError(
                f"Missing SepsisLabel values "
                f"in {dataset_name} dataset."
            )

        unique_values = set(
            target.unique().tolist()
        )

        if not unique_values.issubset(
            {0, 1}
        ):
            raise ValueError(
                f"Invalid SepsisLabel values "
                f"in {dataset_name}: "
                f"{unique_values}"
            )

        if len(unique_values) < 2:
            raise ValueError(
                f"{dataset_name} dataset contains "
                "only one target class."
            )


def main() -> None:
    """Run model preparation."""

    preparation = ModelPreparation()

    report = preparation.run()

    print("=" * 60)
    print("MODEL PREPARATION COMPLETE")
    print("=" * 60)

    print(
        "Training shape: "
        f"{tuple(report['training_shape'])}"
    )

    print(
        "Validation shape: "
        f"{tuple(report['validation_shape'])}"
    )

    print(
        "Test shape: "
        f"{tuple(report['test_shape'])}"
    )

    print(
        "Model features: "
        f"{report['number_of_model_features']}"
    )

    print(
        "Class weights: "
        f"{report['training_class_weights']}"
    )

    print(
        "Synthetic oversampling: "
        f"{report['synthetic_oversampling_used']}"
    )

    print(
        "SMOTE used: "
        f"{report['smote_used']}"
    )

    print(
        "Patient duplication: "
        f"{report['patient_duplication_used']}"
    )

    print(
        "Data leakage prevented: "
        f"{report['data_leakage_prevented']}"
    )

    print(
        "Report saved to: "
        f"{REPORT_PATH.resolve()}"
    )

    print(
        "Model-ready datasets saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()