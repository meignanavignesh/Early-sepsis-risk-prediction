"""Leakage-safe missing-value preprocessing for sepsis prediction."""

import json
from pathlib import Path

import pandas as pd


SPLIT_DIR = Path("data/processed/splits")
OUTPUT_DIR = Path("data/processed/preprocessed")
REPORT_PATH = Path("reports/missing_value_report.json")

TRAIN_PATH = SPLIT_DIR / "train.csv"
VALIDATION_PATH = SPLIT_DIR / "validation.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

TRAIN_OUTPUT_PATH = OUTPUT_DIR / "train.csv"
VALIDATION_OUTPUT_PATH = OUTPUT_DIR / "validation.csv"
TEST_OUTPUT_PATH = OUTPUT_DIR / "test.csv"

IDENTIFIER_COLUMNS = [
    "Patient_ID",
    "Dataset",
]

TARGET_COLUMN = "SepsisLabel"


class MissingValueHandler:
    """Handle missing values using statistics learned from training data."""

    def __init__(
        self,
        train_path: str | Path = TRAIN_PATH,
        validation_path: str | Path = VALIDATION_PATH,
        test_path: str | Path = TEST_PATH,
        output_dir: str | Path = OUTPUT_DIR,
        report_path: str | Path = REPORT_PATH,
    ) -> None:
        self.train_path = Path(train_path)
        self.validation_path = Path(validation_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.report_path = Path(report_path)

        self.medians: dict[str, float] = {}
        self.missing_indicator_columns: list[str] = []

    def run(self) -> dict:
        """Run leakage-safe missing-value preprocessing."""

        print("Loading training dataset...")
        train = self._load_dataset(self.train_path)

        print("Loading validation dataset...")
        validation = self._load_dataset(
            self.validation_path
        )

        print("Loading test dataset...")
        test = self._load_dataset(self.test_path)

        self._validate_columns(
            train,
            validation,
            test,
        )

        feature_columns = [
            column
            for column in train.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        numeric_columns = [
            column
            for column in feature_columns
            if pd.api.types.is_numeric_dtype(
                train[column]
            )
        ]

        categorical_columns = [
            column
            for column in feature_columns
            if column not in numeric_columns
        ]

        print(
            f"Model features: {len(feature_columns)}"
        )

        print(
            f"Numeric features: {len(numeric_columns)}"
        )

        print(
            f"Categorical features: "
            f"{len(categorical_columns)}"
        )

        # Determine missing-indicator columns
        # using TRAINING DATA ONLY.
        self.missing_indicator_columns = []

        for column in feature_columns:
            if train[column].isna().any():
                indicator_name = (
                    f"{column}_was_missing"
                )

                self.missing_indicator_columns.append(
                    indicator_name
                )

        print(
            "Missing indicators: "
            f"{len(self.missing_indicator_columns)}"
        )

        # Calculate medians ONLY from training data.
        self.medians = {}

        for column in numeric_columns:
            median = train[column].median()

            if pd.isna(median):
                raise ValueError(
                    "Training feature has no valid "
                    f"values: {column}"
                )

            self.medians[column] = float(median)

        # Add the SAME indicators to all datasets.
        train_processed = (
            self._add_training_defined_indicators(
                train
            )
        )

        validation_processed = (
            self._add_training_defined_indicators(
                validation
            )
        )

        test_processed = (
            self._add_training_defined_indicators(
                test
            )
        )

        # Apply training-derived medians.
        train_processed = self._impute_numeric(
            train_processed,
            numeric_columns,
        )

        validation_processed = (
            self._impute_numeric(
                validation_processed,
                numeric_columns,
            )
        )

        test_processed = self._impute_numeric(
            test_processed,
            numeric_columns,
        )

        # Make sure all datasets have identical columns.
        (
            train_processed,
            validation_processed,
            test_processed,
        ) = self._align_columns(
            train_processed,
            validation_processed,
            test_processed,
        )

        # Validate numeric missing values.
        self._validate_no_numeric_missing_values(
            train_processed,
            numeric_columns,
            "training",
        )

        self._validate_no_numeric_missing_values(
            validation_processed,
            numeric_columns,
            "validation",
        )

        self._validate_no_numeric_missing_values(
            test_processed,
            numeric_columns,
            "test",
        )

        # Final column consistency check.
        columns_are_identical = (
            list(train_processed.columns)
            == list(validation_processed.columns)
            == list(test_processed.columns)
        )

        if not columns_are_identical:
            raise ValueError(
                "Train, validation and test columns "
                "are not identical."
            )

        # Save processed datasets.
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_processed.to_csv(
            TRAIN_OUTPUT_PATH,
            index=False,
        )

        validation_processed.to_csv(
            VALIDATION_OUTPUT_PATH,
            index=False,
        )

        test_processed.to_csv(
            TEST_OUTPUT_PATH,
            index=False,
        )

        report = self._create_report(
            train=train,
            validation=validation,
            test=test,
            train_processed=train_processed,
            validation_processed=validation_processed,
            test_processed=test_processed,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )

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
        """Load a dataset."""

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
    def _validate_columns(
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> None:
        """Validate that input split columns match."""

        train_columns = list(train.columns)
        validation_columns = list(
            validation.columns
        )
        test_columns = list(test.columns)

        if train_columns != validation_columns:
            raise ValueError(
                "Training and validation input columns differ."
            )

        if train_columns != test_columns:
            raise ValueError(
                "Training and test input columns differ."
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
                f"Missing required columns: {sorted(missing)}"
            )

    def _add_training_defined_indicators(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add missing indicators determined from training data.

        Every dataset receives exactly the same
        missing-indicator columns.
        """

        result = dataframe.copy()

        for indicator_name in (
            self.missing_indicator_columns
        ):
            original_column = (
                indicator_name.replace(
                    "_was_missing",
                    "",
                )
            )

            if original_column not in result.columns:
                raise ValueError(
                    "Original feature missing while "
                    "creating indicator: "
                    f"{original_column}"
                )

            result[indicator_name] = (
                result[original_column]
                .isna()
                .astype("int8")
            )

        return result

    def _impute_numeric(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
    ) -> pd.DataFrame:
        """Impute numeric features using training medians."""

        result = dataframe.copy()

        for column in numeric_columns:
            result[column] = result[column].fillna(
                self.medians[column]
            )

        return result

    @staticmethod
    def _align_columns(
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """Force all datasets to use the same column order."""

        expected_columns = list(train.columns)

        validation_missing = [
            column
            for column in expected_columns
            if column not in validation.columns
        ]

        test_missing = [
            column
            for column in expected_columns
            if column not in test.columns
        ]

        if validation_missing:
            raise ValueError(
                "Validation is missing columns: "
                f"{validation_missing}"
            )

        if test_missing:
            raise ValueError(
                "Test is missing columns: "
                f"{test_missing}"
            )

        train = train[expected_columns]
        validation = validation[expected_columns]
        test = test[expected_columns]

        return (
            train,
            validation,
            test,
        )

    @staticmethod
    def _validate_no_numeric_missing_values(
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
        dataset_name: str,
    ) -> None:
        """Ensure numeric model features have no missing values."""

        remaining = [
            column
            for column in numeric_columns
            if dataframe[column].isna().any()
        ]

        if remaining:
            raise ValueError(
                f"Missing numeric values remain in "
                f"{dataset_name}: {remaining}"
            )

    def _create_report(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        test: pd.DataFrame,
        train_processed: pd.DataFrame,
        validation_processed: pd.DataFrame,
        test_processed: pd.DataFrame,
        numeric_columns: list[str],
        categorical_columns: list[str],
    ) -> dict:
        """Create the preprocessing report."""

        def total_missing(
            dataframe: pd.DataFrame,
        ) -> int:
            return int(
                dataframe.isna().sum().sum()
            )

        def numeric_missing(
            dataframe: pd.DataFrame,
        ) -> int:
            return int(
                dataframe[numeric_columns]
                .isna()
                .sum()
                .sum()
            )

        columns_identical = (
            list(train_processed.columns)
            == list(validation_processed.columns)
            == list(test_processed.columns)
        )

        return {
            "input_shapes": {
                "train": list(train.shape),
                "validation": list(
                    validation.shape
                ),
                "test": list(test.shape),
            },
            "output_shapes": {
                "train": list(
                    train_processed.shape
                ),
                "validation": list(
                    validation_processed.shape
                ),
                "test": list(
                    test_processed.shape
                ),
            },
            "columns_identical_across_splits": (
                columns_identical
            ),
            "numeric_feature_count": len(
                numeric_columns
            ),
            "categorical_feature_count": len(
                categorical_columns
            ),
            "missing_indicator_count": len(
                self.missing_indicator_columns
            ),
            "missing_values_before": {
                "train": total_missing(train),
                "validation": total_missing(
                    validation
                ),
                "test": total_missing(test),
            },
            "missing_numeric_values_after": {
                "train": numeric_missing(
                    train_processed
                ),
                "validation": numeric_missing(
                    validation_processed
                ),
                "test": numeric_missing(
                    test_processed
                ),
            },
            "imputation_method": (
                "Training-set median"
            ),
            "fit_statistics_source": "train_only",
            "data_leakage_prevented": True,
            "target_used_for_imputation": False,
        }


def main() -> None:
    """Run missing-value preprocessing."""

    processor = MissingValueHandler()

    report = processor.run()

    print("=" * 60)
    print("MISSING-VALUE PREPROCESSING COMPLETE")
    print("=" * 60)

    print(
        "Training shape: "
        f"{tuple(report['output_shapes']['train'])}"
    )

    print(
        "Validation shape: "
        f"{tuple(report['output_shapes']['validation'])}"
    )

    print(
        "Test shape: "
        f"{tuple(report['output_shapes']['test'])}"
    )

    print(
        "Numeric features: "
        f"{report['numeric_feature_count']}"
    )

    print(
        "Categorical features: "
        f"{report['categorical_feature_count']}"
    )

    print(
        "Missing indicators: "
        f"{report['missing_indicator_count']}"
    )

    print(
        "Columns identical across splits: "
        f"{report['columns_identical_across_splits']}"
    )

    print(
        "Imputation method: "
        f"{report['imputation_method']}"
    )

    print(
        "Statistics fitted using: "
        f"{report['fit_statistics_source']}"
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
        "Processed datasets saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()