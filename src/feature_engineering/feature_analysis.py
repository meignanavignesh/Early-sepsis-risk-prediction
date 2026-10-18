"""Analyze engineered features before machine-learning preparation."""

import json
from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/engineered_dataset.csv")
REPORT_PATH = Path("reports/feature_analysis_report.json")


IDENTIFIER_COLUMNS = [
    "Patient_ID",
    "Dataset",
]

TARGET_COLUMN = "SepsisLabel"


class FeatureAnalyzer:
    """Analyze feature quality without modifying the dataset."""

    def __init__(
        self,
        input_path: str | Path = INPUT_PATH,
        report_path: str | Path = REPORT_PATH,
    ) -> None:
        self.input_path = Path(input_path)
        self.report_path = Path(report_path)

    def run(self) -> dict:
        """Run the complete feature analysis."""

        dataframe = self._load_data()

        report = {
            "dataset_shape": list(dataframe.shape),
            "number_of_patients": int(
                dataframe["Patient_ID"].nunique()
            ),
            "target_distribution": self._target_distribution(
                dataframe
            ),
            "data_types": self._data_types(dataframe),
            "missing_values": self._missing_values(dataframe),
            "constant_features": self._constant_features(
                dataframe
            ),
            "duplicate_columns": self._duplicate_columns(
                dataframe
            ),
            "feature_summary": self._feature_summary(
                dataframe
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

        return report

    def _load_data(self) -> pd.DataFrame:
        """Load the engineered dataset."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.input_path}"
            )

        dataframe = pd.read_csv(self.input_path)

        if dataframe.empty:
            raise ValueError(
                "Engineered dataset is empty."
            )

        return dataframe

    @staticmethod
    def _target_distribution(
        dataframe: pd.DataFrame,
    ) -> dict:
        """Return SepsisLabel distribution."""

        counts = (
            dataframe[TARGET_COLUMN]
            .value_counts(dropna=False)
            .sort_index()
        )

        return {
            str(index): int(value)
            for index, value in counts.items()
        }

    @staticmethod
    def _data_types(
        dataframe: pd.DataFrame,
    ) -> dict:
        """Return data type for every column."""

        return {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        }

    @staticmethod
    def _missing_values(
        dataframe: pd.DataFrame,
    ) -> dict:
        """Calculate missing-value statistics."""

        missing_count = dataframe.isna().sum()

        total_rows = len(dataframe)

        result = {}

        for column in dataframe.columns:
            count = int(missing_count[column])

            result[column] = {
                "count": count,
                "percentage": round(
                    (count / total_rows) * 100,
                    4,
                ),
            }

        return result

    @staticmethod
    def _constant_features(
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """Find columns containing only one unique value."""

        constant_columns = []

        for column in dataframe.columns:
            if dataframe[column].nunique(
                dropna=False
            ) <= 1:
                constant_columns.append(column)

        return constant_columns

    @staticmethod
    def _duplicate_columns(
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """Find duplicate columns."""

        duplicate_columns = []

        columns = dataframe.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                first = columns[i]
                second = columns[j]

                if dataframe[first].equals(
                    dataframe[second]
                ):
                    duplicate_columns.append(
                        {
                            "column_1": first,
                            "column_2": second,
                        }
                    )

        return duplicate_columns

    @staticmethod
    def _feature_summary(
        dataframe: pd.DataFrame,
    ) -> dict:
        """Create numerical feature statistics."""

        excluded = set(
            IDENTIFIER_COLUMNS + [TARGET_COLUMN]
        )

        numerical_columns = dataframe.select_dtypes(
            include="number"
        ).columns

        summary = {}

        for column in numerical_columns:

            if column in excluded:
                continue

            series = dataframe[column]

            summary[column] = {
                "min": (
                    None
                    if series.dropna().empty
                    else float(series.min())
                ),
                "max": (
                    None
                    if series.dropna().empty
                    else float(series.max())
                ),
                "mean": (
                    None
                    if series.dropna().empty
                    else float(series.mean())
                ),
                "median": (
                    None
                    if series.dropna().empty
                    else float(series.median())
                ),
            }

        return summary


def main() -> None:
    """Run feature analysis."""

    analyzer = FeatureAnalyzer()

    report = analyzer.run()

    print("=" * 60)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 60)

    print(
        f"Dataset shape: "
        f"{tuple(report['dataset_shape'])}"
    )

    print(
        f"Patients: "
        f"{report['number_of_patients']}"
    )

    print(
        f"Target distribution: "
        f"{report['target_distribution']}"
    )

    print(
        "Constant features: "
        f"{len(report['constant_features'])}"
    )

    print(
        "Duplicate column pairs: "
        f"{len(report['duplicate_columns'])}"
    )

    print(
        f"Report saved to: "
        f"{REPORT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()