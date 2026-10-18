"""Data quality analysis for the merged sepsis dataset."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_PATH = Path("data/interim/merged_dataset.csv")
REPORT_PATH = Path("reports/data_quality_report.json")
CHUNK_SIZE = 100_000

CATEGORICAL_COLUMNS = ("Gender", "Unit1", "Unit2", "SepsisLabel", "Dataset")
IDENTIFIER_COLUMNS = ("Patient_ID",)
NON_CLINICAL_COLUMNS = (*IDENTIFIER_COLUMNS, "Dataset")

SUSPICIOUS_RULES = {
    "HR": ("negative_heart_rate", lambda series: series < 0),
    "O2Sat": (
        "impossible_oxygen_saturation",
        lambda series: (series < 0) | (series > 100),
    ),
    "Temp": (
        "impossible_temperature",
        lambda series: (series < 25) | (series > 45),
    ),
    "SBP": ("negative_systolic_blood_pressure", lambda series: series < 0),
    "MAP": ("negative_mean_arterial_pressure", lambda series: series < 0),
    "DBP": ("negative_diastolic_blood_pressure", lambda series: series < 0),
}

NON_NEGATIVE_LAB_COLUMNS = (
    "EtCO2",
    "HCO3",
    "FiO2",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Bilirubin_total",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
)


class DataQualityAnalyzer:
    """Analyze structure and basic quality of the merged dataset."""

    def __init__(
        self,
        dataset_path: str | Path = DATASET_PATH,
        report_path: str | Path = REPORT_PATH,
    ) -> None:
        """Initialize the analyzer with dataset and report paths."""
        self.dataset_path = Path(dataset_path)
        self.report_path = Path(report_path)

    def analyze(self) -> dict[str, Any]:
        """Run the data quality analysis and save a JSON report."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        dataframe = pd.read_csv(self.dataset_path)
        clinical_columns = [
            column
            for column in dataframe.columns
            if column not in NON_CLINICAL_COLUMNS
        ]
        numeric_clinical_columns = dataframe[clinical_columns].select_dtypes(
            include="number"
        ).columns

        report = {
            "dataset_shape": list(dataframe.shape),
            "columns_and_dtypes": self._get_dtypes(dataframe),
            "missing_value_count": self._get_missing_counts(
                dataframe,
                clinical_columns,
            ),
            "missing_value_percentage": self._get_missing_percentages(
                dataframe,
                clinical_columns,
            ),
            "unique_value_count": self._get_unique_counts(dataframe),
            "numerical_min_max": self._get_min_max(
                dataframe,
                numeric_clinical_columns,
            ),
            "categorical_unique_values": self._get_categorical_values(
                dataframe,
            ),
            "patient_count": int(dataframe["Patient_ID"].nunique()),
            "observations_per_patient": self._get_observations_per_patient(
                dataframe,
            ),
            "sepsis_label_distribution": self._get_value_counts(
                dataframe["SepsisLabel"],
            ),
            "sepsis_positive_patient_count": self._get_positive_patient_count(
                dataframe,
            ),
            "non_sepsis_patient_count": self._get_non_sepsis_patient_count(
                dataframe,
            ),
            "suspicious_values": self._get_suspicious_values(dataframe),
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=4)

        return report

    def print_summary(self, report: dict[str, Any]) -> None:
        """Print a concise data quality summary."""
        print("DATA QUALITY ANALYSIS COMPLETE")
        print(f"Dataset shape: {tuple(report['dataset_shape'])}")
        print(f"Patient count: {report['patient_count']}")
        print(
            "Sepsis-positive patients: "
            f"{report['sepsis_positive_patient_count']}"
        )
        print(f"Non-sepsis patients: {report['non_sepsis_patient_count']}")
        print(
            "SepsisLabel distribution: "
            f"{report['sepsis_label_distribution']}"
        )
        print(
            "Observations per patient: "
            f"{report['observations_per_patient']}"
        )
        print("Suspicious values:")
        for check_name, result in report["suspicious_values"].items():
            print(f"  {check_name}: {result['count']}")
        print(f"Report saved to: {self.report_path.resolve()}")

    @staticmethod
    def _get_dtypes(dataframe: pd.DataFrame) -> dict[str, str]:
        """Return column data types as strings."""
        return {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        }

    @staticmethod
    def _get_missing_counts(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, int]:
        """Return missing-value counts for selected columns."""
        return dataframe[columns].isna().sum().astype(int).to_dict()

    @staticmethod
    def _get_missing_percentages(
        dataframe: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, float]:
        """Return missing-value percentages for selected columns."""
        return (
            dataframe[columns]
            .isna()
            .mean()
            .mul(100)
            .round(4)
            .to_dict()
        )

    @staticmethod
    def _get_unique_counts(dataframe: pd.DataFrame) -> dict[str, int]:
        """Return unique-value counts for each column."""
        return dataframe.nunique(dropna=True).astype(int).to_dict()

    @staticmethod
    def _get_min_max(
        dataframe: pd.DataFrame,
        columns: pd.Index,
    ) -> dict[str, dict[str, float | None]]:
        """Return minimum and maximum values for numerical columns."""
        min_max_values: dict[str, dict[str, float | None]] = {}

        for column in columns:
            column_min = dataframe[column].min()
            column_max = dataframe[column].max()
            min_max_values[column] = {
                "min": None if pd.isna(column_min) else float(column_min),
                "max": None if pd.isna(column_max) else float(column_max),
            }

        return min_max_values

    @staticmethod
    def _get_categorical_values(
        dataframe: pd.DataFrame,
    ) -> dict[str, list[Any]]:
        """Return unique values for known categorical columns."""
        categorical_values = {}

        for column in CATEGORICAL_COLUMNS:
            values = dataframe[column].dropna().unique().tolist()
            categorical_values[column] = sorted(values)

        return categorical_values

    @staticmethod
    def _get_observations_per_patient(
        dataframe: pd.DataFrame,
    ) -> dict[str, float]:
        """Return summary statistics for observations per patient."""
        observations = dataframe.groupby("Patient_ID").size()

        return {
            "minimum": int(observations.min()),
            "maximum": int(observations.max()),
            "mean": round(float(observations.mean()), 4),
            "median": float(observations.median()),
        }

    @staticmethod
    def _get_value_counts(series: pd.Series) -> dict[str, int]:
        """Return value counts with JSON-safe string keys."""
        return {
            str(key): int(value)
            for key, value in series.value_counts(dropna=False).items()
        }

    @staticmethod
    def _get_positive_patient_count(dataframe: pd.DataFrame) -> int:
        """Return patients with at least one positive sepsis label."""
        patient_labels = dataframe.groupby("Patient_ID")["SepsisLabel"].max()
        return int((patient_labels == 1).sum())

    @staticmethod
    def _get_non_sepsis_patient_count(dataframe: pd.DataFrame) -> int:
        """Return patients with no positive sepsis label."""
        patient_labels = dataframe.groupby("Patient_ID")["SepsisLabel"].max()
        return int((patient_labels == 0).sum())

    @staticmethod
    def _get_suspicious_values(
        dataframe: pd.DataFrame,
    ) -> dict[str, dict[str, int]]:
        """Detect impossible or suspicious clinical values."""
        suspicious_values = {}

        for column, (check_name, rule) in SUSPICIOUS_RULES.items():
            if column in dataframe.columns:
                suspicious_values[check_name] = {
                    "count": int(rule(dataframe[column]).sum())
                }

        for column in NON_NEGATIVE_LAB_COLUMNS:
            if column in dataframe.columns:
                check_name = f"negative_{column}"
                suspicious_values[check_name] = {
                    "count": int((dataframe[column] < 0).sum())
                }

        return suspicious_values


if __name__ == "__main__":
    analyzer = DataQualityAnalyzer()
    quality_report = analyzer.analyze()
    analyzer.print_summary(quality_report)
