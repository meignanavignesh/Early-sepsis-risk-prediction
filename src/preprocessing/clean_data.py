"""Clinical preprocessing pipeline for the merged sepsis dataset."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


INPUT_DATASET_PATH = Path("data/interim/merged_dataset.csv")
CLEANED_DATASET_PATH = Path("data/processed/cleaned_dataset.csv")
PREPROCESSING_REPORT_PATH = Path("reports/preprocessing_report.json")

IDENTIFIER_COLUMNS = ("Patient_ID",)
TARGET_COLUMNS = ("SepsisLabel",)
METADATA_COLUMNS = ("Dataset",)

CLINICAL_COLUMNS = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "EtCO2",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
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
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
)

OUTPUT_COLUMNS = (
    *CLINICAL_COLUMNS,
    *TARGET_COLUMNS,
    *IDENTIFIER_COLUMNS,
    *METADATA_COLUMNS,
)

NUMERIC_COLUMNS = (*CLINICAL_COLUMNS, *TARGET_COLUMNS)

INVALID_VALUE_RULES = {
    "Temp": lambda series: (series < 25) | (series > 45),
    "FiO2": lambda series: series < 0,
}


class ClinicalDataPreprocessor:
    """Apply conservative clinical cleaning without imputation or features."""

    def __init__(
        self,
        input_path: str | Path = INPUT_DATASET_PATH,
        output_path: str | Path = CLEANED_DATASET_PATH,
        report_path: str | Path = PREPROCESSING_REPORT_PATH,
    ) -> None:
        """Initialize preprocessing input and output paths."""
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report_path = Path(report_path)

    def run(self) -> dict[str, Any]:
        """Run the preprocessing pipeline and save dataset plus report."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input dataset not found: {self.input_path}")

        dataframe = pd.read_csv(self.input_path, na_values=["", "NA", "NaN"])
        original_shape = dataframe.shape

        self._validate_required_columns(dataframe)
        dataframe = dataframe.loc[:, OUTPUT_COLUMNS].copy()
        dataframe = self._convert_numeric_columns(dataframe)
        invalid_values_corrected = self._correct_invalid_values(dataframe)
        self._validate_cleaned_dataframe(dataframe)

        dataframe = dataframe.sort_values(
            by=["Patient_ID", "ICULOS"],
            kind="mergesort",
        ).reset_index(drop=True)

        report = self._create_report(
            dataframe=dataframe,
            original_shape=original_shape,
            invalid_values_corrected=invalid_values_corrected,
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(self.output_path, index=False, encoding="utf-8")

        with self.report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=4)

        return report

    @staticmethod
    def _validate_required_columns(dataframe: pd.DataFrame) -> None:
        """Validate that all required output columns exist."""
        missing_columns = [
            column for column in OUTPUT_COLUMNS if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    @staticmethod
    def _convert_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
        """Convert clinical and target columns to numeric values."""
        for column in NUMERIC_COLUMNS:
            dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

        return dataframe

    @staticmethod
    def _correct_invalid_values(dataframe: pd.DataFrame) -> dict[str, int]:
        """Convert clearly invalid measurements to NaN and count fixes."""
        corrected_counts = {}

        for column, rule in INVALID_VALUE_RULES.items():
            invalid_mask = rule(dataframe[column])
            corrected_count = int(invalid_mask.sum())
            dataframe.loc[invalid_mask, column] = pd.NA
            corrected_counts[column] = corrected_count

        return corrected_counts

    @staticmethod
    def _validate_cleaned_dataframe(dataframe: pd.DataFrame) -> None:
        """Validate essential cleaned dataset constraints."""
        sepsis_values = set(dataframe["SepsisLabel"].dropna().unique())
        invalid_labels = sorted(sepsis_values - {0, 1})

        if invalid_labels:
            raise ValueError(
                f"SepsisLabel contains invalid values: {invalid_labels}"
            )

        missing_patient_ids = int(dataframe["Patient_ID"].isna().sum())
        if missing_patient_ids:
            raise ValueError(
                f"Patient_ID contains missing values: {missing_patient_ids}"
            )

        if not pd.api.types.is_numeric_dtype(dataframe["ICULOS"]):
            raise TypeError("ICULOS must be numeric.")

    @staticmethod
    def _create_report(
        dataframe: pd.DataFrame,
        original_shape: tuple[int, int],
        invalid_values_corrected: dict[str, int],
    ) -> dict[str, Any]:
        """Create the preprocessing report."""
        patient_labels = dataframe.groupby("Patient_ID")["SepsisLabel"].max()
        missing_counts = dataframe.isna().sum().astype(int)
        missing_percentages = dataframe.isna().mean().mul(100).round(4)

        return {
            "original_row_count": original_shape[0],
            "original_column_count": original_shape[1],
            "final_row_count": len(dataframe),
            "final_column_count": len(dataframe.columns),
            "invalid_values_corrected": invalid_values_corrected,
            "missing_value_counts": missing_counts.to_dict(),
            "missing_value_percentages": missing_percentages.to_dict(),
            "number_of_patients": int(dataframe["Patient_ID"].nunique()),
            "number_of_sepsis_patients": int((patient_labels == 1).sum()),
            "number_of_non_sepsis_patients": int((patient_labels == 0).sum()),
            "SepsisLabel_distribution": {
                str(label): int(count)
                for label, count in dataframe["SepsisLabel"]
                .value_counts()
                .sort_index()
                .items()
            },
        }


if __name__ == "__main__":
    preprocessor = ClinicalDataPreprocessor()
    preprocessing_report = preprocessor.run()

    total_corrected = sum(
        preprocessing_report["invalid_values_corrected"].values()
    )
    missing_summary = sorted(
        preprocessing_report["missing_value_percentages"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    print("CLINICAL PREPROCESSING COMPLETE")
    print(
        "Original shape: "
        f"({preprocessing_report['original_row_count']}, "
        f"{preprocessing_report['original_column_count']})"
    )
    print(
        "Cleaned shape: "
        f"({preprocessing_report['final_row_count']}, "
        f"{preprocessing_report['final_column_count']})"
    )
    print(f"Corrected invalid values: {total_corrected}")
    print(f"Number of patients: {preprocessing_report['number_of_patients']}")
    print(
        "Sepsis-positive patients: "
        f"{preprocessing_report['number_of_sepsis_patients']}"
    )
    print("Top missing-value percentages:")
    for column, percentage in missing_summary:
        print(f"  {column}: {percentage}%")
