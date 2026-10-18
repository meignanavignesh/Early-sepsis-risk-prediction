"""DataFrame validation utilities for PhysioNet sepsis patient records."""

from collections.abc import Sequence

import pandas as pd


EXPECTED_CLINICAL_COLUMNS = (
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
    "SepsisLabel",
)


class DataValidator:
    """Validate the structure and basic integrity of patient DataFrames."""

    def __init__(
        self,
        expected_columns: Sequence[str] = EXPECTED_CLINICAL_COLUMNS,
    ) -> None:
        """Initialize the validator with the required patient columns."""
        self.expected_columns = tuple(expected_columns)

    def validate_dataframe(self, dataframe: pd.DataFrame) -> bool:
        """Run all validation checks on a patient DataFrame."""
        self.validate_not_empty(dataframe)
        self.validate_no_duplicate_columns(dataframe)
        self.validate_columns(dataframe)
        self.validate_sepsis_label(dataframe)

        return True

    def validate_columns(self, dataframe: pd.DataFrame) -> bool:
        """Validate that all expected columns are present."""
        missing_columns = [
            column
            for column in self.expected_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            missing = ", ".join(missing_columns)
            raise ValueError(f"Missing required columns: {missing}")

        return True

    def validate_sepsis_label(self, dataframe: pd.DataFrame) -> bool:
        """Validate that SepsisLabel exists and contains only 0 or 1."""
        if "SepsisLabel" not in dataframe.columns:
            raise ValueError("Missing required column: SepsisLabel")

        valid_values = {0, 1}
        label_values = set(dataframe["SepsisLabel"].dropna().unique())
        invalid_values = sorted(label_values - valid_values)

        if invalid_values:
            raise ValueError(
                f"SepsisLabel contains invalid values: {invalid_values}"
            )

        return True

    def validate_not_empty(self, dataframe: pd.DataFrame) -> bool:
        """Validate that the DataFrame contains at least one row."""
        if dataframe.empty:
            raise ValueError("DataFrame is empty.")

        if len(dataframe) < 1:
            raise ValueError("DataFrame must contain at least one row.")

        return True

    def validate_no_duplicate_columns(self, dataframe: pd.DataFrame) -> bool:
        """Validate that the DataFrame has no duplicate column names."""
        duplicated_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        if duplicated_columns:
            duplicates = ", ".join(duplicated_columns)
            raise ValueError(f"Duplicate column names found: {duplicates}")

        return True
