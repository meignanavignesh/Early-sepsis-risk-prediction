"""Leakage-safe temporal feature engineering for sepsis prediction."""

import json
from pathlib import Path
from typing import Any

import pandas as pd


INPUT_DATASET_PATH = Path("data/processed/cleaned_dataset.csv")
ENGINEERED_DATASET_PATH = Path("data/processed/engineered_dataset.csv")
FEATURE_ENGINEERING_REPORT_PATH = Path(
    "reports/feature_engineering_report.json"
)

VITAL_SIGN_COLUMNS = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
)

EXTENDED_TEMPORAL_COLUMNS = (
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "Lactate",
    "FiO2",
    "Creatinine",
    "WBC",
    "Platelets",
)

CANDIDATE_BASE_COLUMNS = (
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "ICULOS",
)

IDENTIFIER_COLUMNS = (
    "Patient_ID",
    "Dataset",
)

TARGET_COLUMN = "SepsisLabel"

ROLLING_WINDOWS = (3, 6, 12)


class TemporalFeatureEngineer:
    """Create patient-wise temporal features without future data leakage."""

    def __init__(
        self,
        input_path: str | Path = INPUT_DATASET_PATH,
        output_path: str | Path = ENGINEERED_DATASET_PATH,
        report_path: str | Path = FEATURE_ENGINEERING_REPORT_PATH,
    ) -> None:
        """Initialize feature engineering input and output paths."""
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report_path = Path(report_path)

        self.engineered_features: list[str] = []

    def run(self) -> dict[str, Any]:
        """Run temporal feature engineering and save the output."""
        dataframe = self._load_dataset()

        original_shape = dataframe.shape

        engineered_dataframe = self.create_temporal_features(
            dataframe
        )

        report = self._create_report(
            original_shape=original_shape,
            engineered_dataframe=engineered_dataframe,
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        engineered_dataframe.to_csv(
            self.output_path,
            index=False,
            encoding="utf-8",
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

    def create_temporal_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create leakage-safe temporal features.

        Features are calculated independently for every patient and
        use only the current observation and previous observations.
        """

        self._validate_required_columns(dataframe)

        engineered_dataframe = dataframe.copy()

        # Ensure correct temporal ordering.
        engineered_dataframe = engineered_dataframe.sort_values(
            by=["Patient_ID", "ICULOS"],
            kind="mergesort",
        ).reset_index(drop=True)

        # Make sure ICULOS is numeric.
        engineered_dataframe["ICULOS"] = pd.to_numeric(
            engineered_dataframe["ICULOS"],
            errors="coerce",
        )

        self.engineered_features = []

        temporal_columns = [
            column
            for column in EXTENDED_TEMPORAL_COLUMNS
            if column in engineered_dataframe.columns
        ]

        # Process each available temporal clinical variable.
        for column in temporal_columns:

            # --------------------------------------------------
            # Missing indicator
            # --------------------------------------------------
            missing_column = f"{column}_missing"

            engineered_dataframe[missing_column] = (
                engineered_dataframe[column]
                .isna()
                .astype("int8")
            )

            # --------------------------------------------------
            # Previous observation
            # --------------------------------------------------
            lag_column = f"{column}_lag_1"

            engineered_dataframe[lag_column] = (
                engineered_dataframe
                .groupby("Patient_ID", sort=False)[column]
                .shift(1)
            )

            # --------------------------------------------------
            # Change from previous observation
            # --------------------------------------------------
            change_column = f"{column}_change_1"

            engineered_dataframe[change_column] = (
                engineered_dataframe[column]
                - engineered_dataframe[lag_column]
            )

            self.engineered_features.extend(
                [
                    missing_column,
                    lag_column,
                    change_column,
                ]
            )

            # --------------------------------------------------
            # Rolling features
            # --------------------------------------------------
            for window in ROLLING_WINDOWS:

                mean_column = (
                    f"{column}_rolling_mean_{window}"
                )

                min_column = (
                    f"{column}_rolling_min_{window}"
                )

                max_column = f"{column}_rolling_max_{window}"

                std_column = f"{column}_rolling_std_{window}"

                trend_column = f"{column}_trend_{window}"

                grouped = (
                    engineered_dataframe
                    .groupby("Patient_ID", sort=False)[column]
                )

                rolling_values = grouped.rolling(
                    window=window,
                    min_periods=1,
                )

                mean_values = (
                    rolling_values
                    .mean()
                    .reset_index(level=0, drop=True)
                )

                min_values = (
                    rolling_values
                    .min()
                    .reset_index(level=0, drop=True)
                )

                max_values = (
                    rolling_values
                    .max()
                    .reset_index(level=0, drop=True)
                )

                std_values = (
                    rolling_values
                    .std()
                    .reset_index(level=0, drop=True)
                )

                engineered_dataframe[mean_column] = (
                    mean_values
                )

                engineered_dataframe[min_column] = (
                    min_values
                )

                engineered_dataframe[max_column] = (
                    max_values
                )

                engineered_dataframe[std_column] = (
                    std_values
                )

                lag_window = (
                    engineered_dataframe
                    .groupby("Patient_ID", sort=False)[column]
                    .shift(window)
                )

                engineered_dataframe[trend_column] = (
                    engineered_dataframe[column]
                    - lag_window
                )

                self.engineered_features.extend(
                    [
                        mean_column,
                        min_column,
                        max_column,
                        std_column,
                        trend_column,
                    ]
                )

            measured_flag = engineered_dataframe[column].notna()

            measurement_index = (
                measured_flag
                .groupby(engineered_dataframe["Patient_ID"])
                .cumsum()
            )

            last_measured_iculos = (
                engineered_dataframe["ICULOS"]
                .where(measured_flag)
                .groupby(engineered_dataframe["Patient_ID"])
                .ffill()
            )

            time_since_column = (
                f"{column}_time_since_measured"
            )

            engineered_dataframe[time_since_column] = (
                engineered_dataframe["ICULOS"]
                - last_measured_iculos
            )

            engineered_dataframe.loc[
                measurement_index == 0,
                time_since_column,
            ] = pd.NA

            self.engineered_features.append(time_since_column)

        self._add_clinical_interactions(engineered_dataframe)

        return engineered_dataframe

    def _add_clinical_interactions(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Add clinically interpretable interaction features."""

        def has_columns(*columns: str) -> bool:
            return all(column in dataframe.columns for column in columns)

        if has_columns("HR", "SBP"):
            dataframe["Shock_Index"] = (
                dataframe["HR"] / dataframe["SBP"]
            )
            self.engineered_features.append("Shock_Index")

        if "MAP" in dataframe.columns:
            dataframe["MAP_Below_65"] = (
                dataframe["MAP"] < 65
            ).astype("int8")
            self.engineered_features.append("MAP_Below_65")

        if "Lactate" in dataframe.columns:
            dataframe["Lactate_Above_2"] = (
                dataframe["Lactate"] > 2
            ).astype("int8")
            dataframe["Lactate_Above_4"] = (
                dataframe["Lactate"] > 4
            ).astype("int8")
            self.engineered_features.extend(
                [
                    "Lactate_Above_2",
                    "Lactate_Above_4",
                ]
            )

        if "O2Sat" in dataframe.columns:
            dataframe["O2Sat_Below_92"] = (
                dataframe["O2Sat"] < 92
            ).astype("int8")
            self.engineered_features.append("O2Sat_Below_92")

        if "Resp" in dataframe.columns:
            dataframe["Resp_Above_22"] = (
                dataframe["Resp"] > 22
            ).astype("int8")
            self.engineered_features.append("Resp_Above_22")

        if has_columns("Age", "HR"):
            dataframe["Age_x_HR"] = (
                dataframe["Age"] * dataframe["HR"]
            )
            self.engineered_features.append("Age_x_HR")

        if has_columns("Age", "Resp"):
            dataframe["Age_x_Resp"] = (
                dataframe["Age"] * dataframe["Resp"]
            )
            self.engineered_features.append("Age_x_Resp")

        if has_columns("Lactate", "MAP"):
            dataframe["Lactate_x_MAP"] = (
                dataframe["Lactate"] * dataframe["MAP"]
            )
            self.engineered_features.append("Lactate_x_MAP")

        if has_columns("Resp", "O2Sat"):
            dataframe["Resp_x_O2Sat"] = (
                dataframe["Resp"] * dataframe["O2Sat"]
            )
            self.engineered_features.append("Resp_x_O2Sat")

    def _load_dataset(self) -> pd.DataFrame:
        """Load the cleaned dataset."""
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input dataset not found: {self.input_path}"
            )

        dataframe = pd.read_csv(self.input_path)

        if dataframe.empty:
            raise ValueError(
                "The input dataset is empty."
            )

        return dataframe

    @staticmethod
    def _validate_required_columns(
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate required columns."""

        required_columns = (
            *IDENTIFIER_COLUMNS,
            *CANDIDATE_BASE_COLUMNS,
            *VITAL_SIGN_COLUMNS,
            TARGET_COLUMN,
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

    def _create_report(
        self,
        original_shape: tuple[int, int],
        engineered_dataframe: pd.DataFrame,
    ) -> dict[str, Any]:
        """Create feature engineering report."""

        original_feature_count = (
            original_shape[1]
            - len(IDENTIFIER_COLUMNS)
            - 1
        )

        final_feature_count = (
            engineered_dataframe.shape[1]
            - len(IDENTIFIER_COLUMNS)
            - 1
        )

        return {
            "original_shape": list(original_shape),

            "engineered_shape": list(
                engineered_dataframe.shape
            ),

            "number_of_patients": int(
                engineered_dataframe["Patient_ID"]
                .nunique()
            ),

            "original_feature_count": (
                original_feature_count
            ),

            "final_feature_count": (
                final_feature_count
            ),

            "engineered_feature_count": len(
                self.engineered_features
            ),

            "engineered_features": (
                self.engineered_features
            ),

            "temporal_ordering_used": [
                "Patient_ID",
                "ICULOS",
            ],

            "rolling_windows": list(
                ROLLING_WINDOWS
            ),

            "future_observations_used": False,

            "SepsisLabel_used_as_input_feature": False,

            "Patient_ID_used_as_input_feature": False,
        }


if __name__ == "__main__":

    engineer = TemporalFeatureEngineer()

    feature_report = engineer.run()

    print(
        "TEMPORAL FEATURE ENGINEERING COMPLETE"
    )

    print(
        "Original shape: "
        f"{tuple(feature_report['original_shape'])}"
    )

    print(
        "Engineered shape: "
        f"{tuple(feature_report['engineered_shape'])}"
    )

    print(
        "Number of patients: "
        f"{feature_report['number_of_patients']}"
    )

    print(
        "Engineered features: "
        f"{feature_report['engineered_feature_count']}"
    )

    print(
        "Future observations used: "
        f"{feature_report['future_observations_used']}"
    )

    print(
        "Report saved to: "
        f"{FEATURE_ENGINEERING_REPORT_PATH.resolve()}"
    )

    print(
        "Dataset saved to: "
        f"{ENGINEERED_DATASET_PATH.resolve()}"
    )
