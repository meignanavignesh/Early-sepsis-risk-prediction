"""Patient-level train, validation, and test split."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = Path(
    "data/processed/engineered_dataset.csv"
)

OUTPUT_DIR = Path(
    "data/processed/splits"
)

REPORT_PATH = Path(
    "reports/patient_split_report.json"
)

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


class PatientLevelSplitter:
    """Split EHR observations without patient-level data leakage."""

    def __init__(
        self,
        input_path: str | Path = INPUT_PATH,
        output_dir: str | Path = OUTPUT_DIR,
        report_path: str | Path = REPORT_PATH,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.report_path = Path(report_path)

    def run(self) -> dict:
        """Perform patient-level train/validation/test split."""

        dataframe = self._load_dataset()

        patient_table = self._create_patient_table(
            dataframe
        )

        train_patients, temporary_patients = (
            train_test_split(
                patient_table,
                test_size=(1.0 - TRAIN_SIZE),
                random_state=RANDOM_STATE,
                stratify=patient_table["has_sepsis"],
            )
        )

        validation_fraction_of_remaining = (
            VALIDATION_SIZE
            / (VALIDATION_SIZE + TEST_SIZE)
        )

        validation_patients, test_patients = (
            train_test_split(
                temporary_patients,
                test_size=(
                    1.0
                    - validation_fraction_of_remaining
                ),
                random_state=RANDOM_STATE,
                stratify=temporary_patients["has_sepsis"],
            )
        )

        train_ids = set(
            train_patients["Patient_ID"]
        )

        validation_ids = set(
            validation_patients["Patient_ID"]
        )

        test_ids = set(
            test_patients["Patient_ID"]
        )

        self._validate_patient_sets(
            train_ids,
            validation_ids,
            test_ids,
        )

        train_dataframe = dataframe[
            dataframe["Patient_ID"].isin(train_ids)
        ].copy()

        validation_dataframe = dataframe[
            dataframe["Patient_ID"].isin(validation_ids)
        ].copy()

        test_dataframe = dataframe[
            dataframe["Patient_ID"].isin(test_ids)
        ].copy()

        # Preserve temporal ordering.
        train_dataframe = self._sort_dataframe(
            train_dataframe
        )

        validation_dataframe = self._sort_dataframe(
            validation_dataframe
        )

        test_dataframe = self._sort_dataframe(
            test_dataframe
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train_path = (
            self.output_dir / "train.csv"
        )

        validation_path = (
            self.output_dir / "validation.csv"
        )

        test_path = (
            self.output_dir / "test.csv"
        )

        train_dataframe.to_csv(
            train_path,
            index=False,
        )

        validation_dataframe.to_csv(
            validation_path,
            index=False,
        )

        test_dataframe.to_csv(
            test_path,
            index=False,
        )

        report = self._create_report(
            dataframe=dataframe,
            train_dataframe=train_dataframe,
            validation_dataframe=validation_dataframe,
            test_dataframe=test_dataframe,
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

    def _load_dataset(self) -> pd.DataFrame:
        """Load the engineered dataset."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.input_path}"
            )

        dataframe = pd.read_csv(
            self.input_path
        )

        required_columns = {
            "Patient_ID",
            "SepsisLabel",
            "ICULOS",
        }

        missing = (
            required_columns
            - set(dataframe.columns)
        )

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        return dataframe

    @staticmethod
    def _create_patient_table(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create one row per patient.

        has_sepsis = 1 if the patient has at least
        one SepsisLabel = 1 observation.
        """

        patient_table = (
            dataframe
            .groupby("Patient_ID", as_index=False)
            ["SepsisLabel"]
            .max()
            .rename(
                columns={
                    "SepsisLabel": "has_sepsis"
                }
            )
        )

        return patient_table

    @staticmethod
    def _sort_dataframe(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sort observations by patient and ICU time."""

        return (
            dataframe
            .sort_values(
                by=[
                    "Patient_ID",
                    "ICULOS",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    @staticmethod
    def _validate_patient_sets(
        train_ids: set,
        validation_ids: set,
        test_ids: set,
    ) -> None:
        """Ensure no patient occurs in multiple splits."""

        train_validation_overlap = (
            train_ids & validation_ids
        )

        train_test_overlap = (
            train_ids & test_ids
        )

        validation_test_overlap = (
            validation_ids & test_ids
        )

        if train_validation_overlap:
            raise ValueError(
                "Patient leakage between train and validation."
            )

        if train_test_overlap:
            raise ValueError(
                "Patient leakage between train and test."
            )

        if validation_test_overlap:
            raise ValueError(
                "Patient leakage between validation and test."
            )

    @staticmethod
    def _dataset_statistics(
        dataframe: pd.DataFrame,
    ) -> dict:
        """Return dataset statistics."""

        patients = int(
            dataframe["Patient_ID"].nunique()
        )

        sepsis_patients = int(
            dataframe
            .groupby("Patient_ID")["SepsisLabel"]
            .max()
            .sum()
        )

        non_sepsis_patients = (
            patients - sepsis_patients
        )

        observations = len(dataframe)

        positive_observations = int(
            dataframe["SepsisLabel"].sum()
        )

        negative_observations = (
            observations
            - positive_observations
        )

        return {
            "patients": patients,
            "sepsis_patients": sepsis_patients,
            "non_sepsis_patients": non_sepsis_patients,
            "observations": observations,
            "sepsis_positive_observations": (
                positive_observations
            ),
            "non_sepsis_observations": (
                negative_observations
            ),
        }

    def _create_report(
        self,
        dataframe: pd.DataFrame,
        train_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame,
        test_dataframe: pd.DataFrame,
    ) -> dict:
        """Create split verification report."""

        return {
            "random_state": RANDOM_STATE,
            "split_ratios": {
                "train": TRAIN_SIZE,
                "validation": VALIDATION_SIZE,
                "test": TEST_SIZE,
            },
            "patient_level_split": True,
            "patient_leakage_detected": False,
            "overall": self._dataset_statistics(
                dataframe
            ),
            "train": self._dataset_statistics(
                train_dataframe
            ),
            "validation": self._dataset_statistics(
                validation_dataframe
            ),
            "test": self._dataset_statistics(
                test_dataframe
            ),
        }


def main() -> None:
    """Run patient-level splitting."""

    splitter = PatientLevelSplitter()

    report = splitter.run()

    print("=" * 60)
    print("PATIENT-LEVEL DATASET SPLIT COMPLETE")
    print("=" * 60)

    print(
        f"Total patients: "
        f"{report['overall']['patients']}"
    )

    print(
        f"Training patients: "
        f"{report['train']['patients']}"
    )

    print(
        f"Validation patients: "
        f"{report['validation']['patients']}"
    )

    print(
        f"Test patients: "
        f"{report['test']['patients']}"
    )

    print(
        f"Training sepsis patients: "
        f"{report['train']['sepsis_patients']}"
    )

    print(
        f"Validation sepsis patients: "
        f"{report['validation']['sepsis_patients']}"
    )

    print(
        f"Test sepsis patients: "
        f"{report['test']['sepsis_patients']}"
    )

    print(
        f"Patient leakage detected: "
        f"{report['patient_leakage_detected']}"
    )

    print(
        f"Report saved to: "
        f"{REPORT_PATH.resolve()}"
    )

    print(
        f"Splits saved to: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()