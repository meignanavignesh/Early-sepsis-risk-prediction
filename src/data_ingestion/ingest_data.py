"""Data ingestion pipeline for PhysioNet sepsis patient PSV files."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.constants import DEFAULT_ENCODING
from src.config.paths import INTERIM_DATA_DIR
from src.data_ingestion.file_reader import FileReader
from src.data_ingestion.merger import PatientDataMerger
from src.data_ingestion.validator import DataValidator


class DataIngestionPipeline:
    """Orchestrate file reading, validation, merging, and output writing."""

    DATASET_FOLDERS = ("training_setA", "training_setB")

    def __init__(
        self,
        data_directory: str | Path,
        output_directory: str | Path = INTERIM_DATA_DIR,
        batch_size: int = 500,
    ) -> None:
        """Initialize the ingestion pipeline."""
        self.data_directory = Path(data_directory)
        self.output_directory = Path(output_directory)
        self.batch_size = batch_size
        self.validator = DataValidator()
        self.merger = PatientDataMerger()

    def run(self) -> Path:
        """Run ingestion for all supported training sets and save outputs."""
        report = self._create_report()
        output_file = self.output_directory / "merged_dataset.csv"
        report_file = self.output_directory / "ingestion_report.json"
        write_header = True

        self.output_directory.mkdir(parents=True, exist_ok=True)

        for dataset_name in self.DATASET_FOLDERS:
            dataset_directory = self.data_directory / dataset_name
            dataset_batches = self._process_dataset(dataset_directory, report)

            for batch in dataset_batches:
                batch.to_csv(
                    output_file,
                    mode="w" if write_header else "a",
                    header=write_header,
                    index=False,
                    encoding=DEFAULT_ENCODING,
                )
                write_header = False
                self._update_report_from_batch(report, batch)

        with report_file.open("w", encoding=DEFAULT_ENCODING) as file:
            json.dump(report, file, indent=4)

        return output_file

    def _process_dataset(
        self,
        dataset_directory: Path,
        report: dict[str, Any],
    ) -> list[pd.DataFrame]:
        """Read, validate, and merge a dataset folder in batches."""
        dataset_name = dataset_directory.name
        merged_batches: list[pd.DataFrame] = []

        try:
            reader = FileReader(dataset_directory)
            psv_files = reader.get_all_psv_files()
        except (FileNotFoundError, NotADirectoryError) as error:
            report["failures"].append(
                {"dataset": dataset_name, "file": None, "error": str(error)}
            )
            return merged_batches

        report["files_discovered"] += len(psv_files)
        report["files_by_dataset"][dataset_name] = len(psv_files)

        valid_records: dict[str, pd.DataFrame] = {}

        for file_path in psv_files:
            patient_id = file_path.stem

            try:
                dataframe = reader.read_single_file(file_path)
                report["files_successfully_read"] += 1
                self.validator.validate_dataframe(dataframe)
                valid_records[patient_id] = dataframe
                report["valid_patient_records"] += 1
            except (ValueError, FileNotFoundError) as error:
                report["invalid_patient_records"] += 1
                report["failures"].append(
                    {
                        "dataset": dataset_name,
                        "file": file_path.name,
                        "error": str(error),
                    }
                )
                continue

            if len(valid_records) >= self.batch_size:
                merged_batches.append(
                    self._merge_batch(valid_records, dataset_name)
                )
                valid_records = {}

        if valid_records:
            merged_batches.append(self._merge_batch(valid_records, dataset_name))

        return merged_batches

    def _merge_batch(
        self,
        patient_records: dict[str, pd.DataFrame],
        dataset_name: str,
    ) -> pd.DataFrame:
        """Merge a batch of valid records and add the source dataset name."""
        merged_dataframe = self.merger.merge_patient_records(patient_records)
        merged_dataframe["Dataset"] = dataset_name

        return merged_dataframe

    def _create_report(self) -> dict[str, Any]:
        """Create the initial ingestion report structure."""
        return {
            "files_discovered": 0,
            "files_successfully_read": 0,
            "valid_patient_records": 0,
            "invalid_patient_records": 0,
            "total_rows": 0,
            "columns": 0,
            "patients": 0,
            "sepsis_positive_patients": 0,
            "non_sepsis_patients": 0,
            "files_by_dataset": {
                "training_setA": 0,
                "training_setB": 0,
            },
            "failures": [],
        }

    def _update_report_from_batch(
        self,
        report: dict[str, Any],
        dataframe: pd.DataFrame,
    ) -> None:
        """Update row, column, and patient-level report values for a batch."""
        report["total_rows"] += len(dataframe)
        report["columns"] = len(dataframe.columns)

        if dataframe.empty:
            return

        report["patients"] += dataframe["Patient_ID"].nunique()
        patient_labels = dataframe.groupby("Patient_ID")["SepsisLabel"].max()
        report["sepsis_positive_patients"] += int((patient_labels == 1).sum())
        report["non_sepsis_patients"] += int((patient_labels == 0).sum())
