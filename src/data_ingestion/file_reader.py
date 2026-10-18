"""Utilities for reading patient PSV files from a data folder."""

from pathlib import Path

import pandas as pd


class FileReader:
    """Read patient-level pipe-separated value files from a folder."""

    def __init__(self, folder_path: str | Path) -> None:
        """Initialize the reader with a folder containing PSV files."""
        self.folder_path = Path(folder_path)

        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"Data folder does not exist: {self.folder_path}"
            )

        if not self.folder_path.is_dir():
            raise NotADirectoryError(
                f"Expected a folder path, got: {self.folder_path}"
            )

    def get_all_psv_files(self) -> list[Path]:
        """Return all PSV files in the folder sorted by filename."""
        psv_files = sorted(self.folder_path.glob("*.psv"))

        if not psv_files:
            raise FileNotFoundError(
                f"No .psv files found in folder: {self.folder_path}"
            )

        return psv_files

    def read_single_file(
        self,
        file_path: str | Path,
    ) -> pd.DataFrame:
        """Read a single PSV file and return it as a DataFrame."""
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PSV file does not exist: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Expected a file path, got: {path}"
            )

        if path.suffix.lower() != ".psv":
            raise ValueError(
                f"Expected a .psv file, got: {path}"
            )

        try:
            return pd.read_csv(path, sep="|")

        except pd.errors.EmptyDataError as error:
            raise ValueError(
                f"PSV file is empty or corrupted: {path}"
            ) from error

        except pd.errors.ParserError as error:
            raise ValueError(
                f"Unable to parse corrupted PSV file: {path}"
            ) from error

        except UnicodeDecodeError as error:
            raise ValueError(
                f"Unable to decode PSV file: {path}"
            ) from error

    def read_all_files(self) -> dict[str, pd.DataFrame]:
        """Read all PSV files and return DataFrames keyed by patient ID."""
        patient_records: dict[str, pd.DataFrame] = {}

        for file_path in self.get_all_psv_files():
            patient_id = file_path.stem
            patient_records[patient_id] = self.read_single_file(file_path)

        return patient_records