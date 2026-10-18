"""Patient DataFrame merging utilities for sepsis records."""

import pandas as pd


class PatientDataMerger:
    """Combine validated patient DataFrames into a single DataFrame."""

    def merge_patient_records(
        self,
        patient_records: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge patient records row-wise and add the Patient_ID column."""
        if not patient_records:
            return pd.DataFrame()

        patient_dataframes: list[pd.DataFrame] = []

        for patient_id, dataframe in patient_records.items():
            patient_dataframe = dataframe.copy()
            patient_dataframe["Patient_ID"] = patient_id
            patient_dataframes.append(patient_dataframe)

        return pd.concat(patient_dataframes, axis=0, ignore_index=True)
