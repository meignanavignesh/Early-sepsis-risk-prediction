"""Small test for temporal feature engineering."""

import pandas as pd

from src.feature_engineering.temporal_features import (
    TemporalFeatureEngineer,
)


INPUT_PATH = "data/processed/cleaned_dataset.csv"


def main() -> None:
    dataframe = pd.read_csv(INPUT_PATH)

    test_patient_ids = [
        "p000001",
        "p000002",
        "p000003",
    ]

    test_dataframe = dataframe[
        dataframe["Patient_ID"].isin(test_patient_ids)
    ].copy()

    print("TEST DATA")
    print("=" * 60)

    print(f"Patients selected: {test_dataframe['Patient_ID'].nunique()}")
    print(f"Rows selected: {len(test_dataframe)}")

    engineer = TemporalFeatureEngineer()

    result = engineer.create_temporal_features(
        test_dataframe
    )

    print("\nRESULT")
    print("=" * 60)

    print(f"Shape: {result.shape}")
    print(
        f"Patients: "
        f"{result['Patient_ID'].nunique()}"
    )

    print(
        f"Patient_ID exists: "
        f"{'Patient_ID' in result.columns}"
    )

    print(
        f"ICULOS exists: "
        f"{'ICULOS' in result.columns}"
    )

    print(
        f"SepsisLabel exists: "
        f"{'SepsisLabel' in result.columns}"
    )

    print("\nENGINEERED COLUMNS")
    print("=" * 60)

    expected_columns = [
        "HR_missing",
        "HR_lag_1",
        "HR_change_1",
        "HR_rolling_mean_3",
        "HR_rolling_min_3",
        "HR_rolling_max_3",
        "HR_rolling_mean_6",
        "HR_rolling_min_6",
        "HR_rolling_max_6",
    ]

    for column in expected_columns:
        print(
            f"{column}: "
            f"{column in result.columns}"
        )

    print("\nFIRST PATIENT")
    print("=" * 60)

    patient = result[
        result["Patient_ID"] == "p000001"
    ].sort_values("ICULOS")

    display_columns = [
        "Patient_ID",
        "ICULOS",
        "HR",
        "HR_lag_1",
        "HR_change_1",
        "HR_rolling_mean_3",
    ]

    print(
        patient[display_columns].head(10).to_string(
            index=False
        )
    )

    # Basic validation
    assert result["Patient_ID"].nunique() == 3
    assert "SepsisLabel" in result.columns
    assert "ICULOS" in result.columns
    assert "HR_lag_1" in result.columns
    assert "HR_change_1" in result.columns
    assert "HR_rolling_mean_3" in result.columns

    # Verify ordering
    for patient_id, group in result.groupby(
        "Patient_ID",
        sort=False,
    ):
        assert group["ICULOS"].is_monotonic_increasing

    print("\n" + "=" * 60)
    print("TEMPORAL FEATURE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()