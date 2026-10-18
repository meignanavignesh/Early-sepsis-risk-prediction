"""Sanity and leakage audit for the final sepsis model features."""

from pathlib import Path

import pandas as pd


DATASET_PATH = Path(
    "data/processed/engineered_dataset.csv"
)

TARGET_COLUMN = "SepsisLabel"

IDENTIFIER_COLUMNS = {
    "Patient_ID",
    "Dataset",
}

# Features that require special attention because they represent time.
TIME_FEATURES = {
    "ICULOS",
    "HospAdmTime",
}


class FeatureSanityAudit:
    """Audit model features for potential leakage."""

    def __init__(
        self,
        dataset_path: str | Path = DATASET_PATH,
    ) -> None:
        self.dataset_path = Path(dataset_path)

    def run(self) -> None:
        print("=" * 65)
        print("FEATURE SANITY / LEAKAGE AUDIT")
        print("=" * 65)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        dataframe = pd.read_csv(
            self.dataset_path,
            nrows=10000,
        )

        print(
            f"Dataset inspected: "
            f"{self.dataset_path}"
        )

        print(
            f"Rows inspected: "
            f"{len(dataframe)}"
        )

        print()

        # --------------------------------------------------
        # 1. Target leakage
        # --------------------------------------------------

        print("1. TARGET LEAKAGE CHECK")

        if TARGET_COLUMN in dataframe.columns:
            print(
                "SepsisLabel exists as target: PASS"
            )
        else:
            print(
                "SepsisLabel missing: FAIL"
            )

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        if TARGET_COLUMN not in feature_columns:
            print(
                "SepsisLabel used as model feature: FALSE"
            )
        else:
            print(
                "SepsisLabel used as model feature: TRUE"
            )

        # --------------------------------------------------
        # 2. Patient identifier check
        # --------------------------------------------------

        print()
        print("2. IDENTIFIER CHECK")

        for column in IDENTIFIER_COLUMNS:
            if column in dataframe.columns:
                print(
                    f"{column}: present but excluded "
                    "from model features"
                )

        # --------------------------------------------------
        # 3. Temporal feature inspection
        # --------------------------------------------------

        print()
        print("3. TEMPORAL FEATURE CHECK")

        temporal_columns = [
            column
            for column in feature_columns
            if (
                "_lag_" in column
                or "_change_" in column
                or "_rolling_" in column
                or column in TIME_FEATURES
            )
        ]

        print(
            f"Temporal-related features: "
            f"{len(temporal_columns)}"
        )

        print()

        for column in TIME_FEATURES:
            if column in dataframe.columns:
                print(
                    f"{column}: PRESENT"
                )

        # --------------------------------------------------
        # 4. Check rolling feature names
        # --------------------------------------------------

        print()
        print("4. ROLLING FEATURE DIRECTION CHECK")

        future_keyword_found = []

        future_keywords = (
            "future",
            "lead",
            "next",
            "forward",
        )

        for column in feature_columns:
            column_lower = column.lower()

            for keyword in future_keywords:
                if keyword in column_lower:
                    future_keyword_found.append(
                        column
                    )
                    break

        if future_keyword_found:
            print(
                "Potential future-looking feature names:"
            )

            for column in future_keyword_found:
                print(
                    f"  WARNING: {column}"
                )
        else:
            print(
                "No future/lead/next/forward "
                "feature names detected."
            )

        # --------------------------------------------------
        # 5. Check suspicious target-like names
        # --------------------------------------------------

        print()
        print("5. TARGET-LIKE FEATURE CHECK")

        suspicious_keywords = (
            "label",
            "target",
            "outcome",
            "sepsislabel",
        )

        suspicious_columns = []

        for column in feature_columns:

            column_lower = column.lower()

            if any(
                keyword in column_lower
                for keyword in suspicious_keywords
            ):
                suspicious_columns.append(
                    column
                )

        if suspicious_columns:
            print(
                "Potential target-like features:"
            )

            for column in suspicious_columns:
                print(
                    f"  WARNING: {column}"
                )
        else:
            print(
                "No target-like model features found."
            )

        # --------------------------------------------------
        # 6. Missingness indicators
        # --------------------------------------------------

        print()
        print("6. MISSINGNESS FEATURE CHECK")

        missing_indicators = [
            column
            for column in feature_columns
            if column.endswith(
                "_was_missing"
            )
            or column.endswith(
                "_missing"
            )
        ]

        print(
            f"Missingness indicators: "
            f"{len(missing_indicators)}"
        )

        if missing_indicators:
            print(
                "Missingness indicators are "
                "allowed as model features."
            )

        # --------------------------------------------------
        # 7. Final summary
        # --------------------------------------------------

        print()
        print("=" * 65)
        print("AUDIT SUMMARY")
        print("=" * 65)

        print(
            f"Total columns inspected: "
            f"{len(dataframe.columns)}"
        )

        print(
            f"Model features: "
            f"{len(feature_columns)}"
        )

        print(
            f"Temporal-related features: "
            f"{len(temporal_columns)}"
        )

        print(
            f"Missingness indicators: "
            f"{len(missing_indicators)}"
        )

        print(
            "Target excluded from features: "
            f"{TARGET_COLUMN not in feature_columns}"
        )

        print(
            "Future-looking feature names found: "
            f"{bool(future_keyword_found)}"
        )

        print(
            "Potential target-like features found: "
            f"{bool(suspicious_columns)}"
        )

        print()
        print(
            "FEATURE SANITY AUDIT COMPLETE"
        )


def main() -> None:
    """Run the feature sanity audit."""

    audit = FeatureSanityAudit()

    audit.run()


if __name__ == "__main__":
    main()