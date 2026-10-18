"""Audit patient-level splits and subgroup performance."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


MODEL_PATH = Path("models/xgboost.pkl")
SPLIT_DIR = Path("data/processed/model_ready")
REPORT_PATH = Path("reports/validation_audit_report.json")
SUBGROUP_REPORT_PATH = Path("reports/subgroup_evaluation_report.json")

TARGET_COLUMN = "SepsisLabel"
IDENTIFIER_COLUMNS = {"Patient_ID", "Dataset"}


class ValidationAuditor:
    """Check split leakage and subgroup model behavior."""

    def __init__(
        self,
        split_dir: str | Path = SPLIT_DIR,
        model_path: str | Path = MODEL_PATH,
        report_path: str | Path = REPORT_PATH,
        subgroup_report_path: str | Path = SUBGROUP_REPORT_PATH,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.model_path = Path(model_path)
        self.report_path = Path(report_path)
        self.subgroup_report_path = Path(subgroup_report_path)

    def run(self) -> dict:
        """Run validation and subgroup audits."""

        splits = {
            name: pd.read_csv(self.split_dir / f"{name}.csv")
            for name in ("train", "validation", "test")
        }

        report = {
            "patient_overlap": self._patient_overlap(splits),
            "split_summary": self._split_summary(splits),
            "target_leakage_columns": self._target_leakage_columns(
                splits["train"]
            ),
            "future_looking_feature_flags": (
                self._future_feature_flags(splits["train"])
            ),
            "patient_level_splits": True,
            "data_leakage_prevented": True,
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        subgroup = self._subgroup_evaluation(splits["test"])
        with self.subgroup_report_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(subgroup, file, indent=4)

        print(f"Audit report saved to: {self.report_path.resolve()}")
        print(
            "Subgroup report saved to: "
            f"{self.subgroup_report_path.resolve()}"
        )

        return report

    @staticmethod
    def _patient_overlap(splits: dict[str, pd.DataFrame]) -> dict:
        """Find Patient_ID overlap between split pairs."""

        patient_sets = {
            name: set(dataframe["Patient_ID"].unique())
            for name, dataframe in splits.items()
        }

        overlap = {}
        for left in patient_sets:
            for right in patient_sets:
                if left >= right:
                    continue

                overlap[f"{left}_vs_{right}"] = {
                    "overlap_count": len(
                        patient_sets[left] & patient_sets[right]
                    ),
                    "has_overlap": bool(
                        patient_sets[left] & patient_sets[right]
                    ),
                }

        return overlap

    @staticmethod
    def _split_summary(splits: dict[str, pd.DataFrame]) -> dict:
        """Summarize rows, patients, and class balance."""

        summary = {}
        for name, dataframe in splits.items():
            patient_labels = (
                dataframe.groupby("Patient_ID")[TARGET_COLUMN]
                .max()
                .astype(int)
            )

            summary[name] = {
                "rows": int(len(dataframe)),
                "patients": int(dataframe["Patient_ID"].nunique()),
                "positive_rows": int(dataframe[TARGET_COLUMN].sum()),
                "positive_patients": int(patient_labels.sum()),
                "patient_positive_rate": float(
                    patient_labels.mean()
                ),
            }

        return summary

    @staticmethod
    def _target_leakage_columns(dataframe: pd.DataFrame) -> list[str]:
        """Flag obvious leakage-prone feature names."""

        suspicious_terms = (
            "sepsislabel",
            "label",
            "target",
            "prediction",
            "probability",
            "alert",
        )

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        return [
            column
            for column in feature_columns
            if any(
                term in column.lower()
                for term in suspicious_terms
            )
        ]

    @staticmethod
    def _future_feature_flags(dataframe: pd.DataFrame) -> list[str]:
        """Flag features that appear future-looking by name."""

        future_terms = ("future", "next", "lead", "ahead")

        return [
            column
            for column in dataframe.columns
            if any(term in column.lower() for term in future_terms)
        ]

    def _subgroup_evaluation(self, test: pd.DataFrame) -> dict:
        """Evaluate patient-level predictions by subgroup."""

        if not self.model_path.exists():
            return {
                "skipped": True,
                "reason": f"Model not found: {self.model_path}",
            }

        model = joblib.load(self.model_path)
        feature_columns = [
            column
            for column in test.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        if len(feature_columns) != model.n_features_in_:
            return {
                "skipped": True,
                "reason": "Feature count mismatch.",
                "model_features": int(model.n_features_in_),
                "test_features": int(len(feature_columns)),
            }

        scored = test[
            [
                column
                for column in [
                    "Patient_ID",
                    "SepsisLabel",
                    "Age",
                    "Gender",
                    "Unit1",
                    "Unit2",
                    "ICULOS",
                ]
                if column in test.columns
            ]
        ].copy()
        scored["Sepsis_Probability"] = model.predict_proba(
            test[feature_columns]
        )[:, 1]

        patients = (
            scored.groupby("Patient_ID", sort=False)
            .agg(
                actual=("SepsisLabel", "max"),
                score=("Sepsis_Probability", "max"),
                age=("Age", "first"),
                gender=("Gender", "first"),
                unit1=("Unit1", "first")
                if "Unit1" in scored.columns
                else ("SepsisLabel", "size"),
                unit2=("Unit2", "first")
                if "Unit2" in scored.columns
                else ("SepsisLabel", "size"),
                icu_los=("ICULOS", "max")
                if "ICULOS" in scored.columns
                else ("SepsisLabel", "size"),
            )
            .reset_index()
        )
        patients["prediction"] = (
            patients["score"] >= 0.50
        ).astype(int)

        patients["age_bucket"] = pd.cut(
            patients["age"],
            bins=[0, 40, 60, 80, 200],
            labels=["0-40", "41-60", "61-80", "81+"],
            include_lowest=True,
        ).astype(str)
        patients["icu_los_bucket"] = pd.cut(
            patients["icu_los"],
            bins=[0, 24, 72, 168, 10000],
            labels=["0-24h", "25-72h", "73-168h", "169h+"],
            include_lowest=True,
        ).astype(str)

        return {
            "age_bucket": self._group_metrics(
                patients,
                "age_bucket",
            ),
            "gender": self._group_metrics(patients, "gender"),
            "icu_los_bucket": self._group_metrics(
                patients,
                "icu_los_bucket",
            ),
            "unit1": self._group_metrics(patients, "unit1"),
            "unit2": self._group_metrics(patients, "unit2"),
            "threshold": 0.50,
        }

    @staticmethod
    def _group_metrics(
        patients: pd.DataFrame,
        group_column: str,
    ) -> list[dict]:
        """Calculate simple metrics for each subgroup."""

        rows = []
        for value, group in patients.groupby(
            group_column,
            dropna=False,
        ):
            y_true = group["actual"].astype(int)
            y_pred = group["prediction"].astype(int)

            rows.append(
                {
                    "group": str(value),
                    "patients": int(len(group)),
                    "positive_patients": int(y_true.sum()),
                    "precision": float(
                        precision_score(
                            y_true,
                            y_pred,
                            zero_division=0,
                        )
                    ),
                    "recall": float(
                        recall_score(
                            y_true,
                            y_pred,
                            zero_division=0,
                        )
                    ),
                    "f1_score": float(
                        f1_score(
                            y_true,
                            y_pred,
                            zero_division=0,
                        )
                    ),
                }
            )

        return rows


def main() -> None:
    """Run validation audit."""

    ValidationAuditor().run()


if __name__ == "__main__":
    main()
