"""Optimize hourly patient-level alert rules on validation data."""

import json
from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.hourly_rules import (
    AlertRule,
    add_hourly_risk_features,
    aggregate_patient_alerts,
    apply_alert_rule,
    evaluate_patient_predictions,
)
from src.utils.experiment_tracker import ExperimentTracker


MODEL_PATH = Path("models/xgboost.pkl")
VALIDATION_PATH = Path("data/processed/model_ready/validation.csv")
REPORT_PATH = Path("reports/hourly_threshold_optimization_report.json")
RESULTS_PATH = Path("reports/hourly_threshold_results.csv")

TARGET_COLUMN = "SepsisLabel"
IDENTIFIER_COLUMNS = {"Patient_ID", "Dataset"}


class HourlyThresholdOptimizer:
    """Compare hourly alert strategies on validation patients."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        validation_path: str | Path = VALIDATION_PATH,
        report_path: str | Path = REPORT_PATH,
        results_path: str | Path = RESULTS_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.validation_path = Path(validation_path)
        self.report_path = Path(report_path)
        self.results_path = Path(results_path)

    def run(self) -> dict:
        """Run hourly rule optimization."""

        print("=" * 70)
        print("HOURLY THRESHOLD AND ALERT-RULE OPTIMIZATION")
        print("=" * 70)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        if not self.validation_path.exists():
            raise FileNotFoundError(
                f"Validation dataset not found: {self.validation_path}"
            )

        model = joblib.load(self.model_path)
        print("Loading validation data...")
        validation = pd.read_csv(self.validation_path)

        feature_columns = [
            column
            for column in validation.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        if len(feature_columns) != model.n_features_in_:
            raise ValueError(
                "Feature count mismatch. "
                f"Model expects {model.n_features_in_}, "
                f"found {len(feature_columns)}."
            )

        print("Generating validation probabilities...")
        probabilities = model.predict_proba(
            validation[feature_columns]
        )[:, 1]

        hourly = validation[
            [
                column
                for column in [
                    "Patient_ID",
                    "ICULOS",
                    "SepsisLabel",
                    "Age",
                    "Gender",
                    "Unit1",
                    "Unit2",
                ]
                if column in validation.columns
            ]
        ].copy()
        hourly["Sepsis_Probability"] = probabilities
        print("Adding hourly risk-window features...")
        hourly = add_hourly_risk_features(hourly)

        rules = self._build_rules()
        rows = []

        print(f"Evaluating {len(rules)} alert rules...")
        for index, rule in enumerate(rules, start=1):
            if index == 1 or index % 10 == 0:
                print(f"  Rule {index}/{len(rules)}: {rule.name}")

            ruled = apply_alert_rule(
                hourly,
                rule,
                features_already_added=True,
            )
            patient_data = aggregate_patient_alerts(ruled)
            metrics = evaluate_patient_predictions(patient_data)

            rows.append(
                {
                    "rule_name": rule.name,
                    "strategy": rule.strategy,
                    "threshold": rule.threshold,
                    "window": rule.window,
                    "required_consecutive": (
                        rule.required_consecutive
                    ),
                    **metrics,
                }
            )

        results = pd.DataFrame(rows)

        best_f1 = (
            results.sort_values(
                by=[
                    "f1_score",
                    "recall",
                    "precision",
                    "false_positives",
                ],
                ascending=[False, False, False, True],
            )
            .iloc[0]
            .to_dict()
        )

        recall_candidates = results[
            results["recall"] >= 0.80
        ].copy()

        if recall_candidates.empty:
            best_recall_preserving = None
        else:
            best_recall_preserving = (
                recall_candidates.sort_values(
                    by=[
                        "precision",
                        "f1_score",
                        "false_positives",
                    ],
                    ascending=[False, False, True],
                )
                .iloc[0]
                .to_dict()
            )

        self.results_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        results.to_csv(self.results_path, index=False)

        report = {
            "evaluation_level": "patient_hourly",
            "dataset_used": "validation",
            "model_path": str(self.model_path),
            "validation_rows": int(len(validation)),
            "validation_patients": int(
                validation["Patient_ID"].nunique()
            ),
            "rules_tested": int(len(results)),
            "selection_metric": (
                "best F1, plus best precision with recall >= 0.80"
            ),
            "best_f1_rule": self._json_ready(best_f1),
            "best_rule_with_recall_at_least_0_80": (
                self._json_ready(best_recall_preserving)
            ),
            "results_csv": str(self.results_path),
            "test_set_used_for_selection": False,
            "data_leakage_prevented": True,
        }

        with self.report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        tracker = ExperimentTracker()
        tracker.log_run(
            model_name="xgboost",
            dataset_paths={
                "validation": str(self.validation_path),
            },
            feature_count=len(feature_columns),
            parameters={
                "rules_tested": len(results),
            },
            metrics={
                "best_f1": best_f1.get("f1_score"),
                "best_recall_preserving_precision": (
                    None
                    if best_recall_preserving is None
                    else best_recall_preserving.get("precision")
                ),
            },
            artifacts={
                "report": str(self.report_path),
                "results": str(self.results_path),
            },
        )

        print("Optimization complete.")
        print(f"Results saved to: {self.results_path.resolve()}")
        print(f"Report saved to: {self.report_path.resolve()}")

        return report

    @staticmethod
    def _build_rules() -> list[AlertRule]:
        """Create candidate hourly alert rules."""

        thresholds = [round(x / 100, 2) for x in range(30, 91, 5)]
        rules: list[AlertRule] = []

        for threshold in thresholds:
            rules.append(
                AlertRule(
                    name=f"single_threshold_{threshold:.2f}",
                    strategy="single_threshold",
                    threshold=threshold,
                )
            )

            for consecutive in (2, 3):
                rules.append(
                    AlertRule(
                        name=(
                            f"{consecutive}_consecutive_"
                            f"{threshold:.2f}"
                        ),
                        strategy="consecutive",
                        threshold=threshold,
                        required_consecutive=consecutive,
                    )
                )

            rules.append(
                AlertRule(
                    name=f"rolling_mean_3h_{threshold:.2f}",
                    strategy="rolling_mean",
                    threshold=threshold,
                    window=3,
                )
            )

            rules.append(
                AlertRule(
                    name=f"rolling_max_6h_{threshold:.2f}",
                    strategy="rolling_max",
                    threshold=threshold,
                    window=6,
                )
            )

        rules.append(
            AlertRule(
                name="risk_tier_high",
                strategy="risk_tier",
                threshold=0.50,
            )
        )

        return rules

    @staticmethod
    def _json_ready(value):
        """Convert pandas/numpy scalars to plain Python values."""

        if value is None:
            return None

        result = {}
        for key, item in value.items():
            if pd.isna(item):
                result[key] = None
            elif hasattr(item, "item"):
                result[key] = item.item()
            else:
                result[key] = item

        return result


def main() -> None:
    """Run hourly threshold optimization."""

    HourlyThresholdOptimizer().run()


if __name__ == "__main__":
    main()
