"""Evaluate probability calibration for the sepsis model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


MODEL_PATH = Path("models/xgboost.pkl")
TEST_PATH = Path("data/processed/model_ready/test.csv")
REPORT_PATH = Path("reports/calibration_report.json")
BINS_PATH = Path("reports/calibration_bins.csv")

TARGET_COLUMN = "SepsisLabel"
IDENTIFIER_COLUMNS = {"Patient_ID", "Dataset"}


class CalibrationEvaluator:
    """Evaluate observed vs predicted risk."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        test_path: str | Path = TEST_PATH,
        report_path: str | Path = REPORT_PATH,
        bins_path: str | Path = BINS_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.test_path = Path(test_path)
        self.report_path = Path(report_path)
        self.bins_path = Path(bins_path)

    def run(self) -> dict:
        """Run calibration evaluation."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        if not self.test_path.exists():
            raise FileNotFoundError(
                f"Test data not found: {self.test_path}"
            )

        model = joblib.load(self.model_path)
        test = pd.read_csv(self.test_path)

        feature_columns = [
            column
            for column in test.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        if len(feature_columns) != model.n_features_in_:
            raise ValueError(
                "Feature count mismatch. "
                f"Model expects {model.n_features_in_}, "
                f"found {len(feature_columns)}."
            )

        y_true = test[TARGET_COLUMN].astype(int)
        probabilities = model.predict_proba(
            test[feature_columns]
        )[:, 1]

        bin_table = self._bin_table(y_true, probabilities)
        fraction_positive, mean_predicted = calibration_curve(
            y_true,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )

        ece = self._expected_calibration_error(bin_table)

        self.bins_path.parent.mkdir(parents=True, exist_ok=True)
        bin_table.to_csv(self.bins_path, index=False)

        report = {
            "evaluation_type": "probability_calibration",
            "dataset": str(self.test_path),
            "rows": int(len(test)),
            "brier_score": float(
                brier_score_loss(y_true, probabilities)
            ),
            "expected_calibration_error": float(ece),
            "calibration_curve": {
                "mean_predicted_probability": [
                    float(value) for value in mean_predicted
                ],
                "fraction_positive": [
                    float(value) for value in fraction_positive
                ],
            },
            "bins_csv": str(self.bins_path),
            "calibration_note": (
                "Optional Platt scaling or isotonic calibration "
                "should be fit on validation data before final test use."
            ),
        }

        with self.report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        print(f"Calibration report saved to: {self.report_path.resolve()}")
        print(f"Calibration bins saved to: {self.bins_path.resolve()}")

        return report

    @staticmethod
    def _bin_table(
        y_true: pd.Series,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        """Build observed vs predicted risk bins."""

        table = pd.DataFrame(
            {
                "actual": y_true.astype(int),
                "probability": probabilities,
            }
        )
        table["bin"] = pd.qcut(
            table["probability"],
            q=10,
            duplicates="drop",
        )

        bins = (
            table.groupby("bin", observed=False)
            .agg(
                observations=("actual", "size"),
                mean_predicted_risk=("probability", "mean"),
                observed_risk=("actual", "mean"),
            )
            .reset_index()
        )
        bins["bin"] = bins["bin"].astype(str)
        bins["absolute_calibration_error"] = (
            bins["observed_risk"]
            - bins["mean_predicted_risk"]
        ).abs()

        return bins

    @staticmethod
    def _expected_calibration_error(
        bin_table: pd.DataFrame,
    ) -> float:
        """Calculate weighted expected calibration error."""

        total = bin_table["observations"].sum()
        if total == 0:
            return 0.0

        return float(
            (
                bin_table["observations"]
                / total
                * bin_table["absolute_calibration_error"]
            ).sum()
        )


def main() -> None:
    """Run calibration evaluation."""

    CalibrationEvaluator().run()


if __name__ == "__main__":
    main()
