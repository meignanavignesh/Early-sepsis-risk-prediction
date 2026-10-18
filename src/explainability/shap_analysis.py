"""SHAP explainability for the final XGBoost sepsis model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path("models/xgboost.pkl")

TEST_PATH = Path(
    "data/processed/model_ready/test.csv"
)

OUTPUT_DIR = Path(
    "reports/shap"
)

REPORT_PATH = OUTPUT_DIR / "shap_report.json"

SHAP_VALUES_PATH = OUTPUT_DIR / "shap_values.csv"

SAMPLE_SIZE = 5000

TARGET_COLUMN = "SepsisLabel"

IDENTIFIER_COLUMNS = (
    "Patient_ID",
    "Dataset",
)


class SHAPAnalyzer:
    """Generate global and individual SHAP explanations."""

    def __init__(
        self,
        model_path: str | Path = MODEL_PATH,
        test_path: str | Path = TEST_PATH,
        output_dir: str | Path = OUTPUT_DIR,
        sample_size: int = SAMPLE_SIZE,
    ) -> None:

        self.model_path = Path(model_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.sample_size = sample_size

    def run(self) -> dict:
        """Run SHAP analysis."""

        print("=" * 65)
        print("SHAP EXPLAINABILITY")
        print("=" * 65)

        # --------------------------------------------------
        # Load model
        # --------------------------------------------------

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        print("Loading XGBoost model...")

        model = joblib.load(
            self.model_path
        )

        # --------------------------------------------------
        # Load test data
        # --------------------------------------------------

        if not self.test_path.exists():
            raise FileNotFoundError(
                f"Test dataset not found: {self.test_path}"
            )

        print("Loading test dataset...")

        test = pd.read_csv(
            self.test_path
        )

        print(
            f"Total test observations: "
            f"{len(test)}"
        )

        # --------------------------------------------------
        # Identify features
        # --------------------------------------------------

        feature_columns = [
            column
            for column in test.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        X = test[
            feature_columns
        ]

        y = test[
            TARGET_COLUMN
        ]

        # --------------------------------------------------
        # Create balanced SHAP sample
        # --------------------------------------------------

        positive_data = test[
            test[TARGET_COLUMN] == 1
        ]

        negative_data = test[
            test[TARGET_COLUMN] == 0
        ]

        half_sample = (
            self.sample_size // 2
        )

        positive_count = min(
            half_sample,
            len(positive_data),
        )

        negative_count = min(
            self.sample_size - positive_count,
            len(negative_data),
        )

        positive_sample = (
            positive_data.sample(
                n=positive_count,
                random_state=42,
            )
        )

        negative_sample = (
            negative_data.sample(
                n=negative_count,
                random_state=42,
            )
        )

        shap_sample = pd.concat(
            [
                positive_sample,
                negative_sample,
            ],
            axis=0,
        )

        shap_sample = shap_sample.sample(
            frac=1.0,
            random_state=42,
        ).reset_index(
            drop=True
        )

        X_sample = shap_sample[
            feature_columns
        ]

        y_sample = shap_sample[
            TARGET_COLUMN
        ]

        print(
            f"SHAP sample size: "
            f"{len(X_sample)}"
        )

        print(
            f"SHAP positive observations: "
            f"{int((y_sample == 1).sum())}"
        )

        print(
            f"SHAP negative observations: "
            f"{int((y_sample == 0).sum())}"
        )

        # --------------------------------------------------
        # Calculate SHAP values
        # --------------------------------------------------

        print("=" * 65)
        print("CALCULATING SHAP VALUES")
        print("=" * 65)

        explainer = shap.TreeExplainer(
            model
        )

        shap_result = explainer(
            X_sample
        )

        shap_values = shap_result.values

        # --------------------------------------------------
        # Handle SHAP output format
        # --------------------------------------------------

        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        if shap_values.ndim != 2:
            raise ValueError(
                "Unexpected SHAP value shape: "
                f"{shap_values.shape}"
            )

        print(
            f"SHAP matrix shape: "
            f"{shap_values.shape}"
        )

        # --------------------------------------------------
        # Global feature importance
        # --------------------------------------------------

        mean_absolute_shap = (
            np.abs(shap_values)
            .mean(axis=0)
        )

        feature_importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "mean_absolute_shap": (
                    mean_absolute_shap
                ),
            }
        )

        feature_importance = (
            feature_importance
            .sort_values(
                "mean_absolute_shap",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        # --------------------------------------------------
        # Save SHAP values
        # --------------------------------------------------

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        shap_dataframe = pd.DataFrame(
            shap_values,
            columns=feature_columns,
        )

        shap_dataframe.to_csv(
            SHAP_VALUES_PATH,
            index=False,
        )

        # --------------------------------------------------
        # Save feature importance
        # --------------------------------------------------

        importance_path = (
            self.output_dir
            / "global_feature_importance.csv"
        )

        feature_importance.to_csv(
            importance_path,
            index=False,
        )

        # --------------------------------------------------
        # Top features
        # --------------------------------------------------

        top_features = (
            feature_importance
            .head(20)
            .to_dict(
                orient="records"
            )
        )

        top_features = [
            {
                "feature": item[
                    "feature"
                ],
                "mean_absolute_shap": float(
                    item[
                        "mean_absolute_shap"
                    ]
                ),
            }
            for item in top_features
        ]

        # --------------------------------------------------
        # Individual example
        # --------------------------------------------------

        individual_index = 0

        individual_feature_values = (
            X_sample.iloc[
                individual_index
            ].to_dict()
        )

        individual_shap_values = (
            shap_values[
                individual_index
            ]
        )

        individual_explanation = pd.DataFrame(
            {
                "feature": feature_columns,
                "feature_value": [
                    individual_feature_values[
                        feature
                    ]
                    for feature in feature_columns
                ],
                "shap_value": (
                    individual_shap_values
                ),
            }
        )

        individual_explanation[
            "absolute_shap"
        ] = individual_explanation[
            "shap_value"
        ].abs()

        individual_explanation = (
            individual_explanation
            .sort_values(
                "absolute_shap",
                ascending=False,
            )
            .head(20)
        )

        individual_path = (
            self.output_dir
            / "individual_explanation.csv"
        )

        individual_explanation.to_csv(
            individual_path,
            index=False,
        )

        # --------------------------------------------------
        # Create report
        # --------------------------------------------------

        report = {
            "model": "XGBoost",
            "model_path": str(
                self.model_path
            ),
            "test_dataset": str(
                self.test_path
            ),
            "total_test_observations": int(
                len(test)
            ),
            "shap_sample_size": int(
                len(X_sample)
            ),
            "shap_positive_observations": int(
                (y_sample == 1).sum()
            ),
            "shap_negative_observations": int(
                (y_sample == 0).sum()
            ),
            "number_of_features": len(
                feature_columns
            ),
            "top_20_features": top_features,
            "explanation_type": (
                "TreeSHAP global feature "
                "importance plus individual "
                "observation explanation"
            ),
            "test_predictions_used_for_model_training": False,
            "data_leakage_prevented": True,
            "outputs": {
                "shap_values": str(
                    SHAP_VALUES_PATH
                ),
                "global_feature_importance": str(
                    importance_path
                ),
                "individual_explanation": str(
                    individual_path
                ),
            },
        }

        with REPORT_PATH.open(
            "w",
            encoding="utf-8",
        ) as report_file:

            json.dump(
                report,
                report_file,
                indent=4,
            )

        print("=" * 65)
        print("SHAP ANALYSIS COMPLETE")
        print("=" * 65)

        print(
            f"Top feature: "
            f"{top_features[0]['feature']}"
        )

        print(
            f"SHAP report saved to: "
            f"{REPORT_PATH.resolve()}"
        )

        print(
            f"Global importance saved to: "
            f"{importance_path.resolve()}"
        )

        print(
            f"Individual explanation saved to: "
            f"{individual_path.resolve()}"
        )

        return report


def main() -> None:
    """Run SHAP analysis."""

    analyzer = SHAPAnalyzer()

    analyzer.run()


if __name__ == "__main__":
    main()