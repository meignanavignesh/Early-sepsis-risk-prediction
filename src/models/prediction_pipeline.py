"""Final temporal prediction pipeline for sepsis risk prediction."""

from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.hourly_rules import add_hourly_risk_features


MODEL_PATH = (
    Path("models/xgboost_optimized.pkl")
    if Path("models/xgboost_optimized.pkl").exists()
    else Path("models/xgboost.pkl")
)

INPUT_PATH = Path(
    "data/processed/model_ready/test.csv"
)

OUTPUT_PATH = Path(
    "data/processed/final_predictions.csv"
)

# FINAL VALUES SELECTED USING VALIDATION DATA
THRESHOLD = 0.60
REQUIRED_CONSECUTIVE = 2

TARGET_COLUMN = "SepsisLabel"

IDENTIFIER_COLUMNS = {
    "Patient_ID",
    "Dataset",
}


def has_consecutive_positive(
    probabilities,
    threshold,
    required_consecutive,
):
    """Check whether risk crosses the threshold consecutively."""

    consecutive = 0

    for probability in probabilities:

        if probability >= threshold:

            consecutive += 1

            if consecutive >= required_consecutive:
                return 1

        else:
            consecutive = 0

    return 0


class SepsisPredictionPipeline:
    """Run the final temporal sepsis prediction pipeline."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        threshold=THRESHOLD,
        required_consecutive=REQUIRED_CONSECUTIVE,
    ):

        self.model_path = Path(model_path)
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

        self.threshold = threshold
        self.required_consecutive = (
            required_consecutive
        )

    def run(self):

        print("=" * 70)
        print("FINAL TEMPORAL SEPSIS PREDICTION PIPELINE")
        print("=" * 70)

        print(
            f"Threshold: {self.threshold}"
        )

        print(
            f"Required consecutive observations: "
            f"{self.required_consecutive}"
        )

        # --------------------------------------------------
        # 1. Check files
        # --------------------------------------------------

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        if not self.input_path.exists():

            raise FileNotFoundError(
                f"Input dataset not found: "
                f"{self.input_path}"
            )

        # --------------------------------------------------
        # 2. Load model
        # --------------------------------------------------

        print()
        print("Loading XGBoost model...")

        model = joblib.load(
            self.model_path
        )

        print(
            f"Model expects "
            f"{model.n_features_in_} features."
        )

        # --------------------------------------------------
        # 3. Load dataset
        # --------------------------------------------------

        print(
            "Loading model-ready dataset..."
        )

        dataframe = pd.read_csv(
            self.input_path
        )

        print(
            f"Input shape: "
            f"{dataframe.shape}"
        )

        # --------------------------------------------------
        # 4. Identify model features
        # --------------------------------------------------

        feature_columns = [
            column
            for column in dataframe.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        print(
            f"Model features found: "
            f"{len(feature_columns)}"
        )

        expected_features = (
            model.n_features_in_
        )

        if len(feature_columns) != expected_features:

            raise ValueError(
                "Feature count mismatch. "
                f"Model expects "
                f"{expected_features}, "
                f"but found "
                f"{len(feature_columns)}."
            )

        # --------------------------------------------------
        # 5. Generate observation probabilities
        # --------------------------------------------------

        print()
        print(
            "Generating XGBoost probabilities..."
        )

        X = dataframe[
            feature_columns
        ]

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        dataframe[
            "Sepsis_Probability"
        ] = probabilities

        # --------------------------------------------------
        # 6. Preserve chronological order
        # --------------------------------------------------

        if "ICULOS" in dataframe.columns:

            dataframe = dataframe.sort_values(
                by=[
                    "Patient_ID",
                    "ICULOS",
                ],
                kind="mergesort",
            )

        else:

            dataframe = dataframe.sort_values(
                by=[
                    "Patient_ID",
                ],
                kind="mergesort",
            )

        # --------------------------------------------------
        # 7. Observation-level threshold
        # --------------------------------------------------

        dataframe[
            "Above_Threshold"
        ] = (
            dataframe[
                "Sepsis_Probability"
            ]
            >= self.threshold
        ).astype(int)

        dataframe = add_hourly_risk_features(
            dataframe,
            probability_column="Sepsis_Probability",
        )

        # --------------------------------------------------
        # 8. Patient-level temporal decision
        # --------------------------------------------------

        print()
        print(
            "Applying temporal decision rule..."
        )

        patient_results = []

        for patient_id, group in dataframe.groupby(
            "Patient_ID",
            sort=False,
        ):

            probabilities_for_patient = (
                group[
                    "Sepsis_Probability"
                ]
                .astype(float)
                .tolist()
            )

            prediction = (
                has_consecutive_positive(
                    probabilities_for_patient,
                    self.threshold,
                    self.required_consecutive,
                )
            )

            max_probability = max(
                probabilities_for_patient
            )

            observations_above_threshold = sum(
                probability >= self.threshold
                for probability
                in probabilities_for_patient
            )

            patient_results.append(
                {
                    "Patient_ID": patient_id,

                    "Max_Sepsis_Probability": (
                        max_probability
                    ),

                    "Observations_Above_Threshold": (
                        observations_above_threshold
                    ),

                    "Patient_Sepsis_Prediction": (
                        prediction
                    ),

                    "Total_Observations": (
                        len(
                            probabilities_for_patient
                        )
                    ),
                }
            )

        patient_results = pd.DataFrame(
            patient_results
        )

        # --------------------------------------------------
        # 9. Merge patient decisions back
        # --------------------------------------------------

        dataframe = dataframe.merge(
            patient_results[
                [
                    "Patient_ID",
                    "Patient_Sepsis_Prediction",
                ]
            ],
            on="Patient_ID",
            how="left",
        )

        # --------------------------------------------------
        # 10. Save observation-level output
        # --------------------------------------------------

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            self.output_path,
            index=False,
        )

        # --------------------------------------------------
        # 11. Save patient-level output
        # --------------------------------------------------

        patient_output_path = (
            self.output_path.parent
            / "final_patient_predictions.csv"
        )

        patient_results.to_csv(
            patient_output_path,
            index=False,
        )

        # --------------------------------------------------
        # 12. Summary
        # --------------------------------------------------

        total_observations = len(
            dataframe
        )

        total_patients = len(
            patient_results
        )

        predicted_sepsis_patients = int(
            patient_results[
                "Patient_Sepsis_Prediction"
            ].sum()
        )

        predicted_non_sepsis_patients = (
            total_patients
            - predicted_sepsis_patients
        )

        # --------------------------------------------------
        # 13. Print result
        # --------------------------------------------------

        print()
        print("=" * 70)
        print("FINAL TEMPORAL PREDICTION PIPELINE COMPLETE")
        print("=" * 70)

        print(
            f"Observations processed: "
            f"{total_observations}"
        )

        print(
            f"Patients processed: "
            f"{total_patients}"
        )

        print(
            f"Final threshold: "
            f"{self.threshold}"
        )

        print(
            f"Consecutive observations required: "
            f"{self.required_consecutive}"
        )

        print(
            f"Predicted sepsis-risk patients: "
            f"{predicted_sepsis_patients}"
        )

        print(
            f"Predicted non-sepsis-risk patients: "
            f"{predicted_non_sepsis_patients}"
        )

        print()
        print(
            f"Observation predictions saved to: "
            f"{self.output_path.resolve()}"
        )

        print(
            f"Patient predictions saved to: "
            f"{patient_output_path.resolve()}"
        )

        return {
            "observations_processed": (
                int(total_observations)
            ),
            "patients_processed": (
                int(total_patients)
            ),
            "threshold": float(
                self.threshold
            ),
            "required_consecutive": int(
                self.required_consecutive
            ),
            "predicted_sepsis_patients": (
                predicted_sepsis_patients
            ),
            "predicted_non_sepsis_patients": (
                predicted_non_sepsis_patients
            ),
            "observation_output": str(
                self.output_path
            ),
            "patient_output": str(
                patient_output_path
            ),
        }


def main():
    """Run the prediction pipeline."""

    pipeline = (
        SepsisPredictionPipeline()
    )

    pipeline.run()


if __name__ == "__main__":
    main()
