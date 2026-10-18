"""Single-patient temporal sepsis prediction service."""

from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path(
    "models/xgboost.pkl"
)

TEST_DATA_PATH = Path(
    "data/processed/model_ready/test.csv"
)

THRESHOLD = 0.50
REQUIRED_CONSECUTIVE = 2

TARGET_COLUMN = "SepsisLabel"

IDENTIFIER_COLUMNS = {
    "Patient_ID",
    "Dataset",
}


def check_consecutive_risk(
    probabilities,
    threshold=THRESHOLD,
    required_consecutive=REQUIRED_CONSECUTIVE,
):
    """
    Check whether the patient has the required number
    of consecutive high-risk observations.
    """

    consecutive = 0

    for probability in probabilities:

        if probability >= threshold:

            consecutive += 1

            if consecutive >= required_consecutive:
                return True

        else:

            consecutive = 0

    return False


class SinglePatientPredictor:
    """Predict sepsis risk for one patient's ICU sequence."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        test_data_path=TEST_DATA_PATH,
        threshold=THRESHOLD,
        required_consecutive=REQUIRED_CONSECUTIVE,
    ):

        self.model_path = Path(
            model_path
        )

        self.test_data_path = Path(
            test_data_path
        )

        self.threshold = threshold

        self.required_consecutive = (
            required_consecutive
        )

        self.model = None

    def load_model(self):
        """Load the trained XGBoost model."""

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        print("Loading XGBoost model...")

        self.model = joblib.load(
            self.model_path
        )

        print(
            f"Model loaded successfully."
        )

        print(
            f"Expected features: "
            f"{self.model.n_features_in_}"
        )

    def load_patient(
        self,
        patient_id,
    ):
        """Load one patient from the model-ready dataset."""

        if not self.test_data_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: "
                f"{self.test_data_path}"
            )

        print(
            f"Loading patient: "
            f"{patient_id}"
        )

        # Read the test dataset.
        # This is only for demonstration/testing.
        dataframe = pd.read_csv(
            self.test_data_path
        )

        patient_data = dataframe[
            dataframe["Patient_ID"]
            == patient_id
        ].copy()

        if patient_data.empty:

            raise ValueError(
                f"Patient '{patient_id}' "
                f"was not found."
            )

        # Maintain ICU chronological order.
        if "ICULOS" in patient_data.columns:

            patient_data = patient_data.sort_values(
                by="ICULOS",
                kind="mergesort",
            )

        patient_data = (
            patient_data
            .reset_index(drop=True)
        )

        print(
            f"Patient observations: "
            f"{len(patient_data)}"
        )

        return patient_data

    def predict_patient(
        self,
        patient_id,
    ):
        """Generate a temporal prediction for one patient."""

        if self.model is None:
            self.load_model()

        patient_data = self.load_patient(
            patient_id
        )

        # --------------------------------------------------
        # Identify model features
        # --------------------------------------------------

        feature_columns = [
            column
            for column in patient_data.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        expected_features = (
            self.model.n_features_in_
        )

        if len(feature_columns) != expected_features:

            raise ValueError(
                "Feature count mismatch. "
                f"Model expects "
                f"{expected_features} features, "
                f"but patient data contains "
                f"{len(feature_columns)}."
            )

        # --------------------------------------------------
        # Generate observation probabilities
        # --------------------------------------------------

        X = patient_data[
            feature_columns
        ]

        probabilities = (
            self.model.predict_proba(X)[:, 1]
        )

        patient_data[
            "Sepsis_Probability"
        ] = probabilities

        # --------------------------------------------------
        # Apply observation threshold
        # --------------------------------------------------

        patient_data[
            "Above_Threshold"
        ] = (
            patient_data[
                "Sepsis_Probability"
            ]
            >= self.threshold
        ).astype(int)

        # --------------------------------------------------
        # Find consecutive high-risk observations
        # --------------------------------------------------

        consecutive = 0

        consecutive_found = False

        for probability in probabilities:

            if probability >= self.threshold:

                consecutive += 1

                if (
                    consecutive
                    >= self.required_consecutive
                ):

                    consecutive_found = True
                    break

            else:

                consecutive = 0

        # --------------------------------------------------
        # Patient-level prediction
        # --------------------------------------------------

        max_probability = float(
            probabilities.max()
        )

        patient_prediction = int(
            consecutive_found
        )

        if patient_prediction == 1:

            risk_level = "HIGH RISK"

        elif max_probability >= 0.30:

            risk_level = "MODERATE RISK"

        else:

            risk_level = "LOW RISK"

        # --------------------------------------------------
        # Actual label
        # --------------------------------------------------

        actual_label = None

        if TARGET_COLUMN in patient_data.columns:

            actual_label = int(
                patient_data[
                    TARGET_COLUMN
                ].max()
            )

        # --------------------------------------------------
        # Find highest-risk observations
        # --------------------------------------------------

        top_observations = (
            patient_data[
                [
                    column
                    for column in [
                        "ICULOS",
                        "Sepsis_Probability",
                        "Above_Threshold",
                    ]
                    if column
                    in patient_data.columns
                ]
            ]
            .sort_values(
                by="Sepsis_Probability",
                ascending=False,
            )
            .head(5)
        )

        # --------------------------------------------------
        # Print result
        # --------------------------------------------------

        print()
        print("=" * 70)
        print("SINGLE PATIENT SEPSIS PREDICTION")
        print("=" * 70)

        print(
            f"Patient ID: "
            f"{patient_id}"
        )

        print(
            f"ICU observations: "
            f"{len(patient_data)}"
        )

        print(
            f"Maximum probability: "
            f"{max_probability:.4f}"
        )

        print(
            f"Threshold: "
            f"{self.threshold:.2f}"
        )

        print(
            f"Required consecutive observations: "
            f"{self.required_consecutive}"
        )

        print()

        if patient_prediction:

            print(
                "PREDICTION: HIGH SEPSIS RISK"
            )

            print(
                "Reason: required consecutive "
                "high-risk observations detected."
            )

        else:

            print(
                "PREDICTION: NO HIGH SEPSIS RISK DETECTED"
            )

            print(
                "Reason: required consecutive "
                "high-risk observations were not detected."
            )

        print(
            f"Risk level: "
            f"{risk_level}"
        )

        if actual_label is not None:

            print()

            print(
                f"Actual dataset label: "
                f"{actual_label}"
            )

        print()
        print(
            "TOP HIGH-RISK OBSERVATIONS"
        )

        print(
            top_observations.to_string(
                index=False
            )
        )

        print(
            "=" * 70
        )

        return {
            "patient_id": patient_id,
            "observations": int(
                len(patient_data)
            ),
            "maximum_probability": (
                max_probability
            ),
            "threshold": float(
                self.threshold
            ),
            "required_consecutive": int(
                self.required_consecutive
            ),
            "prediction": int(
                patient_prediction
            ),
            "risk_level": risk_level,
            "actual_label": actual_label,
            "top_observations": (
                top_observations.to_dict(
                    orient="records"
                )
            ),
        }


def main():
    """Run a single-patient prediction example."""

    predictor = SinglePatientPredictor()

    predictor.load_model()

    # ------------------------------------------------------
    # Automatically select the first patient in the
    # unseen test set for demonstration.
    # ------------------------------------------------------

    data = pd.read_csv(
        TEST_DATA_PATH,
        usecols=["Patient_ID"],
    )

    patient_id = (
        data["Patient_ID"]
        .iloc[0]
    )

    predictor.predict_patient(
        patient_id
    )


if __name__ == "__main__":
    main()