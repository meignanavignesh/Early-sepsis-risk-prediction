"""
New-patient sepsis prediction.
This module accepts clinical observations from a NEW patient,
recreates the project's temporal features, applies the same
training-derived missing-value logic, and sends the resulting
196 features to the trained XGBoost model.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.feature_engineering.temporal_features import (
    TemporalFeatureEngineer,
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    Path("models/xgboost_optimized.pkl")
    if Path("models/xgboost_optimized.pkl").exists()
    else Path("models/xgboost.pkl")
)

TRAIN_MODEL_READY_PATH = Path(
    "data/processed/model_ready/train.csv"
)


# ============================================================
# FINAL DECISION SETTINGS
# ============================================================

THRESHOLD = 0.60
REQUIRED_CONSECUTIVE = 2


# ============================================================
# IDENTIFIERS
# ============================================================

IDENTIFIER_COLUMNS = [
    "Patient_ID",
    "Dataset",
]

TARGET_COLUMN = "SepsisLabel"


# ============================================================
# USER INPUT COLUMNS
# ============================================================

# These are the fields we are asking the user to enter
# for the current demonstration sequence.
#
# ICULOS is generated automatically.
# SepsisLabel is the prediction target and is not entered.

USER_INPUT_COLUMNS = [
    "HR",
    "O2Sat",
    "SBP",
    "MAP",
    "Resp",
    "BaseExcess",
    "FiO2",
    "pH",
    "PaCO2",
    "Age",
    "Gender",
    "HospAdmTime",
]


# ============================================================
# HELPER
# ============================================================

def consecutive_high_risk(
    probabilities,
    threshold=THRESHOLD,
    required=REQUIRED_CONSECUTIVE,
):
    """
    Determine whether the patient has the required number
    of consecutive high-risk observations.
    """

    count = 0

    for probability in probabilities:

        if probability >= threshold:

            count += 1

            if count >= required:
                return True

        else:

            count = 0

    return False


# ============================================================
# PREDICTOR
# ============================================================

class NewPatientPredictor:

    def __init__(
        self,
        model_path=MODEL_PATH,
        training_path=TRAIN_MODEL_READY_PATH,
    ):

        self.model_path = Path(
            model_path
        )

        self.training_path = Path(
            training_path
        )

        self.model = None

        self.training_data = None

        self.feature_columns = []

        self.training_medians = {}

        self.missing_indicator_columns = []

        self.temporal_engineer = (
            TemporalFeatureEngineer()
        )


    # --------------------------------------------------------
    # LOAD MODEL AND TRAINING INFORMATION
    # --------------------------------------------------------

    def load(self):

        print(
            "Loading trained XGBoost model..."
        )

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        if not self.training_path.exists():

            raise FileNotFoundError(
                f"Training dataset not found: "
                f"{self.training_path}"
            )

        self.model = joblib.load(
            self.model_path
        )

        print(
            "Model loaded successfully."
        )

        # ----------------------------------------------------
        # Load processed training data.
        # ----------------------------------------------------

        self.training_data = pd.read_csv(
            self.training_path
        )

        # ----------------------------------------------------
        # Get exact model feature columns.
        # ----------------------------------------------------

        self.feature_columns = [
            column
            for column in self.training_data.columns
            if column not in IDENTIFIER_COLUMNS
            and column != TARGET_COLUMN
        ]

        # ----------------------------------------------------
        # Find missing-indicator columns.
        # ----------------------------------------------------

        self.missing_indicator_columns = [
            column
            for column in self.feature_columns
            if column.endswith(
                "_was_missing"
            )
        ]

        # ----------------------------------------------------
        # Calculate training medians.
        # ----------------------------------------------------

        for column in self.feature_columns:

            if column in self.missing_indicator_columns:
                continue

            if pd.api.types.is_numeric_dtype(
                self.training_data[column]
            ):

                median = (
                    self.training_data[column]
                    .median()
                )

                if pd.isna(median):

                    raise ValueError(
                        f"No valid training median "
                        f"for feature: {column}"
                    )

                self.training_medians[
                    column
                ] = float(median)

        print(
            f"Model features: "
            f"{len(self.feature_columns)}"
        )

        print(
            f"Missing indicators: "
            f"{len(self.missing_indicator_columns)}"
        )

        print(
            f"Expected model features: "
            f"{self.model.n_features_in_}"
        )

        if (
            len(self.feature_columns)
            != self.model.n_features_in_
        ):

            raise ValueError(
                "Model feature mismatch. "
                f"Training data contains "
                f"{len(self.feature_columns)} "
                f"features but model expects "
                f"{self.model.n_features_in_}."
            )


    # --------------------------------------------------------
    # CREATE RAW OBSERVATION
    # --------------------------------------------------------

    def create_observation(
        self,
        patient_id,
        iculos,
        values,
    ):
        """
        Create one raw ICU observation.
        """

        observation = {
            "Patient_ID": patient_id,
            "Dataset": "USER_INPUT",
            "ICULOS": iculos,
        }

        # ----------------------------------------------------
        # Add available user-input fields.
        # Missing fields are kept as None.
        # ----------------------------------------------------

        for column in USER_INPUT_COLUMNS:

            value = values.get(
                column,
                None,
            )

            if value == "":
                value = None

            observation[column] = value

        # ----------------------------------------------------
        # Target does not exist for a new patient.
        # ----------------------------------------------------

        return observation


    # --------------------------------------------------------
    # BUILD TEMPORAL FEATURES
    # --------------------------------------------------------

    def build_features(
        self,
        observations,
    ):
        """
        Convert raw observations into the same temporal
        feature structure used during training.
        """

        dataframe = pd.DataFrame(
            observations
        )

        # ----------------------------------------------------
        # Convert clinical values to numeric.
        # ----------------------------------------------------

        for column in USER_INPUT_COLUMNS:

            if column in dataframe.columns:

                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

        # ----------------------------------------------------
        # Ensure columns required by the temporal
        # feature engineering module exist.
        # ----------------------------------------------------

        required_temporal_columns = [
            "Patient_ID",
            "Dataset",
            "Age",
            "Gender",
            "Unit1",
            "Unit2",
            "ICULOS",
            "HR",
            "O2Sat",
            "Temp",
            "SBP",
            "MAP",
            "DBP",
            "Resp",
            "SepsisLabel",
        ]

        for column in required_temporal_columns:

            if column not in dataframe.columns:

                dataframe[column] = None

        # ----------------------------------------------------
        # Target is unknown for a new patient.
        # ----------------------------------------------------

        dataframe[
            TARGET_COLUMN
        ] = 0

        # ----------------------------------------------------
        # Run the same temporal feature engineering
        # used during model training.
        # ----------------------------------------------------

        engineered = (
            self.temporal_engineer
            .create_temporal_features(
                dataframe
            )
        )

        return engineered


    # --------------------------------------------------------
    # APPLY MISSING INDICATORS
    # --------------------------------------------------------

    def apply_missing_indicators(
        self,
        dataframe,
    ):
        """
        Create the training-defined _was_missing indicators.
        """

        result = dataframe.copy()

        for indicator_name in (
            self.missing_indicator_columns
        ):

            original_column = (
                indicator_name.replace(
                    "_was_missing",
                    "",
                )
            )

            if (
                original_column
                in result.columns
            ):

                result[
                    indicator_name
                ] = (
                    result[
                        original_column
                    ]
                    .isna()
                    .astype("int8")
                )

            else:

                # Treat unavailable features as missing.

                result[
                    indicator_name
                ] = 1

        return result


    # --------------------------------------------------------
    # APPLY TRAINING MEDIANS
    # --------------------------------------------------------

    def apply_training_medians(
        self,
        dataframe,
    ):

        result = dataframe.copy()

        for column in self.feature_columns:

            if column in (
                self.missing_indicator_columns
            ):
                continue

            if column not in result.columns:

                result[column] = None

            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

            if column in self.training_medians:

                result[column] = (
                    result[column]
                    .fillna(
                        self.training_medians[
                            column
                        ]
                    )
                )

        return result


    # --------------------------------------------------------
    # ALIGN EXACT MODEL FEATURES
    # --------------------------------------------------------

    def align_model_features(
        self,
        dataframe,
    ):

        result = dataframe.copy()

        # Add any model features not supplied by the user.
        for column in self.feature_columns:

            if column not in result.columns:

                result[column] = (
                    self.training_medians.get(
                        column,
                        0.0,
                    )
                )

        # Keep EXACT model feature order.
        result = result[
            self.feature_columns
        ]

        return result


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    def predict(
        self,
        observations,
    ):
        """
        Predict sepsis risk for a NEW patient.

        Multiple ICU observations are used so that
        temporal features can be calculated.
        """

        if self.model is None:

            self.load()

        if not observations:

            raise ValueError(
                "At least one patient observation "
                "is required."
            )

        # ----------------------------------------------------
        # Build temporal features.
        # ----------------------------------------------------

        engineered = self.build_features(
            observations
        )

        # ----------------------------------------------------
        # Apply missing indicators.
        # ----------------------------------------------------

        processed = (
            self.apply_missing_indicators(
                engineered
            )
        )

        # ----------------------------------------------------
        # Apply training median imputation.
        # ----------------------------------------------------

        processed = (
            self.apply_training_medians(
                processed
            )
        )

        # ----------------------------------------------------
        # Align exact 196 model features.
        # ----------------------------------------------------

        X = self.align_model_features(
            processed
        )

        # ----------------------------------------------------
        # Final feature-count safety check.
        # ----------------------------------------------------

        if X.shape[1] != (
            self.model.n_features_in_
        ):

            raise ValueError(
                "Final feature count mismatch. "
                f"Generated {X.shape[1]}, "
                f"model expects "
                f"{self.model.n_features_in_}."
            )

        # ----------------------------------------------------
        # Generate probabilities.
        # ----------------------------------------------------

        probabilities = (
            self.model
            .predict_proba(X)[:, 1]
        )

        # ----------------------------------------------------
        # Apply consecutive-observation rule.
        # ----------------------------------------------------

        high_risk = (
            consecutive_high_risk(
                probabilities,
                THRESHOLD,
                REQUIRED_CONSECUTIVE,
            )
        )

        max_probability = float(
            probabilities.max()
        )

        # ----------------------------------------------------
        # Create result.
        # ----------------------------------------------------

        result = {
            "probabilities": [
                float(value)
                for value in probabilities
            ],
            "maximum_probability": (
                max_probability
            ),
            "threshold": THRESHOLD,
            "required_consecutive": (
                REQUIRED_CONSECUTIVE
            ),
            "high_sepsis_risk": bool(
                high_risk
            ),
        }

        # ----------------------------------------------------
        # DISPLAY RESULTS.
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("NEW PATIENT SEPSIS PREDICTION")
        print("=" * 70)

        print(
            f"Observations received: "
            f"{len(observations)}"
        )

        print(
            f"Maximum sepsis probability: "
            f"{max_probability:.4f}"
        )

        print(
            f"Threshold: "
            f"{THRESHOLD:.2f}"
        )

        print(
            f"Required consecutive observations: "
            f"{REQUIRED_CONSECUTIVE}"
        )

        print()

        if high_risk:

            print(
                "PREDICTION: HIGH SEPSIS RISK"
            )

            print(
                "Required consecutive high-risk "
                "observations were detected."
            )

        else:

            print(
                "PREDICTION: "
                "NO HIGH SEPSIS RISK DETECTED"
            )

            print(
                "The temporal decision rule "
                "was not satisfied."
            )

        print()

        print(
            "OBSERVATION PROBABILITIES"
        )

        for index, probability in enumerate(
            probabilities,
            start=1,
        ):

            print(
                f"Observation {index}: "
                f"{probability:.4f}"
            )

        print("=" * 70)

        return result


# ============================================================
# DEMONSTRATION
# ============================================================

def main():

    predictor = NewPatientPredictor()

    predictor.load()

    print()
    print("=" * 70)
    print("NEW PATIENT INPUT DEMONSTRATION")
    print("=" * 70)

    print(
        "Enter clinical values for consecutive ICU observations."
    )

    print(
        "Only the selected dataset-available variables "
        "are requested."
    )

    print(
        "Press ENTER for a missing value."
    )

    print()
    print("ICULOS is generated automatically.")
    print("SepsisLabel is NOT entered.")

    print()
    print("INPUT COLUMNS:")
    print(
        " -> ".join(USER_INPUT_COLUMNS)
    )

    # --------------------------------------------------------
    # ASK FOR NUMBER OF OBSERVATIONS
    # --------------------------------------------------------

    while True:

        try:

            number_of_observations = int(
                input(
                    "\nHow many ICU observations "
                    "do you want to enter? "
                ).strip()
            )

            if number_of_observations < 1:

                print(
                    "Please enter at least 1 observation."
                )

                continue

            break

        except ValueError:

            print(
                "Please enter a valid whole number."
            )

    observations = []

    # --------------------------------------------------------
    # ENTER OBSERVATIONS
    # --------------------------------------------------------

    for observation_number in range(
        1,
        number_of_observations + 1,
    ):

        print()
        print("=" * 70)
        print(
            f"OBSERVATION {observation_number}"
        )
        print("=" * 70)

        values = {}

        for column in USER_INPUT_COLUMNS:

            user_value = input(
                f"{column}: "
            ).strip()

            if user_value == "":

                values[column] = None

            else:

                values[column] = user_value

        # ----------------------------------------------------
        # CREATE OBSERVATION
        # ----------------------------------------------------

        observation = (
            predictor.create_observation(
                patient_id="NEW_PATIENT",
                iculos=observation_number,
                values=values,
            )
        )

        observations.append(
            observation
        )

    # --------------------------------------------------------
    # RUN PREDICTION
    # --------------------------------------------------------

    predictor.predict(
        observations
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
