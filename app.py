"""Streamlit app for hourly sepsis risk prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import joblib
import streamlit as st
import streamlit.components.v1 as components

from src.models.user_prediction import NewPatientPredictor


MODEL_PATH = (
    Path("models/xgboost_optimized.pkl")
    if Path("models/xgboost_optimized.pkl").exists()
    else Path("models/xgboost.pkl")
)
FINAL_PREDICTIONS_PATH = Path("data/processed/final_predictions.csv")
FINAL_PATIENT_PREDICTIONS_PATH = Path(
    "data/processed/final_patient_predictions.csv"
)
TEST_DATA_PATH = Path("data/processed/model_ready/test.csv")
GLOBAL_SHAP_PATH = Path("reports/shap/global_feature_importance.csv")
INDIVIDUAL_SHAP_PATH = Path("reports/shap/individual_explanation.csv")
TARGET_COLUMN = "SepsisLabel"
IDENTIFIER_COLUMNS = {"Patient_ID", "Dataset"}
# Balanced operating point selected from validation data: improves precision
# while retaining more recall than higher precision-only thresholds.
ALERT_THRESHOLD = 0.60
HIGH_RISK_THRESHOLD = 0.70

REQUIRED_COLUMNS = [
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

CLINICAL_FEATURES = {
    "HR": "Heart rate",
    "O2Sat": "Oxygen saturation",
    "Temp": "Body temperature",
    "SBP": "Systolic blood pressure",
    "MAP": "Mean arterial pressure",
    "DBP": "Diastolic blood pressure",
    "Resp": "Respiratory rate",
    "EtCO2": "End-tidal carbon dioxide",
    "Lactate": "Lactate",
    "WBC": "White blood cell count",
    "Creatinine": "Creatinine / kidney function",
    "BUN": "Blood urea nitrogen",
    "Glucose": "Glucose",
    "pH": "Blood pH",
    "PaCO2": "Arterial carbon dioxide",
    "FiO2": "Inspired oxygen fraction",
    "Platelets": "Platelet count",
    "Hgb": "Hemoglobin",
    "Hct": "Hematocrit",
}


def main() -> None:
    """Run the Streamlit prediction interface."""

    st.set_page_config(
        page_title="Early Sepsis Risk Prediction",
        layout="wide",
    )

    st.title("Early Sepsis Risk Prediction")

    mode = st.radio(
        "Prediction source",
        [
            "Use existing test dataset",
            "View all dataset splits",
            "Upload new patient CSV",
        ],
        horizontal=True,
    )

    if mode == "Use existing test dataset":
        run_existing_dataset_view()
    elif mode == "View all dataset splits":
        run_all_splits_view()
    else:
        run_upload_view()


@st.cache_data(show_spinner=False)
def load_split_predictions(split: str, model_path: str) -> pd.DataFrame:
    """Load one split and cache predictions for responsive patient selection."""

    split_path = Path("data/processed/model_ready") / f"{split}.csv"
    data = pd.read_csv(split_path)
    feature_columns = [
        column
        for column in data.columns
        if column not in IDENTIFIER_COLUMNS and column != TARGET_COLUMN
    ]
    model = joblib.load(model_path)
    data["Sepsis_Probability"] = model.predict_proba(
        data[feature_columns]
    )[:, 1]
    data["Risk_Level"] = pd.cut(
        data["Sepsis_Probability"],
        bins=[-1, 0.3, HIGH_RISK_THRESHOLD, 1.0],
        labels=["LOW", "MODERATE", "HIGH"],
    ).astype(str)
    data["Alert"] = (
        data["Sepsis_Probability"] >= ALERT_THRESHOLD
    ).astype(int)
    return data


def run_all_splits_view() -> None:
    """Generate predictions for any prepared dataset split."""

    split = st.selectbox(
        "Choose dataset split",
        ["train", "validation", "test"],
        format_func=lambda value: value.title(),
    )
    split_path = Path("data/processed/model_ready") / f"{split}.csv"

    if not split_path.exists():
        st.error(f"Dataset split not found: {split_path}")
        return

    st.info(
        "These are model predictions for the selected split. "
        "Use the Test split for unbiased performance reporting."
    )

    load_requested = st.button("Load all patients", type="primary")
    if load_requested:
        st.session_state["loaded_prediction_split"] = split

    if st.session_state.get("loaded_prediction_split") != split:
        return

    with st.spinner(f"Loading {split} patients and generating predictions..."):
        data = load_split_predictions(split, str(MODEL_PATH))

    patient_ids = data["Patient_ID"].drop_duplicates().tolist()
    st.success(
        f"Loaded {len(patient_ids):,} patients and "
        f"{len(data):,} hourly observations from the {split} split."
    )
    selected_patient = st.selectbox(
        "Select patient",
        patient_ids,
        key=f"all_split_patient_{split}",
    )
    patient_data = data[data["Patient_ID"] == selected_patient].copy()
    if "ICULOS" in patient_data.columns:
        patient_data = patient_data.sort_values("ICULOS", kind="mergesort")
    patient_data = patient_data.reset_index(drop=True)

    st.subheader(f"Patient Records: {selected_patient}")
    st.dataframe(patient_data, use_container_width=True, height=360)
    display_prediction_output(patient_data)

    max_risk_index = patient_data["Sepsis_Probability"].astype(float).idxmax()
    max_risk_hour = patient_data.loc[max_risk_index, "ICULOS"]
    max_risk_probability = float(
        patient_data.loc[max_risk_index, "Sepsis_Probability"]
    )
    st.subheader("Why This Risk Score?")
    st.caption(
        f"SHAP-ranked explanation for {selected_patient}'s "
        f"highest-risk hour in the {split} split."
    )
    display_saved_shap_reason(
        patient_data.loc[[max_risk_index]],
        max_risk_probability,
        max_risk_hour,
    )


def run_existing_dataset_view() -> None:
    """Show predictions already generated for the test dataset."""

    if not FINAL_PREDICTIONS_PATH.exists():
        st.warning(
            "Existing prediction output was not found. Run dataset "
            "prediction first:"
        )
        st.code(
            ".\\.venv\\Scripts\\python.exe -m src.models.prediction_pipeline",
            language="powershell",
        )
        return

    predictions = pd.read_csv(FINAL_PREDICTIONS_PATH)

    if "Patient_ID" not in predictions.columns:
        st.error("Prediction file does not contain Patient_ID.")
        return

    patient_ids = predictions["Patient_ID"].drop_duplicates().tolist()
    st.caption(
        f"{len(patient_ids):,} patient records available. "
        "Choose one patient below to view hourly observations."
    )
    selected_patient = st.selectbox(
        "Select patient",
        patient_ids,
        index=0,
    )

    patient_data = predictions[
        predictions["Patient_ID"] == selected_patient
    ].copy()

    if "ICULOS" in patient_data.columns:
        patient_data = patient_data.sort_values(
            by="ICULOS",
            kind="mergesort",
        )
    patient_data = patient_data.reset_index(drop=True)

    st.subheader(f"Patient Records: {selected_patient}")
    st.dataframe(
        patient_data,
        use_container_width=True,
        height=360,
    )

    st.subheader("Patient Summary")
    summary = summarize_existing_patient(patient_data)
    st.json(summary)

    display_prediction_output(patient_data)
    display_existing_patient_reason(selected_patient, patient_data)


def validate_uploaded_data(dataframe: pd.DataFrame) -> bool:
    """Show safe input checks before running predictions on an uploaded file."""
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.info("Add the missing columns and upload the CSV again.")
        return False
    if dataframe.empty:
        st.error("The uploaded CSV contains no observations.")
        return False
    if "Patient_ID" in dataframe.columns and "ICULOS" in dataframe.columns:
        duplicates = dataframe.duplicated(["Patient_ID", "ICULOS"]).sum()
        if duplicates:
            st.warning(f"{duplicates} duplicate patient-hour rows found. They will be removed before prediction.")
    numeric_columns = [column for column in REQUIRED_COLUMNS if column != "Gender"]
    invalid_counts = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce").isna().sum()
    invalid_counts = invalid_counts[invalid_counts > 0]
    if not invalid_counts.empty:
        st.warning("Some clinical values are missing or non-numeric; the preprocessing pipeline will handle them.")
    st.success(f"Input validation passed: {len(dataframe):,} hourly observations.")
    return True


def run_upload_view() -> None:
    """Upload a new patient CSV and run model prediction."""

    uploaded = st.file_uploader(
        "Upload patient observation CSV",
        type=["csv"],
    )

    if uploaded is None:
        st.info(
            "Upload a CSV containing hourly observations. "
            "Patient_ID and ICULOS are optional; missing values are allowed."
        )
        st.dataframe(
            pd.DataFrame(columns=["Patient_ID", "ICULOS", *REQUIRED_COLUMNS])
        )
        return

    input_data = pd.read_csv(uploaded)
    st.subheader("Uploaded Data")
    st.dataframe(input_data.head(100))

    if st.button("Run Prediction", type="primary"):
        if not validate_uploaded_data(input_data):
            return
        if "Patient_ID" in input_data.columns and "ICULOS" in input_data.columns:
            input_data = input_data.drop_duplicates(["Patient_ID", "ICULOS"], keep="last")
        predictor = NewPatientPredictor()
        predictor.load()

        prepared = prepare_observations(input_data)
        all_results = []

        for patient_id, group in prepared.groupby("Patient_ID", sort=False):
            observations = group.to_dict(orient="records")
            result = predictor.predict(observations)

            patient_output = group[["Patient_ID", "ICULOS"]].copy()
            patient_output["Sepsis_Probability"] = result[
                "probabilities"
            ]
            patient_output["Risk_Level"] = pd.cut(
                patient_output["Sepsis_Probability"],
                bins=[-1, 0.3, 0.5, 1.0],
                labels=["LOW", "MODERATE", "HIGH"],
            ).astype(str)
            patient_output["Alert"] = (
                patient_output["Sepsis_Probability"] >= ALERT_THRESHOLD
            ).astype(int)
            patient_output["Patient_High_Risk"] = result[
                "high_sepsis_risk"
            ]
            all_results.append(patient_output)

        output = pd.concat(all_results, ignore_index=True)
        display_prediction_output(output)
        display_uploaded_patient_reason(
            predictor,
            prepared,
            output,
        )

        csv_bytes = output.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Export Prediction Report",
            data=csv_bytes,
            file_name="sepsis_prediction_report.csv",
            mime="text/csv",
        )


def summarize_existing_patient(patient_data: pd.DataFrame) -> dict:
    """Create a compact summary for one existing test patient."""

    probability = patient_data["Sepsis_Probability"].astype(float)
    actual_label = None

    if "SepsisLabel" in patient_data.columns:
        actual_label = int(patient_data["SepsisLabel"].max())

    patient_prediction = None
    if "Patient_Sepsis_Prediction" in patient_data.columns:
        patient_prediction = int(
            patient_data["Patient_Sepsis_Prediction"].max()
        )
    else:
        patient_prediction = int((probability >= ALERT_THRESHOLD).any())

    first_alert_hour = None
    alert_rows = patient_data[probability >= ALERT_THRESHOLD]
    if not alert_rows.empty and "ICULOS" in alert_rows.columns:
        first_alert_hour = float(alert_rows["ICULOS"].iloc[0])

    return {
        "observations": int(len(patient_data)),
        "actual_label": actual_label,
        "predicted_high_risk": patient_prediction,
        "max_probability": float(probability.max()),
        "mean_probability": float(probability.mean()),
        "first_alert_hour": first_alert_hour,
        "alert_hours": int((probability >= ALERT_THRESHOLD).sum()),
    }


def display_prediction_output(output: pd.DataFrame) -> None:
    """Display hourly risk table, chart, top alerts, and export."""

    output = output.copy()
    probability = output["Sepsis_Probability"].astype(float)
    output["Alert_Severity"] = pd.cut(
        probability,
        bins=[-1, ALERT_THRESHOLD, HIGH_RISK_THRESHOLD, 0.85, 1.0],
        labels=["Routine", "Alert", "High", "Critical"],
    ).astype(str)

    high_risk_rows = output[
        output["Sepsis_Probability"] >= HIGH_RISK_THRESHOLD
    ]
    if not high_risk_rows.empty:
        peak_index = high_risk_rows["Sepsis_Probability"].astype(float).idxmax()
        peak = high_risk_rows.loc[peak_index]
        patient_text = (
            f" for patient `{peak['Patient_ID']}`"
            if "Patient_ID" in high_risk_rows.columns
            else ""
        )
        hour_text = (
            f" at ICU hour `{peak['ICULOS']}`"
            if "ICULOS" in high_risk_rows.columns
            else ""
        )
        alert_key = "high_risk_ack_" + "_".join(
            sorted(high_risk_rows["Patient_ID"].astype(str).unique())
        )
        if st.session_state.get(alert_key, False):
            st.info("High-risk alert acknowledged for the current patient view.")
        else:
            st.error(
                f"HIGH SEPSIS RISK{patient_text}{hour_text}: "
                f"probability `{float(peak['Sepsis_Probability']):.1%}` "
                f"(threshold `{HIGH_RISK_THRESHOLD:.0%}`). "
                "Clinical review is recommended."
            )
            components.html(
                """
                <script>
                (() => {
                  try {
                    const AudioContext = window.AudioContext || window.webkitAudioContext;
                    const context = new AudioContext();
                    const beep = () => {
                      const oscillator = context.createOscillator();
                      const gain = context.createGain();
                      oscillator.type = 'sine';
                      oscillator.frequency.setValueAtTime(880, context.currentTime);
                      gain.gain.setValueAtTime(0.08, context.currentTime);
                      oscillator.connect(gain);
                      gain.connect(context.destination);
                      oscillator.start();
                      oscillator.stop(context.currentTime + 0.45);
                    };
                    beep();
                    const alarm = window.setInterval(beep, 2000);
                    window.setTimeout(() => window.clearInterval(alarm), 15000);
                  } catch (error) {
                    // The visible Streamlit alert remains if autoplay is blocked.
                  }
                })();
                </script>
                """,
                height=0,
            )
            if st.button("Acknowledge high-risk alert", key=alert_key):
                st.session_state[alert_key] = True
                st.rerun()
        st.caption(
            f"High-risk observations detected: {len(high_risk_rows)}. "
            "This is a decision-support alert, not a diagnosis."
        )
    else:
        st.success(
            f"No high-risk observation reached the {HIGH_RISK_THRESHOLD:.0%} "
            "display threshold in the current view."
        )

    st.subheader("Hourly Risk Scores")

    preferred_columns = [
        column
        for column in [
            "Patient_ID",
            "ICULOS",
            "Sepsis_Probability",
            "Risk_Level",
            "Alert_Severity",
            "Above_Threshold",
            "Alert",
            "Patient_Sepsis_Prediction",
            "Patient_High_Risk",
            "SepsisLabel",
            "Risk_Mean_3h",
            "Risk_Max_6h",
            "Risk_Trend_3h",
        ]
        if column in output.columns
    ]

    ordered_output = output.copy()
    sort_columns = [column for column in ["Patient_ID", "ICULOS"] if column in ordered_output.columns]
    if sort_columns:
        ordered_output = ordered_output.sort_values(sort_columns, kind="mergesort")
    st.dataframe(ordered_output[preferred_columns].head(500), use_container_width=True)

    window_columns = [
        column
        for column in [
            "Patient_ID",
            "ICULOS",
            "Sepsis_Probability",
            "Risk_Mean_3h",
            "Risk_Max_3h",
            "Risk_Trend_3h",
            "Risk_Mean_6h",
            "Risk_Max_6h",
            "Risk_Trend_6h",
            "Risk_Mean_12h",
            "Risk_Max_12h",
            "Risk_Trend_12h",
        ]
        if column in ordered_output.columns
    ]
    if len(window_columns) > 3:
        st.subheader("Hourly Window Analysis")
        st.caption(
            "Each row is one ICU hour. The 3-hour, 6-hour, and 12-hour "
            "columns summarize recent risk history for that patient."
        )
        st.dataframe(
            ordered_output[window_columns].head(500),
            use_container_width=True,
        )

    st.subheader("Risk Trend")
    for patient_id, group in output.groupby("Patient_ID", sort=False):
        if "ICULOS" in group.columns:
            chart = group.set_index("ICULOS")[["Sepsis_Probability"]]
        else:
            chart = group[["Sepsis_Probability"]]

        st.caption(f"Patient {patient_id}")
        st.line_chart(chart)

    high_risk = output[
        output["Sepsis_Probability"] >= HIGH_RISK_THRESHOLD
    ].sort_values(
        by="Sepsis_Probability",
        ascending=False,
    )

    st.subheader("Top High-Risk Observations")
    st.dataframe(high_risk[preferred_columns].head(20))

    st.subheader("High-Risk Alert Summary")
    if high_risk.empty:
        st.info("No high-risk patients are present in the current view.")
    else:
        summary = (
            high_risk.sort_values(["Patient_ID", "ICULOS"])
            .groupby("Patient_ID", as_index=False)
            .agg(
                First_High_Risk_Hour=("ICULOS", "first"),
                High_Risk_Hours=("ICULOS", "size"),
                Maximum_Risk=("Sepsis_Probability", "max"),
            )
            .sort_values("Maximum_Risk", ascending=False)
        )
        summary["Priority"] = pd.cut(
            summary["Maximum_Risk"].astype(float),
            bins=[-1, HIGH_RISK_THRESHOLD, 0.85, 1.0],
            labels=["High", "Very high", "Critical"],
        ).astype(str)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    csv_bytes = output.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Current View",
        data=csv_bytes,
        file_name="sepsis_prediction_view.csv",
        mime="text/csv",
    )


def display_existing_patient_reason(
    patient_id: str,
    patient_data: pd.DataFrame,
) -> None:
    """Explain the max-risk hour for an existing dataset patient."""

    if not TEST_DATA_PATH.exists():
        st.warning("Model-ready test data is unavailable for SHAP reasons.")
        return

    max_risk_index = patient_data[
        "Sepsis_Probability"
    ].astype(float).idxmax()
    max_risk_hour = patient_data.loc[max_risk_index, "ICULOS"]

    patient_rows = pd.read_csv(TEST_DATA_PATH)
    patient_rows = patient_rows[
        patient_rows["Patient_ID"] == patient_id
    ].copy()

    if "ICULOS" in patient_rows.columns:
        selected_rows = patient_rows[
            patient_rows["ICULOS"] == max_risk_hour
        ]
    else:
        selected_rows = patient_rows.iloc[[0]]

    if selected_rows.empty:
        st.warning("Could not find the selected patient's max-risk hour.")
        return

    row = selected_rows.iloc[[0]]
    probability = float(patient_data.loc[max_risk_index, "Sepsis_Probability"])

    st.subheader("Why This Risk Score?")
    st.caption(
        "SHAP explanation for the patient's highest-risk ICU hour."
    )
    display_saved_shap_reason(row, probability, max_risk_hour)


def display_uploaded_patient_reason(
    predictor: NewPatientPredictor,
    prepared: pd.DataFrame,
    output: pd.DataFrame,
) -> None:
    """Explain the max-risk uploaded patient observation."""

    if output.empty:
        return

    max_risk_index = output[
        "Sepsis_Probability"
    ].astype(float).idxmax()
    patient_id = output.loc[max_risk_index, "Patient_ID"]
    max_risk_hour = output.loc[max_risk_index, "ICULOS"]

    observations = prepared[
        prepared["Patient_ID"] == patient_id
    ].to_dict(orient="records")

    engineered = predictor.build_features(observations)
    processed = predictor.apply_missing_indicators(engineered)
    processed = predictor.apply_training_medians(processed)
    model_ready = predictor.align_model_features(processed)

    if "ICULOS" in engineered.columns:
        row_position = (
            engineered["ICULOS"].reset_index(drop=True) == max_risk_hour
        )
        row = model_ready.loc[row_position.values].iloc[[0]]
    else:
        row = model_ready.iloc[[0]]

    probability = float(output.loc[max_risk_index, "Sepsis_Probability"])

    st.subheader("Why This Risk Score?")
    st.caption(
        "SHAP explanation for the uploaded patient's highest-risk hour."
    )
    display_saved_shap_reason(row, probability, max_risk_hour)


def display_saved_shap_reason(
    row: pd.DataFrame,
    probability: float,
    hour,
) -> None:
    """Display saved SHAP-based reasons for one observation."""

    if not GLOBAL_SHAP_PATH.exists():
        st.warning(
            "Saved SHAP feature importance was not found. Run "
            "`python -m src.explainability.shap_analysis` first."
        )
        return

    if probability >= HIGH_RISK_THRESHOLD:
        risk_direction = "high-risk"
    elif probability >= ALERT_THRESHOLD:
        risk_direction = "alert-level but below the high-risk display band"
    else:
        risk_direction = "below-alert"

    st.write(
        f"At ICU hour `{hour}`, the model estimated sepsis risk "
        f"as `{probability:.3f}`, which is `{risk_direction}`. "
        f"The alert threshold is `{ALERT_THRESHOLD:.2f}` and the high-risk "
        f"display threshold is `{HIGH_RISK_THRESHOLD:.2f}`."
    )

    display_local_shap_reason(row)

    top_global = pd.read_csv(GLOBAL_SHAP_PATH)
    clinical = top_global[
        top_global["feature"].map(
            lambda feature: any(
                feature == name or feature.startswith(f"{name}_")
                for name in CLINICAL_FEATURES
            )
        )
    ].head(8)
    if not clinical.empty:
        top_global = clinical
    else:
        top_global = top_global.head(12)
    observed = build_observed_reason_table(row, top_global)

    st.markdown("**Main SHAP-ranked drivers used by the model**")
    st.dataframe(
        observed[
            [
                "feature",
                "clinical_term",
                "patient_value",
                "mean_absolute_shap",
                "interpretation",
            ]
        ],
        hide_index=True,
    )

    st.write(
        "The table ranks features by global SHAP importance, then shows "
        "this patient's value at the selected hour. These are the model "
        "inputs most likely to explain the risk score."
    )

    if INDIVIDUAL_SHAP_PATH.exists():
        example = pd.read_csv(INDIVIDUAL_SHAP_PATH).head(10)
        st.markdown("**Saved example of SHAP direction**")
        st.dataframe(
            example[
                [
                    "feature",
                    "feature_value",
                    "shap_value",
                ]
            ],
            hide_index=True,
        )
        st.caption(
            "This example shows SHAP direction: positive values push risk "
            "up, negative values push risk down. For exact per-patient "
            "SHAP directions, generate local SHAP rows for every patient."
        )


def display_local_shap_reason(row: pd.DataFrame) -> None:
    """Show exact local SHAP contributions for the selected observation."""

    try:
        import shap

        model = joblib.load(MODEL_PATH)
        feature_columns = list(model.get_booster().feature_names or [])
        if not feature_columns:
            feature_columns = [
                column
                for column in row.columns
                if column not in IDENTIFIER_COLUMNS
                and column not in {
                    TARGET_COLUMN,
                    "Sepsis_Probability",
                    "Risk_Level",
                    "Alert",
                }
            ]
        features = row.reindex(columns=feature_columns, fill_value=0)
        shap_values = shap.TreeExplainer(model)(features).values
        values = shap_values[0] if getattr(shap_values, "ndim", 1) == 2 else shap_values
        local = pd.DataFrame(
            {
                "feature": feature_columns,
                "shap_value": values,
                "patient_value": features.iloc[0].to_numpy(),
            }
        )
        local["absolute_contribution"] = local["shap_value"].abs()
        local["clinical_term"] = local["feature"].map(clinical_term_for_feature)
        local["direction"] = local["shap_value"].map(
            lambda value: "Increases predicted risk" if value > 0 else "Decreases predicted risk"
        )
        positive = local[local["shap_value"] > 0].sort_values(
            "shap_value", ascending=False
        ).head(5)
        negative = local[local["shap_value"] < 0].sort_values(
            "shap_value", ascending=True
        ).head(5)
        st.markdown("**What pushed the risk higher**")
        st.dataframe(
            positive[["feature", "clinical_term", "patient_value", "shap_value"]],
            hide_index=True,
            use_container_width=True,
        )
        st.markdown("**What pushed the risk lower**")
        st.dataframe(
            negative[["feature", "clinical_term", "patient_value", "shap_value"]],
            hide_index=True,
            use_container_width=True,
        )
        shap_export = pd.concat([positive, negative], ignore_index=True)
        st.download_button(
            "Download SHAP explanation",
            data=shap_export[
                ["feature", "clinical_term", "patient_value", "shap_value", "direction"]
            ].to_csv(index=False).encode("utf-8"),
            file_name="patient_shap_explanation.csv",
            mime="text/csv",
        )
        st.caption(
            "Positive SHAP values move the model toward higher predicted risk; "
            "negative values move it toward lower predicted risk. These are model "
            "contributions, not confirmed medical causes."
        )
    except Exception as error:
        st.caption(f"Exact local SHAP was unavailable: {error}")


def build_observed_reason_table(
    row: pd.DataFrame,
    top_global: pd.DataFrame,
) -> pd.DataFrame:
    """Join SHAP-ranked features with this patient's observed values."""

    records = []
    source = row.iloc[0]

    for item in top_global.to_dict(orient="records"):
        feature = item["feature"]
        value = source[feature] if feature in source.index else None
        records.append(
            {
                "feature": feature,
                "clinical_term": clinical_term_for_feature(feature),
                "patient_value": value,
                "mean_absolute_shap": item["mean_absolute_shap"],
                "interpretation": explain_feature_value(
                    feature,
                    value,
                ),
            }
        )

    return pd.DataFrame(records)


def clinical_term_for_feature(feature: str) -> str:
    """Map model feature names to readable medical terms."""

    base = feature.split("_")[0]
    for name, label in CLINICAL_FEATURES.items():
        if base == name:
            suffix = feature[len(name):].lstrip("_")
            if suffix:
                return f"{label} ({suffix.replace('_', ' ')})"
            return label
    return "Model timing or measurement feature"


def explain_feature_value(feature: str, value) -> str:
    """Create a short human-readable feature note."""

    if pd.isna(value):
        return "Value is missing for this hour."

    if feature == "ICULOS":
        return f"Patient is at ICU hour {value}."

    if feature.endswith("_was_missing"):
        base = feature.replace("_was_missing", "")
        if float(value) == 1:
            return f"{base} was missing; missingness itself is informative."
        return f"{base} was measured."

    if "rolling_max" in feature:
        return "Recent high value over the rolling window."

    if "rolling_min" in feature:
        return "Recent low value over the rolling window."

    if "rolling_mean" in feature:
        return "Recent average over the rolling window."

    if feature == "Age":
        return f"Patient age is {value}."

    if feature == "HospAdmTime":
        return f"Hospital admission timing value is {value}."

    return f"Observed value is {value}."


def prepare_observations(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded CSV into user_prediction observations."""

    result = dataframe.copy()

    if "Patient_ID" not in result.columns:
        result["Patient_ID"] = "UPLOADED_PATIENT"

    if "ICULOS" not in result.columns:
        result["ICULOS"] = range(1, len(result) + 1)

    if "Dataset" not in result.columns:
        result["Dataset"] = "USER_UPLOAD"

    for column in REQUIRED_COLUMNS:
        if column not in result.columns:
            result[column] = None

    result = result.sort_values(
        by=["Patient_ID", "ICULOS"],
        kind="mergesort",
    ).reset_index(drop=True)

    return result[
        ["Patient_ID", "Dataset", "ICULOS", *REQUIRED_COLUMNS]
    ]


if __name__ == "__main__":
    main()
