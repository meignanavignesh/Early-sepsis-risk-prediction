# Early Sepsis Risk Prediction

This project predicts early sepsis risk from hourly ICU observations using
processed EHR data, temporal feature engineering, XGBoost, patient-level
evaluation, and explainability reports.

The current production model is saved at `models/xgboost.pkl`. The pipeline can
predict from the processed test dataset, from a single patient already in the
dataset, or from manually supplied/new-patient observations.

## What Is Implemented

- Patient-safe train, validation, and test split workflow.
- Leakage-safe temporal feature engineering by `Patient_ID` and `ICULOS`.
- XGBoost model training and validation reports.
- Batch prediction on processed datasets.
- Manual/new-patient prediction through `src.models.user_prediction`.
- Hourly alert rules and early-warning evaluation.
- Threshold/rule comparison for improving precision while preserving recall.
- Calibration and subgroup validation reports.
- Patient explanation summaries and error analysis.
- Lightweight local experiment tracking in `reports/experiments/`.
- Streamlit prediction interface in `app.py`.

## Key Current Results

The existing final test rule uses two consecutive ICU observations with
predicted sepsis probability at least `0.50`.

- Test patients: `6051`
- Sepsis patients: `440`
- Patient-level recall: `0.811`
- Recall at least 3 hours before onset: `0.605`
- Recall at least 6 hours before onset: `0.570`
- Recall at least 12 hours before onset: `0.514`
- Median warning time among detected sepsis patients: `31` hours
- False alerts per non-sepsis patient: about `10.19`

Validation rule optimization found:

- Best F1 rule: `3-hour rolling mean >= 0.80`
- Best recall-preserving rule with recall at least `0.80`:
  `2 consecutive hours >= 0.50`

This means the project now exposes a real clinical tradeoff: stricter rules can
reduce false alarms, while the current rule prioritizes sensitivity.

## Setup

```bash
pip install -r requirements.txt
```

On this workspace, use the local virtual environment:

```powershell
.\.venv\Scripts\python.exe -m src.models.prediction_pipeline
```

## Common Commands

Run dataset prediction:

```powershell
.\.venv\Scripts\python.exe -m src.models.prediction_pipeline
```

Run manual/new-patient input prediction:

```powershell
.\.venv\Scripts\python.exe -m src.models.user_prediction
```

Run single-patient prediction from the existing test dataset:

```powershell
.\.venv\Scripts\python.exe -m src.models.predict_single_patient
```

Compare hourly alert rules on validation data:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.hourly_threshold_optimization
```

Run hourly early-warning evaluation:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.hourly_early_warning_evaluation
```

Run validation and subgroup audit:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.validation_audit
```

Run calibration evaluation:

```powershell
.\.venv\Scripts\python.exe -m src.evaluation.calibration_evaluation
```

Generate patient explanation summaries:

```powershell
.\.venv\Scripts\python.exe -m src.explainability.patient_explanation
```

Run the Streamlit app:

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## New Reports

- `reports/hourly_threshold_optimization_report.json`
- `reports/hourly_threshold_results.csv`
- `reports/hourly_early_warning_report.json`
- `reports/hourly_early_warning_patient_results.csv`
- `reports/validation_audit_report.json`
- `reports/subgroup_evaluation_report.json`
- `reports/calibration_report.json`
- `reports/calibration_bins.csv`
- `reports/patient_explanations.csv`
- `reports/explanation_error_analysis.json`
- `reports/experiments/experiment_index.csv`

## Streamlit App

The app supports uploading a patient CSV, running the existing XGBoost model,
showing hourly sepsis probabilities, assigning low/moderate/high risk tiers,
highlighting alert hours, plotting risk over `ICULOS`, and exporting a CSV
prediction report.

The app does not retrain the model.

## Important Limitations

- The current saved model still expects the existing `196`-feature schema.
- Richer temporal features have been added for future retraining, but the
  current model must be retrained before it can use those new features.
- Probability calibration shows meaningful miscalibration, so raw probabilities
  should be interpreted as risk scores unless a calibrated model is added.
- False alert burden is still high under the recall-preserving operating point.
- Per-patient explanations currently use global SHAP feature importance plus
  patient risk summaries. Exact local SHAP explanations can be added later.
