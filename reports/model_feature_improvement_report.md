# Model Feature Improvement Report

## 1. Current Baseline

The protected production model is `models/xgboost_optimized.pkl` with a
probability threshold of `0.60` and a two-consecutive-observation alert rule.
The locked patient-level test results are:

| Metric | Test result |
|---|---:|
| Accuracy | 91.27% |
| Precision | 41.91% |
| Recall | 51.82% |
| F1-score | 46.34% |
| Specificity | 94.37% |
| ROC-AUC | 86.13% |
| PR-AUC | 52.99% |

## 2. Feature Audit

- Raw merged dataset: 43 columns.
- Model-ready dataset: 199 columns.
- Actual XGBoost input: 196 columns.
- Excluded identifiers: `Patient_ID` and `Dataset`.
- Excluded target: `SepsisLabel`.
- Patient ordering: `Patient_ID`, then `ICULOS`.
- Centered rolling windows: not used.
- Future observations: not used by the audited temporal implementation.
- Patient identifier: used for grouping only, not as a model feature.

## 3. Important Pipeline Finding

The source module `src/feature_engineering/temporal_features.py` defines 3-,
6-, and 12-hour windows and clinical interactions. However, the current
`model_ready` schema actually contains the existing 3- and 6-hour temporal
features, missing indicators, and raw clinical variables. It does not contain
the following source-level feature families:

- 12-hour rolling features
- time-since-measured features
- `Shock_Index`
- MAP/lactate/O2 saturation/respiratory abnormality rule features
- age and clinical interaction features

Therefore, those features must not be described as inputs to the current
production XGBoost model until a new model is retrained with them.

## 4. Leakage Checks

The patient split audit found no overlap:

```text
Train ∩ Validation = 0
Train ∩ Test = 0
Validation ∩ Test = 0
```

The test set remains untouched for feature or model selection.

## 5. Class Imbalance

The production model uses training-only `scale_pos_weight`. No SMOTE,
validation oversampling, test oversampling, or patient duplication is used.

## 6. Feature-Ablation Status

The current production model has not been replaced. A new feature set must be
retrained and evaluated with the same patient-level temporal rule before it is
eligible for production. The existing 196-feature model remains the valid
baseline while that experiment is isolated.

## 7. Recommendation

The next legitimate experiment is to add causal 12-hour rolling statistics,
time-since-measured features, persistence features, and clinically documented
interactions to the training and validation representations only. The model
should be replaced only if validation PR-AUC/F1 improves without an unacceptable
loss of recall or increase in false alerts.
