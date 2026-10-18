# Isolated Paper-Style Pipeline

This experiment is separate from production and does not modify Streamlit or the production XGBoost model.

- Raw data: 40,336 patients, 1,552,210 observations
- Features: 28 predictors listed in the reference paper
- Preprocessing: within-patient forward/backward fill, train-only outlier-aware fallback, min-max scaling
- Balancing: training-only random undersampling to 4:1, then SMOTE to equal classes
- Model: separate Random Forest artifact
- Evaluation: row-level paper-style metrics plus patient-level secondary metrics

## Validation row-level metrics

- accuracy: 0.955840514788798
- precision: 0.15837937384898712
- recall: 0.30102315562735593
- f1: 0.20755592685417246
- specificity: 0.9686669338846462
- roc_auc: 0.8262124753114195
- pr_auc: 0.10954471423951416
- tp: 1118
- tn: 183667
- fp: 5941
- fn: 2596
- rows: 193322
- positive_rows: 3714

## Test row-level metrics

- accuracy: 0.9602281356873138
- precision: 0.13535528596187174
- recall: 0.23065564087418783
- f1: 0.1705985146352119
- specificity: 0.9733995190692765
- roc_auc: 0.8266275008182055
- pr_auc: 0.08925982296772818
- tp: 781
- tn: 182564
- fp: 4989
- fn: 2605
- rows: 190939
- positive_rows: 3386

## Separation and integrity

- Test patients were not used for training or selection.
- Production files were not modified.
