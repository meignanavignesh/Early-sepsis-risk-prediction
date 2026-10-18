# Root-Cause Model Diagnostic

Diagnostic-only audit. No model was trained and production files were not modified.

## Label distributions

| Split | Patients | Sepsis patients | Hourly positives | Hourly negative/positive | Patient negative/positive |
|---|---:|---:|---:|---:|---:|
| train | 28235 | 2052 | 19560 | 54.54 | 12.76 |
| validation | 6050 | 440 | 4162 | 54.62 | 12.75 |
| test | 6051 | 440 | 4194 | 54.87 | 12.75 |

## Consecutive alert diagnostic

| Rule | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 consecutive | 0.3279 | 0.6364 | 0.4328 | 280 | 5037 | 574 | 160 |
| 2 consecutive | 0.4191 | 0.5182 | 0.4634 | 228 | 5295 | 316 | 212 |
| 3 consecutive | 0.4707 | 0.4750 | 0.4729 | 209 | 5376 | 235 | 231 |

## Preliminary diagnosis

- Precision is limited by the rare positive class and false alerts at the fixed threshold.
- Recall is reduced by the two-consecutive requirement for patients with only one threshold crossing.
- The existing patient-level split is documented as leakage-free.
- Pre-onset summaries are diagnostic only and were not used for training.
- No production recommendation is made from this audit alone.

Two-rule misses caused specifically by the consecutive requirement: 52.
False alerts removed versus a single threshold crossing: 258.
