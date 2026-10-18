"""Shared hourly alert rules and patient-level metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PATIENT_COLUMN = "Patient_ID"
TIME_COLUMN = "ICULOS"
LABEL_COLUMN = "SepsisLabel"
PROBABILITY_COLUMN = "Sepsis_Probability"


@dataclass(frozen=True)
class AlertRule:
    """Configuration for one patient-level hourly alert rule."""

    name: str
    threshold: float
    strategy: str
    window: int | None = None
    required_consecutive: int | None = None


def add_hourly_risk_features(
    dataframe: pd.DataFrame,
    probability_column: str = PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """Add leakage-safe post-prediction hourly risk features."""

    result = dataframe.copy()
    sort_columns = [PATIENT_COLUMN]

    if TIME_COLUMN in result.columns:
        sort_columns.append(TIME_COLUMN)

    result = result.sort_values(
        by=sort_columns,
        kind="mergesort",
    ).reset_index(drop=True)

    grouped = result.groupby(
        PATIENT_COLUMN,
        sort=False,
    )[probability_column]

    for window in (3, 6, 12):
        rolling = grouped.rolling(
            window=window,
            min_periods=1,
        )

        result[f"Risk_Mean_{window}h"] = (
            rolling.mean().reset_index(level=0, drop=True)
        )

        result[f"Risk_Max_{window}h"] = (
            rolling.max().reset_index(level=0, drop=True)
        )

        result[f"High_Risk_Count_{window}h"] = (
            grouped.transform(
                lambda values, window=window: (
                    values.ge(0.5)
                    .rolling(window=window, min_periods=1)
                    .sum()
                )
            )
        )

        result[f"Risk_Trend_{window}h"] = (
            result[probability_column]
            - grouped.shift(window)
        )

    result["Risk_Increase_1h"] = (
        result[probability_column]
        - grouped.shift(1)
    )

    result["Risk_Level"] = pd.cut(
        result[probability_column],
        bins=[-np.inf, 0.3, 0.5, np.inf],
        labels=["LOW", "MODERATE", "HIGH"],
    ).astype(str)

    return result


def consecutive_alerts(
    probabilities: Iterable[float],
    threshold: float,
    required_consecutive: int,
) -> list[int]:
    """Return 1 once the sustained-risk alert rule has fired."""

    alerts: list[int] = []
    consecutive = 0
    fired = False

    for probability in probabilities:
        if probability >= threshold:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= required_consecutive:
            fired = True

        alerts.append(int(fired))

    return alerts


def consecutive_alert_series(
    probabilities: pd.Series,
    threshold: float,
    required_consecutive: int,
) -> pd.Series:
    """Return sustained-risk alerts with the original index preserved."""

    alerts = consecutive_alerts(
        probabilities.astype(float).tolist(),
        threshold,
        required_consecutive,
    )

    return pd.Series(alerts, index=probabilities.index)


def apply_alert_rule(
    dataframe: pd.DataFrame,
    rule: AlertRule,
    probability_column: str = PROBABILITY_COLUMN,
    features_already_added: bool = False,
) -> pd.DataFrame:
    """Apply an hourly rule and return row-level alert flags."""

    if features_already_added:
        result = dataframe.copy()
    else:
        result = add_hourly_risk_features(
            dataframe,
            probability_column=probability_column,
        )

    if rule.strategy == "single_threshold":
        result["Hourly_Alert"] = (
            result[probability_column] >= rule.threshold
        ).astype(int)

    elif rule.strategy == "consecutive":
        if rule.required_consecutive is None:
            raise ValueError(
                "Consecutive rule requires required_consecutive."
            )

        result["Hourly_Alert"] = (
            result.groupby(
                PATIENT_COLUMN,
                sort=False,
                group_keys=False,
            )
            [probability_column]
            .apply(
                consecutive_alert_series,
                threshold=rule.threshold,
                required_consecutive=int(
                    rule.required_consecutive
                ),
            )
            .reindex(result.index)
            .astype(int)
        )

    elif rule.strategy == "rolling_mean":
        if rule.window is None:
            raise ValueError("Rolling mean rule requires window.")

        result["Hourly_Alert"] = (
            result[f"Risk_Mean_{rule.window}h"] >= rule.threshold
        ).astype(int)

    elif rule.strategy == "rolling_max":
        if rule.window is None:
            raise ValueError("Rolling max rule requires window.")

        result["Hourly_Alert"] = (
            result[f"Risk_Max_{rule.window}h"] >= rule.threshold
        ).astype(int)

    elif rule.strategy == "risk_tier":
        result["Hourly_Alert"] = (
            result["Risk_Level"] == "HIGH"
        ).astype(int)

    else:
        raise ValueError(f"Unknown alert strategy: {rule.strategy}")

    return result


def aggregate_patient_alerts(
    hourly: pd.DataFrame,
    probability_column: str = PROBABILITY_COLUMN,
) -> pd.DataFrame:
    """Aggregate hourly alerts into patient-level predictions."""

    records = []

    for patient_id, group in hourly.groupby(PATIENT_COLUMN, sort=False):
        labels = (
            group[LABEL_COLUMN].astype(int)
            if LABEL_COLUMN in group.columns
            else pd.Series([0] * len(group))
        )

        alert_rows = group[group["Hourly_Alert"] == 1]
        first_alert_hour = None

        if not alert_rows.empty and TIME_COLUMN in group.columns:
            first_alert_hour = float(alert_rows[TIME_COLUMN].iloc[0])

        records.append(
            {
                PATIENT_COLUMN: patient_id,
                "Actual_Patient_Label": int(labels.max()),
                "Predicted_Patient_Label": int(
                    group["Hourly_Alert"].max()
                ),
                "Max_Probability": float(
                    group[probability_column].max()
                ),
                "Mean_Probability": float(
                    group[probability_column].mean()
                ),
                "Alert_Hours": int(group["Hourly_Alert"].sum()),
                "Total_Observations": int(len(group)),
                "First_Alert_Hour": first_alert_hour,
            }
        )

    return pd.DataFrame(records)


def evaluate_patient_predictions(
    patient_data: pd.DataFrame,
) -> dict:
    """Calculate patient-level metrics and alert-burden measures."""

    y_true = patient_data["Actual_Patient_Label"].astype(int)
    y_pred = patient_data["Predicted_Patient_Label"].astype(int)
    scores = patient_data["Max_Probability"].astype(float)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    metrics = {
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "true_negatives": true_negative,
        "predicted_positive_patients": int(y_pred.sum()),
        "actual_positive_patients": int(y_true.sum()),
        "alerts_per_patient": float(
            patient_data["Alert_Hours"].sum()
            / max(len(patient_data), 1)
        ),
        "false_alerts_per_icu_stay": float(
            patient_data.loc[
                y_true == 0,
                "Alert_Hours",
            ].sum()
            / max(int((y_true == 0).sum()), 1)
        ),
    }

    if y_true.nunique() > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(
            average_precision_score(y_true, scores)
        )
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics
