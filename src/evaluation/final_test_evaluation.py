"""Final evaluation of the locked XGBoost model on the untouched test set."""

import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_PATH = Path("models/xgboost_optimized.pkl")

TEST_PATH = Path(
    "data/processed/model_ready/test.csv"
)

REPORT_PATH = Path(
    "reports/final_test_evaluation_report.json"
)

TARGET_COLUMN = "SepsisLabel"

IDENTIFIER_COLUMNS = (
    "Patient_ID",
    "Dataset",
)

# This threshold was selected using validation data only.
SELECTED_THRESHOLD = 0.60


def main() -> None:
    """Evaluate the locked model on unseen test patients."""

    print("=" * 65)
    print("FINAL TEST EVALUATION")
    print("=" * 65)

    # ---------------------------------------------------------
    # 1. Load model
    # ---------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading locked XGBoost model...")

    model = joblib.load(
        MODEL_PATH
    )

    # ---------------------------------------------------------
    # 2. Load test dataset
    # ---------------------------------------------------------

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_PATH}"
        )

    print("Loading untouched test dataset...")

    test = pd.read_csv(
        TEST_PATH
    )

    print(
        f"Test rows: {len(test)}"
    )

    # ---------------------------------------------------------
    # 3. Validate test dataset
    # ---------------------------------------------------------

    required_columns = {
        "Patient_ID",
        "Dataset",
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(test.columns)
    )

    if missing_columns:
        raise ValueError(
            "Test dataset is missing required "
            f"columns: {sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 4. Identify model features
    # ---------------------------------------------------------

    feature_columns = [
        column
        for column in test.columns
        if column not in IDENTIFIER_COLUMNS
        and column != TARGET_COLUMN
    ]

    X_test = test[
        feature_columns
    ]

    y_test = test[
        TARGET_COLUMN
    ]

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Test patients: "
        f"{test['Patient_ID'].nunique()}"
    )

    # ---------------------------------------------------------
    # 5. Generate probabilities
    # ---------------------------------------------------------

    print(
        "Generating test probabilities..."
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # ---------------------------------------------------------
    # 6. Apply locked threshold
    # ---------------------------------------------------------

    predictions = (
        probabilities >= SELECTED_THRESHOLD
    ).astype(int)

    # ---------------------------------------------------------
    # 7. Calculate metrics
    # ---------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    # ---------------------------------------------------------
    # 8. Sensitivity and specificity
    # ---------------------------------------------------------

    true_negative = int(
        matrix[0][0]
    )

    false_positive = int(
        matrix[0][1]
    )

    false_negative = int(
        matrix[1][0]
    )

    true_positive = int(
        matrix[1][1]
    )

    sensitivity = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative)
        > 0
        else 0.0
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if (true_negative + false_positive)
        > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # 9. Patient-level information
    # ---------------------------------------------------------

    total_patients = int(
        test["Patient_ID"].nunique()
    )

    sepsis_patients = int(
        test.groupby(
            "Patient_ID"
        )[TARGET_COLUMN]
        .max()
        .sum()
    )

    non_sepsis_patients = (
        total_patients
        - sepsis_patients
    )

    # ---------------------------------------------------------
    # 10. Classification report
    # ---------------------------------------------------------

    class_report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    # ---------------------------------------------------------
    # 11. Print results
    # ---------------------------------------------------------

    print()
    print("=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(
        f"Threshold:  "
        f"{SELECTED_THRESHOLD:.2f}"
    )

    print(
        f"Accuracy:   "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision:  "
        f"{precision:.4f}"
    )

    print(
        f"Recall:     "
        f"{recall:.4f}"
    )

    print(
        f"F1-score:   "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC:    "
        f"{roc_auc:.4f}"
    )

    print(
        f"PR-AUC:     "
        f"{pr_auc:.4f}"
    )

    print(
        f"Sensitivity: "
        f"{sensitivity:.4f}"
    )

    print(
        f"Specificity: "
        f"{specificity:.4f}"
    )

    print()
    print("Confusion matrix:")

    print(matrix)

    print()
    print(
        f"True negatives:  "
        f"{true_negative}"
    )

    print(
        f"False positives: "
        f"{false_positive}"
    )

    print(
        f"False negatives: "
        f"{false_negative}"
    )

    print(
        f"True positives:  "
        f"{true_positive}"
    )

    print()
    print(
        f"Total patients: "
        f"{total_patients}"
    )

    print(
        f"Sepsis patients: "
        f"{sepsis_patients}"
    )

    print(
        f"Non-sepsis patients: "
        f"{non_sepsis_patients}"
    )

    # ---------------------------------------------------------
    # 12. Create final report
    # ---------------------------------------------------------

    report = {
        "evaluation_type": (
            "Final evaluation on untouched test set"
        ),
        "model": "XGBoost",
        "model_path": str(
            MODEL_PATH
        ),
        "test_path": str(
            TEST_PATH
        ),
        "threshold": SELECTED_THRESHOLD,
        "test_set_used_for_training": False,
        "test_set_used_for_threshold_selection": False,
        "data_leakage_prevented": True,

        "test_rows": int(
            len(test)
        ),

        "test_patients": total_patients,

        "sepsis_patients": sepsis_patients,

        "non_sepsis_patients": (
            non_sepsis_patients
        ),

        "number_of_features": len(
            feature_columns
        ),

        "metrics": {
            "accuracy": float(
                accuracy
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1_score": float(
                f1
            ),
            "roc_auc": float(
                roc_auc
            ),
            "pr_auc": float(
                pr_auc
            ),
            "sensitivity": float(
                sensitivity
            ),
            "specificity": float(
                specificity
            ),
        },

        "confusion_matrix": (
            matrix.tolist()
        ),

        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,

        "classification_report": (
            class_report
        ),
    }

    # ---------------------------------------------------------
    # 13. Save report
    # ---------------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:

        json.dump(
            report,
            report_file,
            indent=4,
        )

    print()
    print("=" * 65)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 65)

    print(
        "Report saved to: "
        f"{REPORT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()
