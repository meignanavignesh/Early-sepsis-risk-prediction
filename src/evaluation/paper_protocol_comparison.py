"""Separate comparison using a paper-like random hourly-row split.

This is intentionally not production evaluation. It is included only to show
how the paper's row-level protocol can differ from patient-level evaluation.
Patient_ID is never used as a feature.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .paper_style_pipeline import PAPER_FEATURES, CLEANED_RAW, SEED, MAX_BALANCED_PER_CLASS

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models" / "experiments"


def main() -> None:
    started = time.perf_counter()
    data = pd.read_csv(CLEANED_RAW, usecols=PAPER_FEATURES + ["Patient_ID", "SepsisLabel"])
    y = pd.to_numeric(data.pop("SepsisLabel"), errors="coerce").fillna(0).astype("int8")
    data.pop("Patient_ID")
    x = data.apply(pd.to_numeric, errors="coerce")
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=SEED, stratify=y)
    fill = train_x.median(numeric_only=True).fillna(0.0)
    train_x = train_x.fillna(fill)
    test_x = test_x.fillna(fill)
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x).astype("float32")
    test_x = scaler.transform(test_x).astype("float32")
    counts = np.bincount(train_y.to_numpy(), minlength=2)
    minority = int(min(counts))
    target_minority = min(minority, MAX_BALANCED_PER_CLASS)
    target_majority = min(target_minority * 4, MAX_BALANCED_PER_CLASS)
    under = RandomUnderSampler(sampling_strategy=target_minority / target_majority, random_state=SEED)
    bx, by = under.fit_resample(train_x, train_y)
    smote = SMOTE(sampling_strategy=1.0, random_state=SEED, k_neighbors=min(5, target_minority - 1))
    bx, by = smote.fit_resample(bx, by)
    model = RandomForestClassifier(n_estimators=160, min_samples_leaf=2, max_features="sqrt", n_jobs=2, random_state=SEED)
    model.fit(bx, by)
    probability = model.predict_proba(test_x)[:, 1]
    pred = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(test_y, pred, labels=[0, 1]).ravel()
    result = {
        "protocol": "paper-like random hourly-row split",
        "warning": "Not patient-independent; not a production metric. Hourly rows from the same patient may occur in both partitions.",
        "rows": int(len(data)), "train_rows": int(len(train_y)), "test_rows": int(len(test_y)),
        "train_class_counts_before_balance": counts.tolist(),
        "balanced_class_counts": np.bincount(by, minlength=2).tolist(),
        "features": PAPER_FEATURES,
        "accuracy": float(accuracy_score(test_y, pred)),
        "precision": float(precision_score(test_y, pred, zero_division=0)),
        "recall": float(recall_score(test_y, pred, zero_division=0)),
        "f1": float(f1_score(test_y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "roc_auc": float(roc_auc_score(test_y, probability)),
        "pr_auc": float(average_precision_score(test_y, probability)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "training_minutes": round((time.perf_counter() - started) / 60, 3),
        "production_modified": False,
    }
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "paper_protocol_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
