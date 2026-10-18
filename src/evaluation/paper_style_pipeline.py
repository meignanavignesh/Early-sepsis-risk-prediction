"""Isolated reproduction of the paper-style sepsis experiment.

This module is intentionally separate from the production XGBoost pipeline.
It uses the raw PhysioNet-style PSV files, the paper's 28 predictors, row-level
training with undersampling + SMOTE, and a separate Random Forest artifact.

The reported paper-style metrics are row-level. Patient-level metrics are also
reported as a secondary view so the difference from production evaluation is
explicit.
"""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
CLEANED_RAW = ROOT / "data" / "processed" / "cleaned_dataset.csv"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models" / "experiments"
SEED = 42
MAX_BALANCED_PER_CLASS = 90_000

# The 28 predictors listed in the reference paper.
PAPER_FEATURES = [
    "ICULOS", "Unit1", "Unit2", "BUN", "HR", "Resp", "Gender",
    "Bilirubin_direct", "Bilirubin_total", "Creatinine", "EtCO2", "Hct",
    "Hgb", "WBC", "Fibrinogen", "Temp", "MAP", "DBP", "Calcium", "PTT",
    "Glucose", "SBP", "BaseExcess", "Platelets", "Magnesium", "HCO3",
    "AST", "HospAdmTime",
]


def load_raw() -> pd.DataFrame:
    # This is the project's materialized raw-variable table. It has the same
    # original clinical columns as the PSV files, but avoids loading thousands
    # of separate files at once.
    if CLEANED_RAW.exists():
        usecols = PAPER_FEATURES + ["Patient_ID", "SepsisLabel"]
        data = pd.read_csv(CLEANED_RAW, usecols=usecols)
        data["Patient_ID"] = data["Patient_ID"].astype(str)
        data["SepsisLabel"] = pd.to_numeric(data["SepsisLabel"], errors="coerce").fillna(0).astype("int8")
        return data
    files = sorted(RAW_DIR.rglob("*.psv"))
    if not files:
        raise FileNotFoundError(f"No .psv files found below {RAW_DIR}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, sep="|", dtype={"SepsisLabel": "int8"})
        frame.insert(0, "Patient_ID", path.stem)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["ICULOS"] = pd.to_numeric(data["ICULOS"], errors="coerce")
    data["SepsisLabel"] = data["SepsisLabel"].astype("int8")
    return data


def split_by_patient(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = np.array(sorted(data["Patient_ID"].unique()))
    rng = np.random.default_rng(SEED)
    rng.shuffle(patients)
    n_train = int(len(patients) * 0.75)
    n_valid = int(len(patients) * 0.125)
    train_ids = set(patients[:n_train])
    valid_ids = set(patients[n_train:n_train + n_valid])
    train = data[data.Patient_ID.isin(train_ids)].copy()
    valid = data[data.Patient_ID.isin(valid_ids)].copy()
    test = data[~data.Patient_ID.isin(train_ids | valid_ids)].copy()
    return train, valid, test


def paper_impute_scale(
    train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Forward/backward fill within patient, then train-only outlier-aware fill."""
    ordered = []
    for frame in (train, valid, test):
        work = frame.sort_values(["Patient_ID", "ICULOS"]).copy()
        work[PAPER_FEATURES] = work.groupby("Patient_ID", sort=False)[PAPER_FEATURES].ffill()
        work[PAPER_FEATURES] = work.groupby("Patient_ID", sort=False)[PAPER_FEATURES].bfill()
        ordered.append(work)
    train, valid, test = ordered

    fill_values: dict[str, float] = {}
    for feature in PAPER_FEATURES:
        values = pd.to_numeric(train[feature], errors="coerce")
        observed = values.dropna()
        mean = float(observed.mean()) if not observed.empty else 0.0
        median = float(observed.median()) if not observed.empty else 0.0
        q1, q3 = observed.quantile([0.25, 0.75]) if not observed.empty else (0.0, 0.0)
        iqr = float(q3 - q1)
        if iqr == 0 or observed.empty:
            outlier_rate = 0.0
        else:
            outlier_rate = float(((observed < q1 - 1.5 * iqr) | (observed > q3 + 1.5 * iqr)).mean())
        fill_values[feature] = mean if outlier_rate < 0.05 else median

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    # Fill with the paper-style train-only values, then guarantee finite values.
    train_x = train[PAPER_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    valid_x = valid[PAPER_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    test_x = test[PAPER_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(fill_values)
    train_x = imputer.fit_transform(train_x)
    valid_x = imputer.transform(valid_x)
    test_x = imputer.transform(test_x)
    scaler = MinMaxScaler()
    train_x = scaler.fit_transform(train_x).astype("float32")
    valid_x = scaler.transform(valid_x).astype("float32")
    test_x = scaler.transform(test_x).astype("float32")
    return train_x, valid_x, test_x, fill_values


def balance_training(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    counts = np.bincount(y.astype(int), minlength=2)
    minority = int(min(counts))
    if minority == 0:
        raise ValueError(f"Training split has only one class: {counts.tolist()}")
    target_minority = min(minority, MAX_BALANCED_PER_CLASS)
    target_majority = min(target_minority * 4, MAX_BALANCED_PER_CLASS)
    under_ratio = target_minority / target_majority
    under = RandomUnderSampler(
        sampling_strategy=under_ratio, random_state=SEED,
    )
    x_under, y_under = under.fit_resample(x, y)
    smote = SMOTE(
        sampling_strategy=1.0, random_state=SEED, k_neighbors=min(5, target_minority - 1),
    )
    x_balanced, y_balanced = smote.fit_resample(x_under, y_under)
    return x_balanced.astype("float32"), y_balanced.astype("int8"), {
        "before": counts.tolist(),
        "after_undersampling": np.bincount(y_under, minlength=2).tolist(),
        "after_smote": np.bincount(y_balanced, minlength=2).tolist(),
    }


def metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float = 0.5) -> dict:
    y = frame["SepsisLabel"].to_numpy(dtype=int)
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, probability)),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "rows": int(len(frame)), "positive_rows": int(y.sum()),
    }


def patient_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float = 0.5) -> dict:
    work = frame[["Patient_ID", "SepsisLabel"]].copy()
    work["probability"] = probability
    truth = work.groupby("Patient_ID").SepsisLabel.max().astype(int)
    pred = (work.groupby("Patient_ID").probability.max() >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "roc_auc": float(roc_auc_score(truth, work.groupby("Patient_ID").probability.max())) if len(np.unique(truth)) > 1 else None,
        "pr_auc": float(average_precision_score(truth, work.groupby("Patient_ID").probability.max())),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "patients": int(len(truth)), "positive_patients": int(truth.sum()),
    }


def main() -> None:
    start = time.perf_counter()
    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading cleaned raw-variable table...", flush=True)
    data = load_raw()
    print(f"Loaded {len(data):,} rows from {data.Patient_ID.nunique():,} patients.", flush=True)
    train, valid, test = split_by_patient(data)
    print("Applying paper-style imputation and scaling...", flush=True)
    train_x, valid_x, test_x, fill_values = paper_impute_scale(train, valid, test)
    train_y = train.SepsisLabel.to_numpy(dtype="int8")
    print("Balancing training rows with undersampling + SMOTE...", flush=True)
    balanced_x, balanced_y, balance_report = balance_training(train_x, train_y)
    print(f"Balanced rows: {len(balanced_y):,}", flush=True)

    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=2,
        random_state=SEED,
        class_weight=None,
    )
    print("Training separate Random Forest...", flush=True)
    model.fit(balanced_x, balanced_y)
    validation_probability = model.predict_proba(valid_x)[:, 1]
    test_probability = model.predict_proba(test_x)[:, 1]
    validation_metrics = metrics(valid, validation_probability)
    test_metrics = metrics(test, test_probability)
    validation_patients = patient_metrics(valid, validation_probability)
    test_patients = patient_metrics(test, test_probability)

    artifact = {
        "model": model,
        "features": PAPER_FEATURES,
        "fill_values": fill_values,
        "seed": SEED,
    }
    dump(artifact, MODEL_DIR / "paper_style_random_forest.pkl")
    report = {
        "pipeline": "isolated paper-style experiment",
        "paper_features": PAPER_FEATURES,
        "feature_count": len(PAPER_FEATURES),
        "raw_rows": int(len(data)),
        "raw_patients": int(data.Patient_ID.nunique()),
        "split_patients": {"train": int(train.Patient_ID.nunique()), "validation": int(valid.Patient_ID.nunique()), "test": int(test.Patient_ID.nunique())},
        "balancing": {"method": "random undersampling to 4:1 followed by SMOTE", **balance_report},
        "validation_row_level": validation_metrics,
        "test_row_level": test_metrics,
        "validation_patient_level": validation_patients,
        "test_patient_level": test_patients,
        "training_minutes": round((time.perf_counter() - start) / 60, 3),
        "production_modified": False,
        "test_used_for_training_or_selection": False,
        "note": "Row-level metrics are the closest comparison to the reference paper. Patient-level metrics are included because production uses patient-level evaluation.",
    }
    (REPORT_DIR / "paper_style_pipeline.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# Isolated Paper-Style Pipeline",
        "",
        "This experiment is separate from production and does not modify Streamlit or the production XGBoost model.",
        "",
        f"- Raw data: {report['raw_patients']:,} patients, {report['raw_rows']:,} observations",
        "- Features: 28 predictors listed in the reference paper",
        "- Preprocessing: within-patient forward/backward fill, train-only outlier-aware fallback, min-max scaling",
        "- Balancing: training-only random undersampling to 4:1, then SMOTE to equal classes",
        "- Model: separate Random Forest artifact",
        "- Evaluation: row-level paper-style metrics plus patient-level secondary metrics",
        "",
        "## Validation row-level metrics",
        "",
    ]
    for key, value in validation_metrics.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Test row-level metrics", ""]
    for key, value in test_metrics.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Separation and integrity", "", "- Test patients were not used for training or selection.", "- Production files were not modified."]
    (REPORT_DIR / "paper_style_pipeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    del data, train, valid, test, train_x, valid_x, test_x, balanced_x, balanced_y, model
    gc.collect()


if __name__ == "__main__":
    main()
