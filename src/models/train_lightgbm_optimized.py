"""Train LightGBM and select its best patient-level operating threshold."""

from pathlib import Path
import json
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from src.evaluation.hourly_rules import aggregate_patient_alerts, evaluate_patient_predictions

INPUT = Path("data/processed/model_ready")
TARGET = "SepsisLabel"
IDS = {"Patient_ID", "Dataset", TARGET}


def main() -> None:
    train = pd.read_csv(INPUT / "train.csv")
    validation = pd.read_csv(INPUT / "validation.csv")
    features = [c for c in train.columns if c not in IDS]
    y = train[TARGET]
    weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        objective="binary",
        scale_pos_weight=weight * 0.6,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train[features], y)
    probabilities = model.predict_proba(validation[features])[:, 1]
    hourly = validation[["Patient_ID", "ICULOS", TARGET]].copy()
    hourly["Sepsis_Probability"] = probabilities
    best = None
    for threshold in [x / 100 for x in range(10, 96, 5)]:
        scored = hourly.copy()
        scored["Hourly_Alert"] = (scored["Sepsis_Probability"] >= threshold).astype(int)
        metrics = evaluate_patient_predictions(aggregate_patient_alerts(scored))
        metrics["threshold"] = threshold
        if best is None or metrics["f1_score"] > best["f1_score"]:
            best = metrics
    model_path = Path("models/lightgbm_optimized.pkl")
    report_path = Path("reports/lightgbm_optimized_report.json")
    joblib.dump(model, model_path)
    report = {"model_path": str(model_path), "features": len(features), "best_validation": best}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
