"""Lightweight local experiment tracking."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path("reports/experiments")
INDEX_PATH = EXPERIMENT_DIR / "experiment_index.csv"


class ExperimentTracker:
    """Persist simple JSON experiment records and an index table."""

    def __init__(
        self,
        experiment_dir: str | Path = EXPERIMENT_DIR,
        index_path: str | Path = INDEX_PATH,
    ) -> None:
        self.experiment_dir = Path(experiment_dir)
        self.index_path = Path(index_path)

    def log_run(
        self,
        model_name: str,
        dataset_paths: dict[str, str],
        feature_count: int | None,
        parameters: dict[str, Any],
        metrics: dict[str, Any],
        artifacts: dict[str, str],
    ) -> dict[str, Any]:
        """Create one run JSON and append it to the CSV index."""

        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_path = self.experiment_dir / f"{run_id}.json"

        record = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": model_name,
            "dataset_paths": dataset_paths,
            "feature_count": feature_count,
            "parameters": parameters,
            "metrics": metrics,
            "artifacts": artifacts,
        }

        with run_path.open("w", encoding="utf-8") as file:
            json.dump(
                self._json_ready(record),
                file,
                indent=4,
            )

        self._append_index(
            run_id=run_id,
            model_name=model_name,
            feature_count=feature_count,
            primary_metric=self._primary_metric(metrics),
            run_path=str(run_path),
        )

        return record

    def _append_index(
        self,
        run_id: str,
        model_name: str,
        feature_count: int | None,
        primary_metric: Any,
        run_path: str,
    ) -> None:
        """Append one row to the run index."""

        exists = self.index_path.exists()

        with self.index_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "run_id",
                    "model_name",
                    "feature_count",
                    "primary_metric",
                    "run_path",
                ],
            )

            if not exists:
                writer.writeheader()

            writer.writerow(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "feature_count": feature_count,
                    "primary_metric": primary_metric,
                    "run_path": run_path,
                }
            )

    @staticmethod
    def _primary_metric(metrics: dict[str, Any]) -> Any:
        """Pick a useful metric for the index."""

        for key in (
            "best_f1",
            "f1_score",
            "patient_level_recall",
            "roc_auc",
            "pr_auc",
        ):
            if key in metrics:
                return metrics[key]

        return None

    @classmethod
    def _json_ready(cls, value: Any) -> Any:
        """Convert non-JSON scalar values into plain Python types."""

        if isinstance(value, dict):
            return {
                str(key): cls._json_ready(item)
                for key, item in value.items()
            }

        if isinstance(value, list):
            return [cls._json_ready(item) for item in value]

        if hasattr(value, "item"):
            return value.item()

        return value
