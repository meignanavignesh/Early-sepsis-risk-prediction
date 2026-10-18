"""Compatibility entry point for the locked production evaluation."""

from src.evaluation.final_temporal_patient_evaluation import (
    FinalTemporalPatientEvaluator,
    evaluate_prediction_frame,
)


class FinalPatientEvaluator(FinalTemporalPatientEvaluator):
    """Keep the historical class name while using the deployed rule."""


def main() -> None:
    FinalPatientEvaluator().run()


if __name__ == "__main__":
    main()
