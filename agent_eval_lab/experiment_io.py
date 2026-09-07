from __future__ import annotations

from pathlib import Path

from agent_eval_lab.core.models import ExperimentResult


def save_experiment(
    experiment: ExperimentResult,
    path: str | Path,
) -> Path:
    """
    Save an ExperimentResult as JSON.

    The complete experiment is persisted, including:
    - experiment metadata
    - raw runs
    - traces
    - evaluation results
    - aggregate metrics
    """

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        experiment.model_dump_json(
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


def load_experiment(
    path: str | Path,
) -> ExperimentResult:
    """
    Load an ExperimentResult from a JSON file.

    Pydantic performs schema validation when the JSON is
    reconstructed into an ExperimentResult.
    """

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Experiment file not found: {input_path}"
        )

    return ExperimentResult.model_validate_json(
        input_path.read_text(
            encoding="utf-8",
        )
    )