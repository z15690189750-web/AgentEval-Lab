from pathlib import Path

import pytest

from agent_eval_lab.core.models import (
    EvaluationResult,
    ExperimentResult,
    Run,
)
from agent_eval_lab.experiment_io import (
    load_experiment,
    save_experiment,
)


def make_experiment() -> ExperimentResult:
    run = Run(
        id="run-task-001-001",
        task_id="task-001",
    )

    run.trace.add_event(
        event_type="agent_start",
        name="agent_started",
    )

    run.trace.add_event(
        event_type="tool_call",
        name="search",
        data={
            "query": "example",
        },
    )

    run.finish(
        status="success",
        output={
            "answer": "correct",
        },
    )

    evaluation = EvaluationResult(
        task_id="task-001",
        run_id=run.id,
        passed=True,
        score=1.0,
        reason="Output matched expected result.",
        metrics={
            "exact_match": 1.0,
        },
        evaluator="output",
    )

    return ExperimentResult(
        id="experiment-001",
        name="Test Experiment",
        benchmark_name="test-benchmark",
        runs=[run],
        evaluations=[evaluation],
        metrics={
            "task_success_rate": 1.0,
            "average_score": 1.0,
            "failure_rate": 0.0,
            "average_latency_ms": run.duration_ms or 0.0,
        },
        metadata={
            "model": "test-model",
        },
    )


def test_save_experiment_creates_json_file(
    tmp_path: Path,
) -> None:
    experiment = make_experiment()

    output_path = (
        tmp_path
        / "results"
        / "experiment.json"
    )

    result = save_experiment(
        experiment,
        output_path,
    )

    assert result == output_path
    assert output_path.exists()

    content = output_path.read_text(
        encoding="utf-8",
    )

    assert '"experiment-001"' in content
    assert '"task_success_rate"' in content


def test_save_experiment_creates_parent_directories(
    tmp_path: Path,
) -> None:
    experiment = make_experiment()

    output_path = (
        tmp_path
        / "nested"
        / "results"
        / "experiment.json"
    )

    save_experiment(
        experiment,
        output_path,
    )

    assert output_path.exists()


def test_load_experiment_restores_experiment(
    tmp_path: Path,
) -> None:
    experiment = make_experiment()

    output_path = (
        tmp_path
        / "experiment.json"
    )

    save_experiment(
        experiment,
        output_path,
    )

    loaded = load_experiment(output_path)

    assert loaded.id == experiment.id
    assert loaded.name == experiment.name
    assert (
        loaded.benchmark_name
        == experiment.benchmark_name
    )

    assert len(loaded.runs) == 1
    assert len(loaded.evaluations) == 1

    assert loaded.runs[0].id == (
        "run-task-001-001"
    )

    assert loaded.runs[0].task_id == "task-001"

    assert loaded.runs[0].status == "success"

    assert loaded.runs[0].output == {
        "answer": "correct",
    }

    assert len(loaded.runs[0].trace.events) == 2

    assert loaded.evaluations[0].run_id == (
        "run-task-001-001"
    )

    assert loaded.evaluations[0].passed is True
    assert loaded.evaluations[0].score == 1.0

    assert loaded.metrics == experiment.metrics
    assert loaded.metadata == {
        "model": "test-model",
    }


def test_save_and_load_preserve_trace_data(
    tmp_path: Path,
) -> None:
    experiment = make_experiment()

    output_path = (
        tmp_path
        / "experiment.json"
    )

    save_experiment(
        experiment,
        output_path,
    )

    loaded = load_experiment(output_path)

    tool_event = loaded.runs[0].trace.events[1]

    assert tool_event.event_type == "tool_call"
    assert tool_event.name == "search"
    assert tool_event.data == {
        "query": "example",
    }


def test_load_experiment_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = (
        tmp_path
        / "does-not-exist.json"
    )

    with pytest.raises(
        FileNotFoundError,
        match="Experiment file not found",
    ):
        load_experiment(missing_path)


def test_load_experiment_validates_json(
    tmp_path: Path,
) -> None:
    invalid_path = (
        tmp_path
        / "invalid.json"
    )

    invalid_path.write_text(
        '{"id": "invalid"}',
        encoding="utf-8",
    )

    with pytest.raises(Exception):
        load_experiment(invalid_path)