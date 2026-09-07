import pytest

from agent_eval_lab.core.models import (
    EvaluationResult,
    ExperimentResult,
    Run,
)


def make_run(
    run_id: str,
    task_id: str = "task-001",
) -> Run:
    return Run(
        id=run_id,
        task_id=task_id,
    )


def make_evaluation(
    run_id: str,
    task_id: str = "task-001",
) -> EvaluationResult:
    return EvaluationResult(
        task_id=task_id,
        run_id=run_id,
        passed=True,
        score=1.0,
        reason="Evaluation passed.",
        evaluator="output",
    )


def make_experiment() -> ExperimentResult:
    runs = [
        make_run("run-001"),
        make_run("run-002"),
    ]

    evaluations = [
        make_evaluation("run-001"),
        make_evaluation("run-002"),
    ]

    return ExperimentResult(
        id="experiment-001",
        name="baseline experiment",
        benchmark_name="basic-benchmark",
        runs=runs,
        evaluations=evaluations,
        metrics={
            "task_success_rate": 1.0,
            "average_latency_ms": 125.0,
        },
    )


def test_experiment_result_stores_experiment_metadata() -> None:
    experiment = make_experiment()

    assert experiment.id == "experiment-001"
    assert experiment.name == "baseline experiment"
    assert experiment.benchmark_name == "basic-benchmark"
    assert experiment.created_at is not None


def test_experiment_result_stores_runs() -> None:
    experiment = make_experiment()

    assert len(experiment.runs) == 2
    assert [run.id for run in experiment.runs] == [
        "run-001",
        "run-002",
    ]


def test_experiment_result_stores_evaluations() -> None:
    experiment = make_experiment()

    assert len(experiment.evaluations) == 2
    assert [
        evaluation.run_id
        for evaluation in experiment.evaluations
    ] == [
        "run-001",
        "run-002",
    ]


def test_experiment_result_stores_metrics() -> None:
    experiment = make_experiment()

    assert experiment.metrics["task_success_rate"] == 1.0
    assert experiment.metrics["average_latency_ms"] == 125.0


def test_experiment_result_allows_empty_collections() -> None:
    experiment = ExperimentResult(
        id="experiment-001",
        name="empty experiment",
        benchmark_name="empty-benchmark",
    )

    assert experiment.runs == []
    assert experiment.evaluations == []
    assert experiment.metrics == {}


def test_experiment_result_rejects_duplicate_run_ids() -> None:
    with pytest.raises(
        ValueError,
        match="run IDs must be unique",
    ):
        ExperimentResult(
            id="experiment-001",
            name="invalid experiment",
            benchmark_name="basic-benchmark",
            runs=[
                make_run("run-001"),
                make_run("run-001"),
            ],
        )


def test_experiment_result_rejects_evaluation_for_unknown_run() -> None:
    with pytest.raises(
        ValueError,
        match="Every evaluation must reference a run",
    ):
        ExperimentResult(
            id="experiment-001",
            name="invalid experiment",
            benchmark_name="basic-benchmark",
            runs=[
                make_run("run-001"),
            ],
            evaluations=[
                make_evaluation("run-999"),
            ],
        )