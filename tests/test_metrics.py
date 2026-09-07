from datetime import timedelta

from agent_eval_lab.core.models import EvaluationResult, Run
from agent_eval_lab.metrics import (
    average_latency_ms,
    average_score,
    failure_rate,
    task_success_rate,
)


def make_result(
    passed: bool,
    score: float,
) -> EvaluationResult:
    return EvaluationResult(
        task_id="task-001",
        run_id="run-001",
        passed=passed,
        score=score,
        reason="test",
        evaluator="test",
    )


def make_run(
    status: str,
    duration_ms: float | None,
) -> Run:
    run = Run(
        id=f"run-{status}",
        task_id="task-001",
    )

    run.status = status  # type: ignore[assignment]

    if duration_ms is not None:
        run.finished_at = (
            run.started_at
            + timedelta(milliseconds=duration_ms)
        )
        run.duration_ms = duration_ms

    return run


def test_task_success_rate() -> None:
    results = [
        make_result(True, 1.0),
        make_result(False, 0.0),
        make_result(True, 1.0),
    ]

    assert task_success_rate(results) == 2 / 3


def test_task_success_rate_empty() -> None:
    assert task_success_rate([]) == 0.0


def test_average_score() -> None:
    results = [
        make_result(True, 1.0),
        make_result(False, 0.0),
        make_result(True, 0.5),
    ]

    assert average_score(results) == 0.5


def test_average_score_empty() -> None:
    assert average_score([]) == 0.0


def test_failure_rate() -> None:
    runs = [
        make_run("success", 10.0),
        make_run("error", 20.0),
        make_run("success", 30.0),
    ]

    assert failure_rate(runs) == 1 / 3


def test_failure_rate_empty() -> None:
    assert failure_rate([]) == 0.0


def test_average_latency_ms() -> None:
    runs = [
        make_run("success", 10.0),
        make_run("success", 20.0),
        make_run("error", 30.0),
    ]

    assert average_latency_ms(runs) == 20.0


def test_average_latency_ignores_missing_values() -> None:
    runs = [
        make_run("success", 10.0),
        make_run("success", None),
    ]

    assert average_latency_ms(runs) == 10.0


def test_average_latency_empty() -> None:
    assert average_latency_ms([]) == 0.0