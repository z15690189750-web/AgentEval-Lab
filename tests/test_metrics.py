from datetime import datetime, timezone

from agent_eval_lab.core.models import EvaluationResult, Run
from agent_eval_lab.metrics import (
    average_latency_ms,
    average_score,
    failure_rate,
    task_success_rate,
)


def make_evaluation(
    run_id: str,
    passed: bool,
    score: float,
    evaluator: str = "test",
) -> EvaluationResult:
    return EvaluationResult(
        task_id="task-001",
        run_id=run_id,
        passed=passed,
        score=score,
        reason="test result",
        evaluator=evaluator,
    )


def make_run(
    run_id: str,
    status: str = "success",
    duration_ms: float | None = 100.0,
) -> Run:
    run = Run(
        id=run_id,
        task_id="task-001",
    )

    run.status = status  # type: ignore[assignment]
    run.finished_at = datetime.now(timezone.utc)
    run.duration_ms = duration_ms

    return run


def test_task_success_rate_with_one_evaluator_per_run() -> None:
    results = [
        make_evaluation("run-001", True, 1.0),
        make_evaluation("run-002", False, 0.0),
        make_evaluation("run-003", True, 1.0),
    ]

    assert task_success_rate(results) == 2 / 3


def test_task_success_rate_requires_all_evaluators_to_pass() -> None:
    results = [
        make_evaluation(
            "run-001",
            True,
            1.0,
            evaluator="output",
        ),
        make_evaluation(
            "run-001",
            True,
            1.0,
            evaluator="schema",
        ),
        make_evaluation(
            "run-001",
            False,
            0.0,
            evaluator="trace",
        ),
        make_evaluation(
            "run-002",
            True,
            1.0,
            evaluator="output",
        ),
        make_evaluation(
            "run-002",
            True,
            1.0,
            evaluator="schema",
        ),
        make_evaluation(
            "run-002",
            True,
            1.0,
            evaluator="trace",
        ),
    ]

    assert task_success_rate(results) == 1 / 2


def test_task_success_rate_returns_zero_for_empty_results() -> None:
    assert task_success_rate([]) == 0.0


def test_average_score() -> None:
    results = [
        make_evaluation("run-001", True, 1.0),
        make_evaluation("run-002", False, 0.0),
        make_evaluation("run-003", True, 0.5),
    ]

    assert average_score(results) == 0.5


def test_average_score_returns_zero_for_empty_results() -> None:
    assert average_score([]) == 0.0


def test_failure_rate() -> None:
    runs = [
        make_run("run-001", "success"),
        make_run("run-002", "error"),
        make_run("run-003", "success"),
        make_run("run-004", "error"),
    ]

    assert failure_rate(runs) == 0.5


def test_failure_rate_returns_zero_for_empty_runs() -> None:
    assert failure_rate([]) == 0.0


def test_average_latency_ms() -> None:
    runs = [
        make_run("run-001", duration_ms=100.0),
        make_run("run-002", duration_ms=200.0),
        make_run("run-003", duration_ms=300.0),
    ]

    assert average_latency_ms(runs) == 200.0


def test_average_latency_ignores_missing_duration() -> None:
    runs = [
        make_run("run-001", duration_ms=100.0),
        make_run("run-002", duration_ms=None),
        make_run("run-003", duration_ms=300.0),
    ]

    assert average_latency_ms(runs) == 200.0


def test_average_latency_returns_zero_when_no_duration_exists() -> None:
    runs = [
        make_run("run-001", duration_ms=None),
        make_run("run-002", duration_ms=None),
    ]

    assert average_latency_ms(runs) == 0.0