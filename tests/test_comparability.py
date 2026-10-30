"""Tests for evaluator parity and task-level metrics."""

from datetime import datetime, timezone

import pytest

from agent_eval_lab.core.models import (
    EvaluationResult,
    ExperimentResult,
    Run,
)
from agent_eval_lab.metrics import (
    average_score,
    evaluation_failure_rate,
    task_average_score,
)
from agent_eval_lab.regression import (
    EvaluatorCoverageMismatch,
    check_evaluator_parity,
    evaluator_names,
)


def make_run(run_id):
    return Run(
        id=run_id,
        task_id="task-1",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        duration_ms=1000.0,
        status="success",
    )


def make_eval(run_id, evaluator, passed, score):
    return EvaluationResult(
        task_id="task-1",
        run_id=run_id,
        passed=passed,
        score=score,
        reason="check",
        evaluator=evaluator,
    )


def make_experiment(experiment_id, evaluations):
    return ExperimentResult(
        id=experiment_id,
        name=experiment_id,
        benchmark_name="bench",
        runs=[make_run("run-1"), make_run("run-2")],
        evaluations=evaluations,
    )


def test_parity_passes_for_same_evaluator_set():
    baseline = make_experiment(
        "baseline",
        [
            make_eval("run-1", "schema", True, 1.0),
            make_eval("run-2", "schema", True, 1.0),
        ],
    )
    current = make_experiment(
        "current",
        [
            make_eval("run-1", "schema", True, 1.0),
            make_eval("run-2", "schema", False, 0.0),
        ],
    )

    check_evaluator_parity(baseline, current)


def test_parity_rejects_one_evaluator_versus_three():
    baseline = make_experiment(
        "baseline",
        [
            make_eval("run-1", "schema", True, 1.0),
            make_eval("run-2", "schema", True, 1.0),
        ],
    )
    current = make_experiment(
        "current",
        [
            make_eval("run-1", "schema", True, 1.0),
            make_eval("run-2", "schema", True, 1.0),
            make_eval("run-1", "execution_outcome", False, 0.0),
            make_eval("run-2", "execution_outcome", False, 0.0),
        ],
    )

    with pytest.raises(EvaluatorCoverageMismatch, match="not comparable"):
        check_evaluator_parity(baseline, current)

    assert evaluator_names(current) == ("execution_outcome", "schema")


def test_task_average_score_weights_each_run_once():
    evaluations = [
        make_eval("run-1", "schema", True, 1.0),
        make_eval("run-1", "execution_outcome", False, 0.0),
        make_eval("run-2", "schema", True, 1.0),
    ]

    assert average_score(evaluations) == pytest.approx(2.0 / 3.0)
    assert task_average_score(evaluations) == pytest.approx(0.75)


def test_evaluation_failure_rate_counts_failed_runs():
    evaluations = [
        make_eval("run-1", "schema", True, 1.0),
        make_eval("run-1", "execution_outcome", False, 0.0),
        make_eval("run-2", "schema", True, 1.0),
        make_eval("run-2", "execution_outcome", True, 1.0),
    ]

    assert evaluation_failure_rate(evaluations) == pytest.approx(0.5)
