import pytest

from agent_eval_lab.core.models import ExperimentResult
from agent_eval_lab.regression import (
    RegressionComparator,
    RegressionResult,
)


def make_experiment(
    experiment_id: str,
    metrics: dict[str, float],
) -> ExperimentResult:
    return ExperimentResult(
        id=experiment_id,
        name=experiment_id,
        benchmark_name="basic-benchmark",
        metrics=metrics,
    )


def test_detects_regression_for_higher_is_better_metric() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.90,
        },
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.85,
        },
    )

    result = RegressionComparator().compare(
        baseline=baseline,
        current=current,
        metric="task_success_rate",
        threshold=0.02,
        higher_is_better=True,
    )

    assert isinstance(result, RegressionResult)
    assert result.baseline == 0.90
    assert result.current == 0.85
    assert result.delta == pytest.approx(-0.05)
    assert result.threshold == 0.02
    assert result.higher_is_better is True
    assert result.regressed is True


def test_detects_regression_for_lower_is_better_metric() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "average_latency_ms": 200.0,
        },
    )

    current = make_experiment(
        "current",
        {
            "average_latency_ms": 215.0,
        },
    )

    result = RegressionComparator().compare(
        baseline=baseline,
        current=current,
        metric="average_latency_ms",
        threshold=10.0,
        higher_is_better=False,
    )

    assert result.delta == 15.0
    assert result.higher_is_better is False
    assert result.regressed is True


def test_change_within_threshold_is_not_regression() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.90,
        },
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.89,
        },
    )

    result = RegressionComparator().compare(
        baseline=baseline,
        current=current,
        metric="task_success_rate",
        threshold=0.02,
        higher_is_better=True,
    )

    assert result.delta == pytest.approx(-0.01)
    assert result.regressed is False


def test_improvement_is_not_regression() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.85,
        },
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.90,
        },
    )

    result = RegressionComparator().compare(
        baseline=baseline,
        current=current,
        metric="task_success_rate",
        threshold=0.02,
        higher_is_better=True,
    )

    assert result.delta == pytest.approx(0.05)
    assert result.regressed is False


def test_lower_is_better_improvement_is_not_regression() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "average_latency_ms": 200.0,
        },
    )

    current = make_experiment(
        "current",
        {
            "average_latency_ms": 180.0,
        },
    )

    result = RegressionComparator().compare(
        baseline=baseline,
        current=current,
        metric="average_latency_ms",
        threshold=10.0,
        higher_is_better=False,
    )

    assert result.delta == -20.0
    assert result.regressed is False


def test_rejects_negative_threshold() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.90,
        },
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.85,
        },
    )

    with pytest.raises(
        ValueError,
        match="threshold must be non-negative",
    ):
        RegressionComparator().compare(
            baseline=baseline,
            current=current,
            metric="task_success_rate",
            threshold=-0.01,
        )


def test_rejects_missing_baseline_metric() -> None:
    baseline = make_experiment(
        "baseline",
        {},
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.85,
        },
    )

    with pytest.raises(
        KeyError,
        match="not found in baseline",
    ):
        RegressionComparator().compare(
            baseline=baseline,
            current=current,
            metric="task_success_rate",
        )


def test_rejects_missing_current_metric() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.90,
        },
    )

    current = make_experiment(
        "current",
        {},
    )

    with pytest.raises(
        KeyError,
        match="not found in current",
    ):
        RegressionComparator().compare(
            baseline=baseline,
            current=current,
            metric="task_success_rate",
        )