import pytest

from agent_eval_lab.core.models import ExperimentResult
from agent_eval_lab.regression import (
    MetricConfig,
    RegressionComparator,
    RegressionResult,
    RegressionSummary,
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


def test_metric_config_rejects_empty_name() -> None:
    with pytest.raises(
        ValueError,
        match="Metric name must not be empty",
    ):
        MetricConfig(name="")


def test_metric_config_rejects_negative_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="threshold must be non-negative",
    ):
        MetricConfig(
            name="task_success_rate",
            threshold=-0.01,
        )


def test_compare_all_handles_different_metric_directions() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.90,
            "average_score": 0.90,
            "failure_rate": 0.10,
            "average_latency_ms": 200.0,
        },
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.85,
            "average_score": 0.94,
            "failure_rate": 0.15,
            "average_latency_ms": 180.0,
        },
    )

    summary = RegressionComparator().compare_all(
        baseline=baseline,
        current=current,
        metrics=[
            MetricConfig(
                name="task_success_rate",
                threshold=0.02,
                higher_is_better=True,
            ),
            MetricConfig(
                name="average_score",
                threshold=0.02,
                higher_is_better=True,
            ),
            MetricConfig(
                name="failure_rate",
                threshold=0.01,
                higher_is_better=False,
            ),
            MetricConfig(
                name="average_latency_ms",
                threshold=10.0,
                higher_is_better=False,
            ),
        ],
    )

    assert isinstance(summary, RegressionSummary)

    assert len(summary.results) == 4

    assert summary.regressed is True

    assert summary.regressed_metrics == (
        "task_success_rate",
        "failure_rate",
    )

    assert summary.improved_metrics == (
        "average_score",
        "average_latency_ms",
    )


def test_compare_all_rejects_empty_metric_configuration() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.90,
        },
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.90,
        },
    )

    with pytest.raises(
        ValueError,
        match="At least one metric configuration",
    ):
        RegressionComparator().compare_all(
            baseline=baseline,
            current=current,
            metrics=[],
        )


def test_compare_all_rejects_missing_metric() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.90,
        },
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.90,
        },
    )

    with pytest.raises(
        KeyError,
        match="Metric 'average_score' not found in baseline",
    ):
        RegressionComparator().compare_all(
            baseline=baseline,
            current=current,
            metrics=[
                MetricConfig(
                    name="average_score",
                ),
            ],
        )


def test_regression_summary_reports_no_regression() -> None:
    baseline = make_experiment(
        "baseline",
        {
            "task_success_rate": 0.90,
        },
    )

    current = make_experiment(
        "current",
        {
            "task_success_rate": 0.92,
        },
    )

    summary = RegressionComparator().compare_all(
        baseline=baseline,
        current=current,
        metrics=[
            MetricConfig(
                name="task_success_rate",
                threshold=0.01,
                higher_is_better=True,
            ),
        ],
    )

    assert summary.regressed is False
    assert summary.regressed_metrics == ()
    assert summary.improved_metrics == (
        "task_success_rate",
    )