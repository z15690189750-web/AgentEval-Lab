from pathlib import Path

import pytest

from agent_eval_lab.core.models import ExperimentResult
from agent_eval_lab.regression import (
    MetricConfig,
    RegressionComparator,
)
from agent_eval_lab.report import RegressionReporter


def make_experiment(
    experiment_id: str,
    name: str,
    metrics: dict[str, float],
) -> ExperimentResult:
    return ExperimentResult(
        id=experiment_id,
        name=name,
        benchmark_name="test-benchmark",
        metrics=metrics,
    )


def make_comparison() -> tuple[
    ExperimentResult,
    ExperimentResult,
]:
    baseline = make_experiment(
        experiment_id="exp-baseline",
        name="Baseline Experiment",
        metrics={
            "task_success_rate": 0.90,
            "average_score": 0.90,
            "failure_rate": 0.10,
            "average_latency_ms": 200.0,
        },
    )

    current = make_experiment(
        experiment_id="exp-current",
        name="Current Experiment",
        metrics={
            "task_success_rate": 0.85,
            "average_score": 0.94,
            "failure_rate": 0.15,
            "average_latency_ms": 180.0,
        },
    )

    return baseline, current


def make_summary(
    baseline: ExperimentResult,
    current: ExperimentResult,
):
    return RegressionComparator().compare_all(
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


def test_to_dict_contains_experiment_information() -> None:
    baseline, current = make_comparison()
    summary = make_summary(baseline, current)

    report = RegressionReporter().to_dict(
        baseline=baseline,
        current=current,
        summary=summary,
    )

    assert report["baseline"]["id"] == "exp-baseline"
    assert report["current"]["id"] == "exp-current"

    assert report["regressed"] is True

    assert report["regressed_metrics"] == [
        "task_success_rate",
        "failure_rate",
    ]

    assert report["improved_metrics"] == [
        "average_score",
        "average_latency_ms",
    ]

    assert len(report["results"]) == 4


def test_to_json_returns_valid_json() -> None:
    baseline, current = make_comparison()
    summary = make_summary(baseline, current)

    json_text = RegressionReporter().to_json(
        baseline=baseline,
        current=current,
        summary=summary,
    )

    import json

    data = json.loads(json_text)

    assert data["regressed"] is True
    assert data["baseline"]["id"] == "exp-baseline"
    assert data["current"]["id"] == "exp-current"

    assert data["results"][0]["metric"] == (
        "task_success_rate"
    )


def test_to_markdown_contains_overall_status() -> None:
    baseline, current = make_comparison()
    summary = make_summary(baseline, current)

    markdown = RegressionReporter().to_markdown(
        baseline=baseline,
        current=current,
        summary=summary,
    )

    assert "# AgentEval Regression Report" in markdown
    assert "**Overall:** `REGRESSION`" in markdown


def test_to_markdown_contains_all_metrics() -> None:
    baseline, current = make_comparison()
    summary = make_summary(baseline, current)

    markdown = RegressionReporter().to_markdown(
        baseline=baseline,
        current=current,
        summary=summary,
    )

    assert "task_success_rate" in markdown
    assert "average_score" in markdown
    assert "failure_rate" in markdown
    assert "average_latency_ms" in markdown

    assert "REGRESSED" in markdown
    assert "PASS" in markdown


def test_to_markdown_contains_experiment_information() -> None:
    baseline, current = make_comparison()
    summary = make_summary(baseline, current)

    markdown = RegressionReporter().to_markdown(
        baseline=baseline,
        current=current,
        summary=summary,
    )

    assert "exp-baseline" in markdown
    assert "exp-current" in markdown

    assert "Baseline Experiment" in markdown
    assert "Current Experiment" in markdown

    assert "test-benchmark" in markdown


def test_write_json_creates_parent_directories(
    tmp_path: Path,
) -> None:
    baseline, current = make_comparison()
    summary = make_summary(baseline, current)

    output_path = (
        tmp_path
        / "reports"
        / "regression"
        / "report.json"
    )

    result = RegressionReporter().write_json(
        output_path,
        baseline=baseline,
        current=current,
        summary=summary,
    )

    assert result == output_path
    assert output_path.exists()

    content = output_path.read_text(
        encoding="utf-8",
    )

    assert '"regressed": true' in content


def test_write_markdown_creates_parent_directories(
    tmp_path: Path,
) -> None:
    baseline, current = make_comparison()
    summary = make_summary(baseline, current)

    output_path = (
        tmp_path
        / "reports"
        / "regression"
        / "report.md"
    )

    result = RegressionReporter().write_markdown(
        output_path,
        baseline=baseline,
        current=current,
        summary=summary,
    )

    assert result == output_path
    assert output_path.exists()

    content = output_path.read_text(
        encoding="utf-8",
    )

    assert "# AgentEval Regression Report" in content
    assert "REGRESSION" in content


def test_reporter_does_not_recalculate_regression() -> None:
    baseline, current = make_comparison()

    summary = make_summary(
        baseline,
        current,
    )

    reporter = RegressionReporter()

    report = reporter.to_dict(
        baseline=baseline,
        current=current,
        summary=summary,
    )

    assert report["results"][0]["baseline"] == 0.90
    assert report["results"][0]["current"] == 0.85
    assert report["results"][0]["delta"] == pytest.approx(-0.05)