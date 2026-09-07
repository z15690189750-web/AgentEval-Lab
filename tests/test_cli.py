from __future__ import annotations

import json
from pathlib import Path

from agent_eval_lab.cli import main
from agent_eval_lab.core.models import (
    EvaluationResult,
    ExperimentResult,
    Run,
)
from agent_eval_lab.experiment_io import save_experiment


def make_experiment(
    *,
    experiment_id: str,
    name: str,
    success_rate: float,
    score: float,
    failure_rate: float,
    latency_ms: float,
) -> ExperimentResult:
    run = Run(
        id=f"run-{experiment_id}",
        task_id="task-001",
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
        passed=success_rate == 1.0,
        score=score,
        reason="Test evaluation.",
        metrics={
            "exact_match": score,
        },
        evaluator="output",
    )

    return ExperimentResult(
        id=experiment_id,
        name=name,
        benchmark_name="test-benchmark",
        runs=[run],
        evaluations=[evaluation],
        metrics={
            "task_success_rate": success_rate,
            "average_score": score,
            "failure_rate": failure_rate,
            "average_latency_ms": latency_ms,
        },
    )


def test_cli_compare_passes_when_no_regression(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    baseline = make_experiment(
        experiment_id="baseline-001",
        name="Baseline",
        success_rate=1.0,
        score=1.0,
        failure_rate=0.0,
        latency_ms=100.0,
    )

    current = make_experiment(
        experiment_id="current-001",
        name="Current",
        success_rate=1.0,
        score=1.0,
        failure_rate=0.0,
        latency_ms=100.0,
    )

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    save_experiment(
        baseline,
        baseline_path,
    )

    save_experiment(
        current,
        current_path,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-eval",
            "compare",
            str(baseline_path),
            str(current_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "PASS" in captured.out
    assert "Regression detected: `False`" in captured.out


def test_cli_compare_returns_nonzero_on_regression(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    baseline = make_experiment(
        experiment_id="baseline-001",
        name="Baseline",
        success_rate=1.0,
        score=1.0,
        failure_rate=0.0,
        latency_ms=100.0,
    )

    current = make_experiment(
        experiment_id="current-001",
        name="Current",
        success_rate=0.8,
        score=0.8,
        failure_rate=0.2,
        latency_ms=150.0,
    )

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    save_experiment(
        baseline,
        baseline_path,
    )

    save_experiment(
        current,
        current_path,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-eval",
            "compare",
            str(baseline_path),
            str(current_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "REGRESSION" in captured.out
    assert "task_success_rate" in captured.out
    assert "average_score" in captured.out
    assert "failure_rate" in captured.out
    assert "average_latency_ms" in captured.out


def test_cli_compare_writes_reports(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    baseline = make_experiment(
        experiment_id="baseline-001",
        name="Baseline",
        success_rate=1.0,
        score=1.0,
        failure_rate=0.0,
        latency_ms=100.0,
    )

    current = make_experiment(
        experiment_id="current-001",
        name="Current",
        success_rate=0.9,
        score=0.9,
        failure_rate=0.1,
        latency_ms=120.0,
    )

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    json_report = (
        tmp_path
        / "reports"
        / "regression.json"
    )

    markdown_report = (
        tmp_path
        / "reports"
        / "regression.md"
    )

    save_experiment(
        baseline,
        baseline_path,
    )

    save_experiment(
        current,
        current_path,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-eval",
            "compare",
            str(baseline_path),
            str(current_path),
            "--json-report",
            str(json_report),
            "--report",
            str(markdown_report),
        ],
    )

    exit_code = main()

    capsys.readouterr()

    assert exit_code == 1

    assert json_report.exists()
    assert markdown_report.exists()

    report = json.loads(
        json_report.read_text(
            encoding="utf-8",
        )
    )

    assert report["regressed"] is True

    assert (
        "task_success_rate"
        in report["regressed_metrics"]
    )

    markdown = markdown_report.read_text(
        encoding="utf-8",
    )

    assert "# AgentEval Regression Report" in markdown
    assert "REGRESSION" in markdown


def test_cli_compare_rejects_different_benchmarks(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    baseline = make_experiment(
        experiment_id="baseline-001",
        name="Baseline",
        success_rate=1.0,
        score=1.0,
        failure_rate=0.0,
        latency_ms=100.0,
    )

    current = make_experiment(
        experiment_id="current-001",
        name="Current",
        success_rate=1.0,
        score=1.0,
        failure_rate=0.0,
        latency_ms=100.0,
    )

    current.benchmark_name = "different-benchmark"

    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    save_experiment(
        baseline,
        baseline_path,
    )

    save_experiment(
        current,
        current_path,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "agent-eval",
            "compare",
            str(baseline_path),
            str(current_path),
        ],
    )

    exit_code = main()

    captured = capsys.readouterr()

    assert exit_code == 2
    assert "same benchmark" in captured.err