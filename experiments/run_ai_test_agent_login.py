from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from agent_eval_lab.adapters.ai_test_agent import AITestAgentAdapter
from agent_eval_lab.core.models import ExperimentResult
from agent_eval_lab.evaluation_pipeline import EvaluationPipeline
from agent_eval_lab.experiment import Experiment
from agent_eval_lab.evaluate import evaluate_schema
from agent_eval_lab.repeated_runner import RepeatedRunner
from benchmarks.ai_test_agent_login import (
    create_ai_test_agent_login_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

AI_TEST_AGENT_ROOT = Path(
    r"E:\code\1\AI-Agent-Portfolio\AI-Test-Agent1"
)

LOCAL_API_MODULE = (
    AI_TEST_AGENT_ROOT
    / "tests"
    / "support"
    / "local_api.py"
)

REPORTS_DIR = PROJECT_ROOT / "reports"


def load_local_api_server():
    """Load the AI Test Agent's existing local API fixture."""

    if not LOCAL_API_MODULE.exists():
        raise FileNotFoundError(
            f"AI Test Agent local API fixture not found: "
            f"{LOCAL_API_MODULE}"
        )

    spec = importlib.util.spec_from_file_location(
        "ai_test_agent_local_api",
        LOCAL_API_MODULE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Unable to load AI Test Agent local API fixture."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.LocalAPIServer


def save_experiment(
    result: ExperimentResult,
    path: Path,
) -> None:
    """Persist a complete experiment result as JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        result.model_dump_json(
            indent=2,
        ),
        encoding="utf-8",
    )


def build_markdown_report(
    result: ExperimentResult,
) -> str:
    """Build a human-readable experiment report."""

    lines = [
        f"# Experiment: {result.name}",
        "",
        f"- ID: `{result.id}`",
        f"- Benchmark: `{result.benchmark_name}`",
        f"- Runs: {len(result.runs)}",
        f"- Evaluations: {len(result.evaluations)}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for name, value in result.metrics.items():
        lines.append(
            f"| `{name}` | {value:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Runs",
            "",
        ]
    )

    for run in result.runs:
        lines.extend(
            [
                f"### {run.id}",
                "",
                f"- Task: `{run.task_id}`",
                f"- Status: `{run.status}`",
                (
                    f"- Duration: "
                    f"{run.duration_ms:.2f} ms"
                    if run.duration_ms is not None
                    else "- Duration: unavailable"
                ),
                (
                    f"- Error: `{run.error}`"
                    if run.error
                    else "- Error: none"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Evaluations",
            "",
        ]
    )

    for evaluation in result.evaluations:
        lines.extend(
            [
                f"### {evaluation.evaluator}",
                "",
                f"- Task: `{evaluation.task_id}`",
                f"- Run: `{evaluation.run_id}`",
                f"- Passed: `{evaluation.passed}`",
                f"- Score: `{evaluation.score:.4f}`",
                f"- Reason: {evaluation.reason}",
                "",
            ]
        )

    return "\n".join(lines)


def save_markdown_report(
    result: ExperimentResult,
    path: Path,
) -> None:
    """Persist the experiment report as Markdown."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        build_markdown_report(result),
        encoding="utf-8",
    )


def run_experiment(
    experiment_id: str,
    experiment_name: str,
    repetitions: int,
) -> ExperimentResult:
    """
    Run one real AI Test Agent evaluation experiment.

    The experiment uses the existing AI Test Agent CLI and
    local API fixture.
    """

    if repetitions < 1:
        raise ValueError(
            "repetitions must be at least 1."
        )

    LocalAPIServer = load_local_api_server()

    server = LocalAPIServer()
    server.start()

    try:
        benchmark = create_ai_test_agent_login_benchmark(
            server.base_url,
        )

        adapter = AITestAgentAdapter(
            executable="ai-test-agent",
            cwd=AI_TEST_AGENT_ROOT,
            timeout=60.0,
        )

        repeated_runner = RepeatedRunner(
            runner=adapter,
        )

        evaluation_pipeline = EvaluationPipeline(
            evaluators=(
                evaluate_schema,
            ),
        )

        experiment = Experiment(
            repeated_runner=repeated_runner,
            evaluation_pipeline=evaluation_pipeline,
        )

        return experiment.run(
            benchmark=benchmark,
            repetitions=repetitions,
            experiment_id=experiment_id,
            experiment_name=experiment_name,
        )

    finally:
        server.stop()


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the real AI Test Agent evaluation experiment."
        )
    )

    parser.add_argument(
        "--id",
        required=True,
        help="Unique experiment ID.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Human-readable experiment name.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per benchmark task.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the experiment JSON report.",
    )

    parser.add_argument(
        "--markdown",
        type=Path,
        help="Optional path to the Markdown experiment report.",
    )

    return parser


def main() -> None:
    """Run and persist one evaluation experiment."""

    parser = build_parser()
    args = parser.parse_args()

    result = run_experiment(
        experiment_id=args.id,
        experiment_name=args.name,
        repetitions=args.repetitions,
    )

    save_experiment(
        result,
        args.output,
    )

    if args.markdown is not None:
        save_markdown_report(
            result,
            args.markdown,
        )

    print(
        json.dumps(
            {
                "experiment": result.name,
                "benchmark": result.benchmark_name,
                "runs": len(result.runs),
                "evaluations": len(result.evaluations),
                "metrics": result.metrics,
                "json_report": str(args.output),
                "markdown_report": (
                    str(args.markdown)
                    if args.markdown is not None
                    else None
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()