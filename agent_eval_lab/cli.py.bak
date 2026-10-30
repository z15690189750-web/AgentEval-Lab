from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_eval_lab.experiment_io import load_experiment
from agent_eval_lab.regression import MetricConfig, RegressionComparator
from agent_eval_lab.report import RegressionReporter


DEFAULT_METRICS = (
    MetricConfig(
        name="task_success_rate",
        threshold=0.0,
        higher_is_better=True,
    ),
    MetricConfig(
        name="average_score",
        threshold=0.0,
        higher_is_better=True,
    ),
    MetricConfig(
        name="failure_rate",
        threshold=0.0,
        higher_is_better=False,
    ),
    MetricConfig(
        name="average_latency_ms",
        threshold=0.0,
        higher_is_better=False,
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the AgentEval Lab command-line parser."""

    parser = argparse.ArgumentParser(
        prog="agent-eval",
        description=(
            "AgentEval Lab: evaluate AI agents "
            "and detect regressions."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare a baseline experiment with a current experiment.",
    )

    compare_parser.add_argument(
        "baseline",
        type=Path,
        help="Path to the baseline experiment JSON.",
    )

    compare_parser.add_argument(
        "current",
        type=Path,
        help="Path to the current experiment JSON.",
    )

    compare_parser.add_argument(
        "--json-report",
        type=Path,
        help="Optional path for the JSON regression report.",
    )

    compare_parser.add_argument(
        "--report",
        type=Path,
        help="Optional path for the Markdown regression report.",
    )

    compare_parser.add_argument(
        "--latency-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum latency increase in milliseconds "
            "required to classify a regression."
        ),
    )

    compare_parser.add_argument(
        "--success-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum task success rate decrease "
            "required to classify a regression."
        ),
    )

    compare_parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum average score decrease "
            "required to classify a regression."
        ),
    )

    compare_parser.add_argument(
        "--failure-threshold",
        type=float,
        default=0.0,
        help=(
            "Minimum failure rate increase "
            "required to classify a regression."
        ),
    )

    return parser


def build_metric_configs(
    args: argparse.Namespace,
) -> tuple[MetricConfig, ...]:
    """Build regression metric configuration from CLI arguments."""

    return (
        MetricConfig(
            name="task_success_rate",
            threshold=args.success_threshold,
            higher_is_better=True,
        ),
        MetricConfig(
            name="average_score",
            threshold=args.score_threshold,
            higher_is_better=True,
        ),
        MetricConfig(
            name="failure_rate",
            threshold=args.failure_threshold,
            higher_is_better=False,
        ),
        MetricConfig(
            name="average_latency_ms",
            threshold=args.latency_threshold,
            higher_is_better=False,
        ),
    )


def compare_experiments(
    args: argparse.Namespace,
) -> int:
    """Compare two persisted experiments."""

    baseline = load_experiment(args.baseline)
    current = load_experiment(args.current)

    if baseline.benchmark_name != current.benchmark_name:
        raise ValueError(
            "Baseline and current experiments must use "
            "the same benchmark."
        )

    comparator = RegressionComparator()

    summary = comparator.compare_all(
        baseline=baseline,
        current=current,
        metrics=build_metric_configs(args),
    )

    reporter = RegressionReporter()

    if args.json_report is not None:
        reporter.write_json(
            path=args.json_report,
            baseline=baseline,
            current=current,
            summary=summary,
        )

    if args.report is not None:
        reporter.write_markdown(
            path=args.report,
            baseline=baseline,
            current=current,
            summary=summary,
        )

    print(
        reporter.to_markdown(
            baseline=baseline,
            current=current,
            summary=summary,
        )
    )

    if summary.regressed:
        return 1

    return 0


def main() -> int:
    """Run the AgentEval Lab CLI."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "compare":
            return compare_experiments(args)

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as exc:
        print(
            f"Error: {exc}",
            file=sys.stderr,
        )
        return 2

    parser.error(
        f"Unknown command: {args.command}"
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())