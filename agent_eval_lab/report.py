from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_eval_lab.core.models import ExperimentResult
from agent_eval_lab.regression import (
    RegressionResult,
    RegressionSummary,
)


class RegressionReporter:
    """
    Render a RegressionSummary into JSON or Markdown.

    The reporter is responsible only for presentation and
    serialization. It does not execute experiments, calculate
    metrics, or determine whether a regression occurred.
    """

    def to_dict(
        self,
        baseline: ExperimentResult,
        current: ExperimentResult,
        summary: RegressionSummary,
    ) -> dict[str, Any]:
        """
        Convert a regression comparison into a JSON-compatible
        dictionary.
        """

        return {
            "baseline": {
                "id": baseline.id,
                "name": baseline.name,
                "benchmark_name": baseline.benchmark_name,
            },
            "current": {
                "id": current.id,
                "name": current.name,
                "benchmark_name": current.benchmark_name,
            },
            "regressed": summary.regressed,
            "regressed_metrics": list(
                summary.regressed_metrics
            ),
            "improved_metrics": list(
                summary.improved_metrics
            ),
            "results": [
                self._result_to_dict(result)
                for result in summary.results
            ],
        }

    def to_json(
        self,
        baseline: ExperimentResult,
        current: ExperimentResult,
        summary: RegressionSummary,
        *,
        indent: int = 2,
    ) -> str:
        """Render a regression report as JSON."""

        return json.dumps(
            self.to_dict(
                baseline=baseline,
                current=current,
                summary=summary,
            ),
            indent=indent,
            ensure_ascii=False,
        )

    def to_markdown(
        self,
        baseline: ExperimentResult,
        current: ExperimentResult,
        summary: RegressionSummary,
    ) -> str:
        """Render a regression report as Markdown."""

        overall_status = (
            "REGRESSION"
            if summary.regressed
            else "PASS"
        )

        lines = [
            "# AgentEval Regression Report",
            "",
            f"**Overall:** `{overall_status}`",
            "",
            "## Experiments",
            "",
            "| | ID | Name | Benchmark |",
            "|---|---|---|---|",
            (
                f"| Baseline | `{baseline.id}` | "
                f"{baseline.name} | "
                f"{baseline.benchmark_name} |"
            ),
            (
                f"| Current | `{current.id}` | "
                f"{current.name} | "
                f"{current.benchmark_name} |"
            ),
            "",
            "## Metric Comparison",
            "",
            (
                "| Metric | Baseline | Current | Delta | "
                "Threshold | Direction | Status |"
            ),
            (
                "|---|---:|---:|---:|---:|---|---|"
            ),
        ]

        for result in summary.results:
            direction = (
                "higher is better"
                if result.higher_is_better
                else "lower is better"
            )

            status = (
                "REGRESSED"
                if result.regressed
                else "PASS"
            )

            lines.append(
                "| "
                f"{result.metric} | "
                f"{self._format_number(result.baseline)} | "
                f"{self._format_number(result.current)} | "
                f"{self._format_delta(result.delta)} | "
                f"{self._format_number(result.threshold)} | "
                f"{direction} | "
                f"{status} |"
            )

        lines.extend(
            [
                "",
                "## Summary",
                "",
                (
                    f"- Regression detected: "
                    f"`{summary.regressed}`"
                ),
                (
                    f"- Regressed metrics: "
                    f"{self._format_metric_names(summary.regressed_metrics)}"
                ),
                (
                    f"- Improved metrics: "
                    f"{self._format_metric_names(summary.improved_metrics)}"
                ),
                "",
            ]
        )

        return "\n".join(lines)

    def write_json(
        self,
        path: str | Path,
        baseline: ExperimentResult,
        current: ExperimentResult,
        summary: RegressionSummary,
    ) -> Path:
        """Write a regression report to a JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.to_json(
                baseline=baseline,
                current=current,
                summary=summary,
            ),
            encoding="utf-8",
        )

        return output_path

    def write_markdown(
        self,
        path: str | Path,
        baseline: ExperimentResult,
        current: ExperimentResult,
        summary: RegressionSummary,
    ) -> Path:
        """Write a regression report to a Markdown file."""

        output_path = Path(path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            self.to_markdown(
                baseline=baseline,
                current=current,
                summary=summary,
            ),
            encoding="utf-8",
        )

        return output_path

    @staticmethod
    def _result_to_dict(
        result: RegressionResult,
    ) -> dict[str, Any]:
        """Convert one regression result into JSON-compatible data."""

        return {
            "metric": result.metric,
            "baseline": result.baseline,
            "current": result.current,
            "delta": result.delta,
            "threshold": result.threshold,
            "higher_is_better": result.higher_is_better,
            "regressed": result.regressed,
        }

    @staticmethod
    def _format_number(value: float) -> str:
        """Format a numeric value for Markdown output."""

        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_delta(value: float) -> str:
        """Format a delta with an explicit sign."""

        if value > 0:
            return f"+{value:.6f}".rstrip("0").rstrip(".")

        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_metric_names(
        metrics: tuple[str, ...],
    ) -> str:
        """Format metric names for the Markdown summary."""

        if not metrics:
            return "None"

        return ", ".join(
            f"`{metric}`"
            for metric in metrics
        )