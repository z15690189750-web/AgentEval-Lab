from __future__ import annotations

from dataclasses import dataclass

from agent_eval_lab.core.models import ExperimentResult


@dataclass(frozen=True)
class RegressionResult:
    """
    Comparison result for one metric.

    A regression is determined by comparing the baseline and
    current metric values against a configured threshold.
    """

    metric: str
    baseline: float
    current: float
    delta: float
    threshold: float
    higher_is_better: bool
    regressed: bool


class RegressionComparator:
    """
    Compare metrics between a baseline and current experiment.

    The comparator performs deterministic threshold-based
    regression detection. It does not execute agents,
    evaluate runs, or calculate experiment metrics.
    """

    def compare(
        self,
        baseline: ExperimentResult,
        current: ExperimentResult,
        metric: str,
        threshold: float = 0.0,
        higher_is_better: bool = True,
    ) -> RegressionResult:
        """
        Compare one metric between two experiments.

        ``threshold`` represents the minimum change that counts
        as a regression.

        For higher-is-better metrics:

            current < baseline - threshold
                -> regression

        For lower-is-better metrics:

            current > baseline + threshold
                -> regression
        """

        if threshold < 0:
            raise ValueError(
                "threshold must be non-negative."
            )

        if metric not in baseline.metrics:
            raise KeyError(
                f"Metric '{metric}' not found in baseline."
            )

        if metric not in current.metrics:
            raise KeyError(
                f"Metric '{metric}' not found in current."
            )

        baseline_value = baseline.metrics[metric]
        current_value = current.metrics[metric]

        delta = current_value - baseline_value

        if higher_is_better:
            regressed = (
                current_value
                < baseline_value - threshold
            )
        else:
            regressed = (
                current_value
                > baseline_value + threshold
            )

        return RegressionResult(
            metric=metric,
            baseline=baseline_value,
            current=current_value,
            delta=delta,
            threshold=threshold,
            higher_is_better=higher_is_better,
            regressed=regressed,
        )