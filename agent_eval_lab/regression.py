from __future__ import annotations

from dataclasses import dataclass

from agent_eval_lab.core.models import ExperimentResult


@dataclass(frozen=True)
class MetricConfig:
    """
    Configuration for comparing one experiment metric.

    ``higher_is_better`` defines the direction of improvement.
    ``threshold`` defines the minimum change required to classify
    a change as a regression.
    """

    name: str
    threshold: float = 0.0
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Metric name must not be empty.")

        if self.threshold < 0:
            raise ValueError(
                "threshold must be non-negative."
            )


@dataclass(frozen=True)
class RegressionResult:
    """
    Comparison result for one metric.
    """

    metric: str
    baseline: float
    current: float
    delta: float
    threshold: float
    higher_is_better: bool
    regressed: bool


@dataclass(frozen=True)
class RegressionSummary:
    """
    Aggregate result of comparing multiple metrics.
    """

    results: tuple[RegressionResult, ...]

    @property
    def regressed(self) -> bool:
        """Return whether any metric regressed."""

        return any(
            result.regressed
            for result in self.results
        )

    @property
    def regressed_metrics(self) -> tuple[str, ...]:
        """Return names of all regressed metrics."""

        return tuple(
            result.metric
            for result in self.results
            if result.regressed
        )

    @property
    def improved_metrics(self) -> tuple[str, ...]:
        """
        Return names of metrics that improved according to
        their configured direction.
        """

        return tuple(
            result.metric
            for result in self.results
            if (
                result.delta > 0
                and result.higher_is_better
            )
            or (
                result.delta < 0
                and not result.higher_is_better
            )
        )


class RegressionComparator:
    """
    Compare metrics between baseline and current experiments.

    The comparator performs deterministic threshold-based
    regression detection.

    It does not execute agents, evaluate runs, or calculate
    experiment metrics.
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

        For higher-is-better metrics:

            current < baseline - threshold
                -> regression

        For lower-is-better metrics:

            current > baseline + threshold
                -> regression
        """

        config = MetricConfig(
            name=metric,
            threshold=threshold,
            higher_is_better=higher_is_better,
        )

        return self._compare_metric(
            baseline=baseline,
            current=current,
            config=config,
        )

    def compare_all(
        self,
        baseline: ExperimentResult,
        current: ExperimentResult,
        metrics: list[MetricConfig] | tuple[MetricConfig, ...],
    ) -> RegressionSummary:
        """
        Compare all configured metrics.

        Metrics are evaluated independently using their own
        direction and threshold configuration.
        """

        if not metrics:
            raise ValueError(
                "At least one metric configuration is required."
            )

        results = tuple(
            self._compare_metric(
                baseline=baseline,
                current=current,
                config=config,
            )
            for config in metrics
        )

        return RegressionSummary(
            results=results,
        )

    def _compare_metric(
        self,
        baseline: ExperimentResult,
        current: ExperimentResult,
        config: MetricConfig,
    ) -> RegressionResult:
        """Compare one configured metric."""

        if config.name not in baseline.metrics:
            raise KeyError(
                f"Metric '{config.name}' not found in baseline."
            )

        if config.name not in current.metrics:
            raise KeyError(
                f"Metric '{config.name}' not found in current."
            )

        baseline_value = baseline.metrics[config.name]
        current_value = current.metrics[config.name]

        delta = current_value - baseline_value

        if config.higher_is_better:
            regressed = (
                current_value
                < baseline_value - config.threshold
            )
        else:
            regressed = (
                current_value
                > baseline_value + config.threshold
            )

        return RegressionResult(
            metric=config.name,
            baseline=baseline_value,
            current=current_value,
            delta=delta,
            threshold=config.threshold,
            higher_is_better=config.higher_is_better,
            regressed=regressed,
        )