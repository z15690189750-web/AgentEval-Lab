from __future__ import annotations

from agent_eval_lab.benchmark import Benchmark
from agent_eval_lab.core.models import EvaluationResult, ExperimentResult
from agent_eval_lab.evaluation_pipeline import EvaluationPipeline
from agent_eval_lab.metrics import (
    average_latency_ms,
    average_score,
    failure_rate,
    task_success_rate,
)
from agent_eval_lab.repeated_runner import RepeatedRunner


class Experiment:
    """
    Execute and evaluate a benchmark as one complete experiment.

    Experiment coordinates existing components:

        Benchmark
            ↓
        RepeatedRunner
            ↓
        EvaluationPipeline
            ↓
        Metrics
            ↓
        ExperimentResult

    It does not implement agent execution, evaluator logic,
    metric formulas, or regression analysis itself.
    """

    def __init__(
        self,
        repeated_runner: RepeatedRunner,
        evaluation_pipeline: EvaluationPipeline,
    ) -> None:
        self.repeated_runner = repeated_runner
        self.evaluation_pipeline = evaluation_pipeline

    def run(
        self,
        benchmark: Benchmark,
        repetitions: int,
        experiment_id: str,
        experiment_name: str,
    ) -> ExperimentResult:
        """
        Execute all benchmark tasks repeatedly and evaluate every run.

        Each task is executed ``repetitions`` times.
        Every resulting run is evaluated by every configured evaluator.
        """

        if repetitions < 1:
            raise ValueError("repetitions must be at least 1.")

        runs = []
        evaluations: list[EvaluationResult] = []

        for task in benchmark.tasks:
            task_runs = self.repeated_runner.run(
                task,
                repetitions=repetitions,
            )

            runs.extend(task_runs)

            for run in task_runs:
                evaluations.extend(
                    self.evaluation_pipeline.evaluate(
                        task,
                        run,
                    )
                )

        metrics = {
            "task_success_rate": task_success_rate(evaluations),
            "average_score": average_score(evaluations),
            "failure_rate": failure_rate(runs),
            "average_latency_ms": average_latency_ms(runs),
        }

        return ExperimentResult(
            id=experiment_id,
            name=experiment_name,
            benchmark_name=benchmark.name,
            runs=runs,
            evaluations=evaluations,
            metrics=metrics,
        )