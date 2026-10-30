from __future__ import annotations

from statistics import mean

from agent_eval_lab.core.models import EvaluationResult, Run


def task_success_rate(
    results: list[EvaluationResult],
) -> float:
    """
    Return the proportion of runs that passed all evaluations.

    One run may produce multiple EvaluationResult objects.
    A run is considered successful only when every evaluation
    result belonging to that run passes.

    This makes the metric a true run/task success rate rather
    than a raw evaluator pass rate.
    """

    if not results:
        return 0.0

    results_by_run: dict[str, list[EvaluationResult]] = {}

    for result in results:
        results_by_run.setdefault(
            result.run_id,
            [],
        ).append(result)

    successful_runs = sum(
        all(result.passed for result in run_results)
        for run_results in results_by_run.values()
    )

    return successful_runs / len(results_by_run)


def average_score(
    results: list[EvaluationResult],
) -> float:
    """
    Return the average evaluation score.

    This metric operates at the evaluator-result level.
    """

    if not results:
        return 0.0

    return mean(result.score for result in results)


def failure_rate(
    runs: list[Run],
) -> float:
    """
    Return the proportion of runs that ended in execution error.

    This measures execution failures, not evaluation failures.
    """

    if not runs:
        return 0.0

    return sum(
        run.status == "error"
        for run in runs
    ) / len(runs)


def average_latency_ms(
    runs: list[Run],
) -> float:
    """
    Return the average duration of completed runs in milliseconds.

    Runs without a recorded duration are excluded.
    """

    durations = [
        run.duration_ms
        for run in runs
        if run.duration_ms is not None
    ]

    if not durations:
        return 0.0

    return mean(durations)