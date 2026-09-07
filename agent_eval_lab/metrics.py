from __future__ import annotations

from statistics import mean

from agent_eval_lab.core.models import EvaluationResult, Run


def task_success_rate(
    results: list[EvaluationResult],
) -> float:
    """
    Return the proportion of evaluations that passed.
    """

    if not results:
        return 0.0

    return sum(result.passed for result in results) / len(results)


def average_score(
    results: list[EvaluationResult],
) -> float:
    """
    Return the average evaluation score.
    """

    if not results:
        return 0.0

    return mean(result.score for result in results)


def failure_rate(
    runs: list[Run],
) -> float:
    """
    Return the proportion of runs that ended in error.
    """

    if not runs:
        return 0.0

    return sum(run.status == "error" for run in runs) / len(runs)


def average_latency_ms(
    runs: list[Run],
) -> float:
    """
    Return the average duration of completed runs in milliseconds.
    """

    durations = [
        run.duration_ms
        for run in runs
        if run.duration_ms is not None
    ]

    if not durations:
        return 0.0

    return mean(durations)