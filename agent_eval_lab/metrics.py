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


def task_average_score(
    results: list[EvaluationResult],
) -> float:
    """
    Return the average of per-run mean scores.

    Each run contributes exactly one value, so a run evaluated by three
    evaluators does not outweigh a run evaluated by one.
    """

    if not results:
        return 0.0

    scores_by_run: dict[str, list[float]] = {}

    for result in results:
        scores_by_run.setdefault(
            result.run_id,
            [],
        ).append(result.score)

    return mean(
        mean(scores)
        for scores in scores_by_run.values()
    )


def evaluation_failure_rate(
    results: list[EvaluationResult],
) -> float:
    """
    Return the proportion of evaluated runs with at least one failed check.

    Unlike ``failure_rate`` this measures evaluation verdicts, not
    execution errors, so it answers "did the agent do the job" rather than
    "did the run crash".
    """

    if not results:
        return 0.0

    failed_by_run: dict[str, bool] = {}

    for result in results:
        failed_by_run[result.run_id] = (
            failed_by_run.get(result.run_id, False)
            or not result.passed
        )

    return sum(failed_by_run.values()) / len(failed_by_run)
