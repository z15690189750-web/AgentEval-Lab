from __future__ import annotations

from agent_eval_lab.core.models import EvaluationResult, Run, Task


def evaluate_output(task: Task, run: Run) -> EvaluationResult:
    """
    Compare the agent output with the expected output.

    This evaluator is deterministic:
    the same task and run produce the same evaluation result.
    """

    if run.status != "success":
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="Agent execution did not succeed.",
            evaluator="output",
        )

    if task.expected_output is None:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="No expected output was defined.",
            evaluator="output",
        )

    passed = run.output == task.expected_output

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=passed,
        score=1.0 if passed else 0.0,
        reason=(
            "Output matches expected output."
            if passed
            else "Output does not match expected output."
        ),
        evaluator="output",
    )