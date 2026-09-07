from __future__ import annotations

from collections.abc import Callable, Sequence

from agent_eval_lab.core.models import EvaluationResult, Run, Task


Evaluator = Callable[[Task, Run], EvaluationResult]


class EvaluationPipeline:
    """
    Run multiple evaluators against one task execution.

    The pipeline coordinates evaluators but does not define
    evaluation logic itself.
    """

    def __init__(self, evaluators: Sequence[Evaluator]) -> None:
        if not evaluators:
            raise ValueError("At least one evaluator is required.")

        self.evaluators = tuple(evaluators)

    def evaluate(
        self,
        task: Task,
        run: Run,
    ) -> list[EvaluationResult]:
        """Evaluate one run with every configured evaluator."""

        return [
            evaluator(task, run)
            for evaluator in self.evaluators
        ]