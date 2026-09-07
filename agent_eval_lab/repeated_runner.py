from __future__ import annotations

from agent_eval_lab.core.models import Run, Task
from agent_eval_lab.run import Runner


class RepeatedRunner:
    """
    Execute the same task multiple times.

    This class is responsible only for repeated execution.
    It does not evaluate results or calculate metrics.
    """

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    def run(
        self,
        task: Task,
        repetitions: int,
    ) -> list[Run]:
        """
        Execute one task repeatedly and return all runs.

        Each run receives a unique, human-readable ID.
        """

        if repetitions < 1:
            raise ValueError("repetitions must be at least 1.")

        return [
            self.runner.run(
                task,
                run_id=f"run-{task.id}-{index:03d}",
            )
            for index in range(1, repetitions + 1)
        ]