from __future__ import annotations

from dataclasses import dataclass

from agent_eval_lab.core.models import Task


@dataclass(frozen=True)
class Benchmark:
    """
    A fixed collection of evaluation tasks.

    A benchmark defines WHAT should be evaluated.
    It does not execute tasks or evaluate results.
    """

    name: str
    tasks: tuple[Task, ...]

    def __post_init__(self) -> None:
        """Validate benchmark-level invariants."""

        if not self.name.strip():
            raise ValueError("Benchmark name must not be empty.")

        task_ids = [task.id for task in self.tasks]

        if len(task_ids) != len(set(task_ids)):
            raise ValueError("Benchmark task IDs must be unique.")

    def __len__(self) -> int:
        """Return the number of tasks in the benchmark."""

        return len(self.tasks)

    def get_task(self, task_id: str) -> Task:
        """Return a task by ID."""

        for task in self.tasks:
            if task.id == task_id:
                return task

        raise KeyError(f"Task not found: {task_id}")