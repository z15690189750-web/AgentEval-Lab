from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_eval_lab.core.models import Run, Task


Agent = Callable[[Any], Any]


class Adapter:
    """Adapt a target agent to the evaluation runner."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    def run(self, task: Task) -> Any:
        """Execute the target agent with the task input."""
        return self.agent(task.input)


class Runner:
    """Run evaluation tasks through a target agent."""

    def __init__(self, adapter: Adapter) -> None:
        self.adapter = adapter

    def run(self, task: Task) -> Run:
        """Execute one task and return its Run record."""

        run = Run(
            id=f"run-{task.id}",
            task_id=task.id,
        )

        run.trace.add_event(
            event_type="agent_start",
            name="agent_started",
        )

        try:
            output = self.adapter.run(task)

            run.trace.add_event(
                event_type="agent_end",
                name="agent_finished",
                data={"output": output},
            )

            run.finish(
                status="success",
                output=output,
            )

        except Exception as exc:
            run.trace.add_event(
                event_type="error",
                name="agent_error",
                data={"error": str(exc)},
            )

            run.finish(
                status="error",
                error=str(exc),
            )

        return run