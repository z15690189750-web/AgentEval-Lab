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
        """Execute the target agent and return its raw output."""

        return self.agent(task.input)


class Runner:
    """
    Run evaluation tasks through a target agent adapter.

    Runner owns the AgentEval Lab Run lifecycle and attaches
    adapter-provided trace evidence when available.
    """

    def __init__(self, adapter: Adapter) -> None:
        self.adapter = adapter

    def run(
        self,
        task: Task,
        run_id: str | None = None,
    ) -> Run:
        """
        Execute one task and return its Run record.

        The adapter result becomes Run.output directly.

        If the adapter returns a dictionary containing
        ``trace_events``, those events are attached to the Run trace.
        This keeps target-specific evidence inside the adapter while
        keeping Run lifecycle management generic.
        """

        run = Run(
            id=run_id or f"run-{task.id}",
            task_id=task.id,
        )

        run.trace.add_event(
            event_type="agent_start",
            name="agent_started",
        )

        try:
            output = self.adapter.run(task)

            self._attach_adapter_trace(
                run=run,
                output=output,
            )

            run.trace.add_event(
                event_type="agent_end",
                name="agent_finished",
                data={
                    "output_type": type(output).__name__,
                },
            )

            run.finish(
                status="success",
                output=output,
            )

        except Exception as exc:
            run.trace.add_event(
                event_type="error",
                name="agent_error",
                data={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

            run.finish(
                status="error",
                error=str(exc),
            )

        return run

    @staticmethod
    def _attach_adapter_trace(
        *,
        run: Run,
        output: Any,
    ) -> None:
        """
        Attach trace evidence supplied by an adapter.

        Invalid adapter trace entries are ignored rather than allowing
        diagnostic metadata to break the target-agent execution record.
        """

        if not isinstance(output, dict):
            return

        trace_events = output.get("trace_events")

        if not isinstance(trace_events, list):
            return

        for event in trace_events:
            if not isinstance(event, dict):
                continue

            event_type = event.get("event_type")
            name = event.get("name")
            data = event.get("data", {})

            if event_type not in {
                "agent_start",
                "agent_end",
                "tool_call",
                "tool_result",
                "error",
                "message",
            }:
                continue

            if not isinstance(name, str) or not name.strip():
                continue

            if not isinstance(data, dict):
                data = {}

            run.trace.add_event(
                event_type=event_type,
                name=name,
                data=data,
            )