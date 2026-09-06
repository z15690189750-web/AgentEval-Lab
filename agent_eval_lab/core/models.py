from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


EventType = Literal[
    "agent_start",
    "agent_end",
    "tool_call",
    "tool_result",
    "error",
    "message",
]


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class Task(BaseModel):
    """
    A reproducible evaluation task.

    A Task describes what the target agent is expected to do.
    It does not represent one specific execution.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)

    input: Any

    expected_output: Any | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceEvent(BaseModel):
    """
    One event in an agent execution trace.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=utc_now)

    event_type: EventType

    name: str

    data: dict[str, Any] = Field(default_factory=dict)


class Trace(BaseModel):
    """
    Complete execution trace for one agent run.
    """

    model_config = ConfigDict(extra="forbid")

    events: list[TraceEvent] = Field(default_factory=list)

    def add_event(
        self,
        event_type: EventType,
        name: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        """Append one event to the trace."""

        event = TraceEvent(
            event_type=event_type,
            name=name,
            data=data or {},
        )

        self.events.append(event)


class Run(BaseModel):
    """
    One execution of a Task by a target agent.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)

    status: Literal["running", "success", "failure", "error"] = "running"

    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None

    output: Any | None = None

    trace: Trace = Field(default_factory=Trace)

    error: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def finish(
        self,
        status: Literal["success", "failure", "error"],
        output: Any | None = None,
        error: str | None = None,
    ) -> None:
        """Finish the run and record its final state."""

        self.status = status
        self.output = output
        self.error = error
        self.finished_at = utc_now()