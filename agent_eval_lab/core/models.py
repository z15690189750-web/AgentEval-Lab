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
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)

    input: Any
    expected_output: Any | None = None
    expected_schema: dict[str, Any] | None = None

    required_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)

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

    duration_ms: float | None = None

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

        self.duration_ms = (
            self.finished_at - self.started_at
        ).total_seconds() * 1000


class EvaluationResult(BaseModel):
    """
    Result produced by evaluating one agent run.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)

    passed: bool

    score: float = Field(ge=0.0, le=1.0)

    reason: str = Field(min_length=1)

    metrics: dict[str, float] = Field(default_factory=dict)

    evaluator: str = Field(min_length=1)


class ExperimentResult(BaseModel):
    """
    Complete result of one evaluation experiment.

    An experiment contains the raw execution runs, the evaluation
    results derived from those runs, and the aggregate metrics
    calculated from them.

    This model is a data boundary only. It does not execute agents,
    evaluate runs, calculate metrics, or perform regression analysis.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)

    benchmark_name: str = Field(min_length=1)

    created_at: datetime = Field(default_factory=utc_now)

    runs: list[Run] = Field(default_factory=list)

    evaluations: list[EvaluationResult] = Field(
        default_factory=list
    )

    metrics: dict[str, float] = Field(default_factory=dict)

    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        run_ids = [run.id for run in self.runs]

        if len(run_ids) != len(set(run_ids)):
            raise ValueError(
                "Experiment run IDs must be unique."
            )

        run_id_set = set(run_ids)

        for evaluation in self.evaluations:
            if evaluation.run_id not in run_id_set:
                raise ValueError(
                    "Every evaluation must reference a run "
                    "contained in the experiment."
                )