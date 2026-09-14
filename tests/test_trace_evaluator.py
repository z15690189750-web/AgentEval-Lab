from __future__ import annotations

from agent_eval_lab.core.models import Run, Task
from agent_eval_lab.evaluate import evaluate_trace


def make_task() -> Task:
    return Task(
        id="trace-task",
        name="Trace Task",
        description="Trace evaluation test.",
        input={},
    )


def make_run() -> Run:
    return Run(
        id="run-001",
        task_id="trace-task",
    )


def add_lifecycle_events(run: Run) -> None:
    run.trace.add_event(
        event_type="agent_start",
        name="agent_started",
    )

    run.trace.add_event(
        event_type="agent_end",
        name="agent_finished",
    )


def add_execution_event(
    run: Run,
    trace_id: str = "trace-001",
) -> None:
    run.trace.add_event(
        event_type="message",
        name="ai_test_agent_execution",
        data={
            "source": "ai_test_agent",
            "evidence_type": "test_execution",
            "trace_id": trace_id,
            "status": "passed",
        },
    )


def test_trace_evaluator_accepts_complete_trace() -> None:
    task = make_task()
    run = make_run()

    add_lifecycle_events(run)
    add_execution_event(run)

    result = evaluate_trace(task, run)

    assert result.passed is True
    assert result.score == 1.0
    assert result.evaluator == "trace"


def test_trace_evaluator_rejects_empty_trace() -> None:
    task = make_task()
    run = make_run()

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert result.score == 0.0
    assert result.reason == "Execution trace is empty."


def test_trace_evaluator_requires_agent_start() -> None:
    task = make_task()
    run = make_run()

    run.trace.add_event(
        event_type="agent_end",
        name="agent_finished",
    )
    add_execution_event(run)

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert "agent_start" in result.reason


def test_trace_evaluator_requires_agent_end() -> None:
    task = make_task()
    run = make_run()

    run.trace.add_event(
        event_type="agent_start",
        name="agent_started",
    )
    add_execution_event(run)

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert "agent_end" in result.reason


def test_trace_evaluator_requires_execution_evidence() -> None:
    task = make_task()
    run = make_run()

    add_lifecycle_events(run)

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert "no target-agent execution evidence" in result.reason


def test_trace_evaluator_matches_report_execution_count() -> None:
    task = make_task()
    run = make_run()

    add_lifecycle_events(run)
    add_execution_event(run, "trace-001")

    run.output = {
        "report": {
            "executions": [
                {
                    "trace_id": "trace-001",
                }
            ]
        }
    }

    result = evaluate_trace(task, run)

    assert result.passed is True
    assert result.metrics["expected_execution_count"] == 1.0
    assert result.metrics["execution_event_count"] == 1.0
    assert result.metrics["execution_count_matches"] == 1.0


def test_trace_evaluator_rejects_execution_count_mismatch() -> None:
    task = make_task()
    run = make_run()

    add_lifecycle_events(run)
    add_execution_event(run)

    run.output = {
        "report": {
            "executions": [
                {
                    "trace_id": "trace-001",
                },
                {
                    "trace_id": "trace-002",
                },
            ]
        }
    }

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert result.metrics["expected_execution_count"] == 2.0
    assert result.metrics["execution_event_count"] == 1.0
    assert result.metrics["execution_count_matches"] == 0.0


def test_trace_evaluator_ignores_unrelated_message_events() -> None:
    task = make_task()
    run = make_run()

    add_lifecycle_events(run)

    run.trace.add_event(
        event_type="message",
        name="unrelated_message",
        data={
            "source": "other",
            "evidence_type": "other",
        },
    )

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert result.metrics["execution_event_count"] == 0.0


def test_trace_evaluator_reports_trace_metrics() -> None:
    task = make_task()
    run = make_run()

    add_lifecycle_events(run)
    add_execution_event(run)

    result = evaluate_trace(task, run)

    assert result.metrics["trace_event_count"] == 3.0
    assert result.metrics["execution_event_count"] == 1.0
    assert result.metrics["has_agent_start"] == 1.0
    assert result.metrics["has_agent_end"] == 1.0