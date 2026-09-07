import pytest

from agent_eval_lab.core.models import Task
from agent_eval_lab.evaluate import (
    evaluate_output,
    evaluate_schema,
    evaluate_trace,
)
from agent_eval_lab.run import Adapter, Runner


def successful_agent(data: dict) -> dict:
    return {
        "result": "passed",
    }


def wrong_agent(data: dict) -> dict:
    return {
        "result": "failed",
    }


def failing_agent(data: dict) -> dict:
    raise RuntimeError("agent failed")


def structured_agent(data: dict) -> dict:
    return {
        "status": "passed",
        "test_cases": [],
    }


def missing_field_agent(data: dict) -> dict:
    return {
        "status": "passed",
    }


def wrong_type_agent(data: dict) -> dict:
    return {
        "status": "passed",
        "test_cases": "not-a-list",
    }


def make_task() -> Task:
    return Task(
        id="task-001",
        name="login task",
        description="Check login result.",
        input={
            "username": "alice",
        },
        expected_output={
            "result": "passed",
        },
    )


def make_schema_task() -> Task:
    return Task(
        id="task-schema-001",
        name="structured output task",
        description="Check structured agent output.",
        input={
            "value": 1,
        },
        expected_schema={
            "required": [
                "status",
                "test_cases",
            ],
            "types": {
                "status": "str",
                "test_cases": "list",
            },
        },
    )


def make_trace_task() -> Task:
    return Task(
        id="task-trace-001",
        name="tool use task",
        description="Check required and forbidden tool usage.",
        input={
            "query": "test",
        },
        required_tools=[
            "search",
            "execute",
        ],
        forbidden_tools=[
            "delete",
        ],
    )


def make_tool_run(task: Task, tools: list[str]):
    run = Runner(
        Adapter(successful_agent)
    ).run(task)

    for tool in tools:
        run.trace.add_event(
            event_type="tool_call",
            name=tool,
        )

    return run


def test_output_evaluator_passes_correct_output() -> None:
    task = make_task()
    run = Runner(Adapter(successful_agent)).run(task)

    result = evaluate_output(task, run)

    assert result.passed is True
    assert result.score == 1.0
    assert result.evaluator == "output"


def test_output_evaluator_rejects_wrong_output() -> None:
    task = make_task()
    run = Runner(Adapter(wrong_agent)).run(task)

    result = evaluate_output(task, run)

    assert result.passed is False
    assert result.score == 0.0


def test_output_evaluator_rejects_failed_run() -> None:
    task = make_task()
    run = Runner(Adapter(failing_agent)).run(task)

    result = evaluate_output(task, run)

    assert result.passed is False
    assert result.score == 0.0
    assert result.reason == "Agent execution did not succeed."


def test_schema_evaluator_accepts_valid_structure() -> None:
    task = make_schema_task()
    run = Runner(Adapter(structured_agent)).run(task)

    result = evaluate_schema(task, run)

    assert result.passed is True
    assert result.score == 1.0
    assert result.evaluator == "schema"


def test_schema_evaluator_rejects_missing_field() -> None:
    task = make_schema_task()
    run = Runner(Adapter(missing_field_agent)).run(task)

    result = evaluate_schema(task, run)

    assert result.passed is False
    assert result.score == 0.0
    assert "Missing required fields" in result.reason


def test_schema_evaluator_rejects_wrong_type() -> None:
    task = make_schema_task()
    run = Runner(Adapter(wrong_type_agent)).run(task)

    result = evaluate_schema(task, run)

    assert result.passed is False
    assert result.score == 0.0
    assert "Invalid field types" in result.reason


def test_trace_evaluator_accepts_required_tools() -> None:
    task = make_trace_task()
    run = make_tool_run(
        task,
        ["search", "execute"],
    )

    result = evaluate_trace(task, run)

    assert result.passed is True
    assert result.score == 1.0
    assert result.evaluator == "trace"
    assert result.metrics["tool_call_count"] == 2.0


def test_trace_evaluator_rejects_missing_required_tool() -> None:
    task = make_trace_task()
    run = make_tool_run(
        task,
        ["search"],
    )

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert result.score == 0.0
    assert "execute" in result.reason
    assert result.metrics["missing_tool_count"] == 1.0


def test_trace_evaluator_rejects_forbidden_tool() -> None:
    task = make_trace_task()
    run = make_tool_run(
        task,
        ["search", "execute", "delete"],
    )

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert result.score == 0.0
    assert "delete" in result.reason
    assert result.metrics["forbidden_tool_count"] == 1.0


def test_trace_evaluator_rejects_failed_run() -> None:
    task = make_trace_task()
    run = Runner(Adapter(failing_agent)).run(task)

    result = evaluate_trace(task, run)

    assert result.passed is False
    assert result.score == 0.0
    assert result.reason == "Agent execution did not succeed."