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


def evaluate_schema(task: Task, run: Run) -> EvaluationResult:
    """
    Evaluate the structure of an agent output.

    The schema format is intentionally lightweight:

    {
        "required": ["status", "test_cases"],
        "types": {
            "status": "str",
            "test_cases": "list"
        }
    }
    """

    if run.status != "success":
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="Agent execution did not succeed.",
            evaluator="schema",
        )

    schema = task.expected_schema

    if schema is None:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="No expected schema was defined.",
            evaluator="schema",
        )

    if not isinstance(run.output, dict):
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="Agent output is not a dictionary.",
            evaluator="schema",
        )

    required = schema.get("required", [])
    types = schema.get("types", {})

    missing = [
        field
        for field in required
        if field not in run.output
    ]

    if missing:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason=f"Missing required fields: {', '.join(missing)}.",
            evaluator="schema",
        )

    invalid_types = []

    expected_types = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    for field, type_name in types.items():
        if field not in run.output:
            continue

        expected_type = expected_types.get(type_name)

        if expected_type is None:
            invalid_types.append(
                f"{field}: unsupported type '{type_name}'"
            )
            continue

        if not isinstance(run.output[field], expected_type):
            invalid_types.append(
                f"{field}: expected {type_name}, "
                f"got {type(run.output[field]).__name__}"
            )

    if invalid_types:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="Invalid field types: " + "; ".join(invalid_types),
            evaluator="schema",
        )

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=True,
        score=1.0,
        reason="Output matches the expected schema.",
        evaluator="schema",
    )


def evaluate_trace(task: Task, run: Run) -> EvaluationResult:
    """
    Evaluate whether the agent followed the required tool-use constraints.

    The evaluator checks:

    1. The agent execution succeeded.
    2. Required tools were called.
    3. Forbidden tools were not called.
    4. The number of tool calls is recorded.

    Tool names are read from trace events with event_type="tool_call".
    """

    if run.status != "success":
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="Agent execution did not succeed.",
            evaluator="trace",
        )

    tool_calls = [
        event.name
        for event in run.trace.events
        if event.event_type == "tool_call"
    ]

    required_tools = set(task.required_tools)
    forbidden_tools = set(task.forbidden_tools)
    actual_tools = set(tool_calls)

    missing_tools = sorted(required_tools - actual_tools)
    forbidden_called = sorted(forbidden_tools & actual_tools)

    metrics = {
        "tool_call_count": float(len(tool_calls)),
        "required_tool_count": float(len(required_tools)),
        "missing_tool_count": float(len(missing_tools)),
        "forbidden_tool_count": float(len(forbidden_called)),
    }

    if missing_tools:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason=(
                "Required tools were not called: "
                + ", ".join(missing_tools)
                + "."
            ),
            metrics=metrics,
            evaluator="trace",
        )

    if forbidden_called:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason=(
                "Forbidden tools were called: "
                + ", ".join(forbidden_called)
                + "."
            ),
            metrics=metrics,
            evaluator="trace",
        )

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=True,
        score=1.0,
        reason="Tool-use trace satisfies the task constraints.",
        metrics=metrics,
        evaluator="trace",
    )