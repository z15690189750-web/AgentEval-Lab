from __future__ import annotations

from typing import Any

from agent_eval_lab.core.models import EvaluationResult, Run, Task


def _base_failure_result(
    *,
    task: Task,
    run: Run,
    evaluator: str,
    reason: str,
) -> EvaluationResult:
    """Create a standard failed evaluation result."""

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=False,
        score=0.0,
        reason=reason,
        evaluator=evaluator,
    )


def _get_report(run: Run) -> dict[str, Any] | None:
    """
    Return the target-agent report from a Run.

    Supports both:

        run.output = report

    and:

        run.output = {"report": report, ...}
    """

    if not isinstance(run.output, dict):
        return None

    nested_report = run.output.get("report")

    if isinstance(nested_report, dict):
        return nested_report

    return run.output


def evaluate_output(
    task: Task,
    run: Run,
) -> EvaluationResult:
    """Evaluate whether the run output matches the expected output."""

    evaluator = "output"

    if run.status == "error":
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent execution did not succeed.",
        )

    if task.expected_output is None:
        passed = run.output is not None

        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=passed,
            score=1.0 if passed else 0.0,
            reason=(
                "Agent produced an output."
                if passed
                else "Agent produced no output."
            ),
            evaluator=evaluator,
        )

    passed = run.output == task.expected_output

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=passed,
        score=1.0 if passed else 0.0,
        reason=(
            "Output matches the expected output."
            if passed
            else "Output does not match the expected output."
        ),
        evaluator=evaluator,
    )


def evaluate_schema(
    task: Task,
    run: Run,
) -> EvaluationResult:
    """
    Evaluate whether the output satisfies the expected schema.

    Supported schema:

        {
            "required": ["field_a", "field_b"],
            "types": {
                "field_a": "str",
                "field_b": "int"
            }
        }
    """

    evaluator = "schema"

    if run.status == "error":
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent execution did not succeed.",
        )

    if task.expected_schema is None:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=True,
            score=1.0,
            reason="No expected schema was configured.",
            evaluator=evaluator,
        )

    if not isinstance(run.output, dict):
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent output is not a dictionary.",
        )

    required = task.expected_schema.get("required", [])
    types = task.expected_schema.get("types", {})

    if not isinstance(required, list):
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Schema 'required' must be a list.",
        )

    if not isinstance(types, dict):
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Schema 'types' must be a dictionary.",
        )

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
            reason=(
                "Missing required fields: "
                + ", ".join(missing)
            ),
            evaluator=evaluator,
        )

    invalid_types: list[str] = []

    for field, expected_type in types.items():
        if field not in run.output:
            continue

        if not _matches_type(
            value=run.output[field],
            expected_type=expected_type,
        ):
            invalid_types.append(field)

    if invalid_types:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason=(
                "Invalid field types: "
                + ", ".join(invalid_types)
            ),
            evaluator=evaluator,
        )

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=True,
        score=1.0,
        reason="Output matches the expected schema.",
        evaluator=evaluator,
    )


def _matches_type(
    *,
    value: Any,
    expected_type: Any,
) -> bool:
    """Check a lightweight named Python type contract."""

    if isinstance(expected_type, type):
        return isinstance(value, expected_type)

    if not isinstance(expected_type, str):
        return True

    type_map = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": (int, float),
        "bool": bool,
        "boolean": bool,
        "list": list,
        "dict": dict,
        "object": dict,
    }

    python_type = type_map.get(
        expected_type.lower()
    )

    if python_type is None:
        raise ValueError(
            f"Unsupported schema type: {expected_type}"
        )

    return isinstance(value, python_type)


def evaluate_trace(
    task: Task,
    run: Run,
) -> EvaluationResult:
    """
    Evaluate execution trace requirements.

    Checks:

    - execution errors;
    - lifecycle evidence;
    - required tools;
    - forbidden tools;
    - optional AI Test Agent execution evidence.

    Trace evaluation operates on the trace itself. A manually
    constructed Run may still be evaluated while its status remains
    ``running``.
    """

    evaluator = "trace"

    if run.status == "error":
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent execution did not succeed.",
        )

    events = run.trace.events

    if not events:
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Execution trace is empty.",
        )

    event_types = [
        event.event_type
        for event in events
    ]

    has_start = "agent_start" in event_types
    has_end = "agent_end" in event_types

    metrics: dict[str, float] = {
        "trace_event_count": float(len(events)),
        "has_agent_start": 1.0 if has_start else 0.0,
        "has_agent_end": 1.0 if has_end else 0.0,
        "required_tool_count": float(
            len(task.required_tools)
        ),
        "forbidden_tool_count": 0.0,
        "missing_tool_count": 0.0,
    }

    if not has_start:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="Execution trace is missing agent_start.",
            metrics=metrics,
            evaluator=evaluator,
        )

    if not has_end:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason="Execution trace is missing agent_end.",
            metrics=metrics,
            evaluator=evaluator,
        )

    tool_names = {
        event.name
        for event in events
        if event.event_type == "tool_call"
    }

    missing_tools = [
        tool
        for tool in task.required_tools
        if tool not in tool_names
    ]

    forbidden_tools = [
        tool
        for tool in task.forbidden_tools
        if tool in tool_names
    ]

    metrics["tool_call_count"] = float(len(tool_names))
    metrics["missing_tool_count"] = float(
        len(missing_tools)
    )
    metrics["forbidden_tool_count"] = float(
        len(forbidden_tools)
    )

    if missing_tools:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason=(
                "Missing required tools: "
                + ", ".join(missing_tools)
            ),
            metrics=metrics,
            evaluator=evaluator,
        )

    if forbidden_tools:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason=(
                "Forbidden tools were used: "
                + ", ".join(forbidden_tools)
            ),
            metrics=metrics,
            evaluator=evaluator,
        )

    execution_events = [
        event
        for event in events
        if (
            event.event_type == "message"
            and event.data.get("source") == "ai_test_agent"
            and event.data.get("evidence_type")
            == "test_execution"
        )
    ]

    process_events = [
        event
        for event in events
        if (
            event.event_type == "message"
            and event.data.get("source") == "ai_test_agent"
            and event.data.get("evidence_type")
            == "process_result"
        )
    ]

    report = _get_report(run)

    expected_execution_count: int | None = None

    if isinstance(report, dict):
        executions = report.get("executions")

        if isinstance(executions, list):
            expected_execution_count = len(executions)

    actual_execution_count = len(execution_events)

    metrics["execution_event_count"] = float(
        actual_execution_count
    )

    metrics["process_result_event_count"] = float(
        len(process_events)
    )

    if expected_execution_count is not None:
        metrics["expected_execution_count"] = float(
            expected_execution_count
        )

        execution_count_matches = (
            actual_execution_count
            == expected_execution_count
        )

        metrics["execution_count_matches"] = (
            1.0
            if execution_count_matches
            else 0.0
        )

        if not execution_count_matches:
            return EvaluationResult(
                task_id=task.id,
                run_id=run.id,
                passed=False,
                score=0.0,
                reason=(
                    "Trace execution evidence count "
                    f"({actual_execution_count}) does not match "
                    "the target report execution count "
                    f"({expected_execution_count})."
                ),
                metrics=metrics,
                evaluator=evaluator,
            )

    if (
        not task.required_tools
        and not task.forbidden_tools
        and actual_execution_count == 0
        and report is None
    ):
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.0,
            reason=(
                "Execution trace contains lifecycle events "
                "but no target-agent execution evidence."
            ),
            metrics=metrics,
            evaluator=evaluator,
        )

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=True,
        score=1.0,
        reason=(
            "Execution trace satisfies lifecycle and "
            "tool requirements."
        ),
        metrics=metrics,
        evaluator=evaluator,
    )


def evaluate_execution_outcome(
    task: Task,
    run: Run,
) -> EvaluationResult:
    """
    Evaluate target-agent execution outcomes.

    An execution passes when:

    - status == ``passed``;
    - and expected/actual HTTP status codes agree when both exist.
    """

    evaluator = "execution_outcome"

    if run.status == "error":
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent execution did not succeed.",
        )

    report = _get_report(run)

    if report is None:
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent output is not a dictionary.",
        )

    executions = report.get("executions")

    if not isinstance(executions, list):
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason=(
                "Agent report executions list is missing or invalid."
            ),
        )

    if not executions:
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent report executions list is empty.",
        )

    passed_count = 0
    failed_count = 0
    comparable_count = 0
    status_mismatch_count = 0

    for execution in executions:
        if not isinstance(execution, dict):
            failed_count += 1
            continue

        status = execution.get("status")

        expected_status = execution.get(
            "expected_status_code"
        )

        actual_status = execution.get(
            "http_status_code"
        )

        execution_passed = status == "passed"

        if (
            isinstance(expected_status, int)
            and isinstance(actual_status, int)
        ):
            comparable_count += 1

            if expected_status != actual_status:
                status_mismatch_count += 1
                execution_passed = False

        if execution_passed:
            passed_count += 1
        else:
            failed_count += 1

    total = len(executions)
    pass_rate = passed_count / total

    metrics = {
        "execution_total": float(total),
        "execution_passed": float(passed_count),
        "execution_failed": float(failed_count),
        "execution_pass_rate": pass_rate,
        "comparable_execution_count": float(
            comparable_count
        ),
        "status_mismatch_count": float(
            status_mismatch_count
        ),
        "http_status_mismatch_count": float(
            status_mismatch_count
        ),
    }

    passed = failed_count == 0

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=passed,
        score=pass_rate,
        reason=(
            "All target-agent executions passed."
            if passed
            else (
                f"{failed_count} of {total} target-agent "
                "execution(s) failed."
            )
        ),
        metrics=metrics,
        evaluator=evaluator,
    )


def evaluate_generation_completeness(
    task: Task,
    run: Run,
) -> EvaluationResult:
    """
    Evaluate basic generated-test completeness.

    Checks:

    - generated_tests is present and is a list;
    - generated count is not lower than planned count when
      test_cases are available;
    - every generated test contains scenario_type.
    """

    evaluator = "generation_completeness"

    if run.status == "error":
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent execution did not succeed.",
        )

    report = _get_report(run)

    if report is None:
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason="Agent output is not a dictionary.",
        )

    generated_tests = report.get("generated_tests")

    if not isinstance(generated_tests, list):
        return _base_failure_result(
            task=task,
            run=run,
            evaluator=evaluator,
            reason=(
                "Agent report generated_tests must be a list."
            ),
        )

    test_cases = report.get("test_cases")

    planned_test_case_count: int | None = None

    if isinstance(test_cases, list):
        planned_test_case_count = len(test_cases)

    generated_test_count = len(generated_tests)

    missing_scenario_count = sum(
        not isinstance(test, dict)
        or not test.get("scenario_type")
        for test in generated_tests
    )

    missing_generation_count = 0

    if planned_test_case_count is not None:
        missing_generation_count = max(
            planned_test_case_count
            - generated_test_count,
            0,
        )

    metrics: dict[str, float] = {
        "generated_test_count": float(
            generated_test_count
        ),
        "missing_scenario_count": float(
            missing_scenario_count
        ),
        "missing_generation_count": float(
            missing_generation_count
        ),
    }

    if planned_test_case_count is not None:
        metrics["planned_test_case_count"] = float(
            planned_test_case_count
        )

    if missing_generation_count > 0:
        score = (
            generated_test_count
            / planned_test_case_count
            if planned_test_case_count > 0
            else 0.0
        )

        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=score,
            reason=(
                "Generated test count is lower than the "
                "planned test case count."
            ),
            metrics=metrics,
            evaluator=evaluator,
        )

    if missing_scenario_count > 0:
        score = (
            1.0
            - missing_scenario_count
            / generated_test_count
            if generated_test_count > 0
            else 0.0
        )

        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=score,
            reason=(
                f"{missing_scenario_count} generated test(s) "
                "are missing scenario_type."
            ),
            metrics=metrics,
            evaluator=evaluator,
        )

    return EvaluationResult(
        task_id=task.id,
        run_id=run.id,
        passed=True,
        score=1.0,
        reason=(
            "Generated tests meet the basic completeness "
            "requirements."
        ),
        metrics=metrics,
        evaluator=evaluator,
    )