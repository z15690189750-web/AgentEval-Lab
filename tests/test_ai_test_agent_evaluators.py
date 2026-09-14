from agent_eval_lab.core.models import Run, Task
from agent_eval_lab.evaluate import (
    evaluate_execution_outcome,
    evaluate_generation_completeness,
)


def make_task() -> Task:
    return Task(
        id="task-1",
        name="Test task",
        description="Test AI Test Agent evaluation.",
        input={},
    )


def make_run(output: dict) -> Run:
    run = Run(
        id="run-1",
        task_id="task-1",
    )
    run.finish(
        status="success",
        output=output,
    )
    return run


def test_execution_outcome_records_execution_metrics() -> None:
    task = make_task()

    run = make_run(
        {
            "executions": [
                {
                    "status": "passed",
                    "expected_status_code": 200,
                    "http_status_code": 200,
                },
                {
                    "status": "failed",
                    "expected_status_code": 200,
                    "http_status_code": 400,
                },
            ]
        }
    )

    result = evaluate_execution_outcome(task, run)

    assert result.evaluator == "execution_outcome"
    assert result.passed is False
    assert result.score == 0.5

    assert result.metrics["execution_total"] == 2.0
    assert result.metrics["execution_passed"] == 1.0
    assert result.metrics["execution_failed"] == 1.0
    assert result.metrics["execution_pass_rate"] == 0.5
    assert result.metrics["http_status_mismatch_count"] == 1.0


def test_execution_outcome_passes_when_all_executions_pass() -> None:
    task = make_task()

    run = make_run(
        {
            "executions": [
                {
                    "status": "passed",
                    "expected_status_code": 200,
                    "http_status_code": 200,
                },
                {
                    "status": "passed",
                    "expected_status_code": 201,
                    "http_status_code": 201,
                },
            ]
        }
    )

    result = evaluate_execution_outcome(task, run)

    assert result.passed is True
    assert result.score == 1.0
    assert result.metrics["execution_pass_rate"] == 1.0
    assert result.metrics["http_status_mismatch_count"] == 0.0


def test_execution_outcome_rejects_missing_executions() -> None:
    task = make_task()

    run = make_run(
        {
            "generated_tests": [],
        }
    )

    result = evaluate_execution_outcome(task, run)

    assert result.passed is False
    assert result.score == 0.0
    assert "executions list" in result.reason


def test_generation_completeness_passes_for_complete_generation() -> None:
    task = make_task()

    run = make_run(
        {
            "test_cases": [
                {"id": "case-1"},
                {"id": "case-2"},
            ],
            "generated_tests": [
                {"scenario_type": "normal"},
                {"scenario_type": "edge"},
            ],
        }
    )

    result = evaluate_generation_completeness(task, run)

    assert result.evaluator == "generation_completeness"
    assert result.passed is True
    assert result.score == 1.0

    assert result.metrics["planned_test_case_count"] == 2.0
    assert result.metrics["generated_test_count"] == 2.0
    assert result.metrics["missing_generation_count"] == 0.0
    assert result.metrics["missing_scenario_count"] == 0.0


def test_generation_completeness_detects_missing_generated_tests() -> None:
    task = make_task()

    run = make_run(
        {
            "test_cases": [
                {"id": "case-1"},
                {"id": "case-2"},
                {"id": "case-3"},
            ],
            "generated_tests": [
                {"scenario_type": "normal"},
            ],
        }
    )

    result = evaluate_generation_completeness(task, run)

    assert result.passed is False
    assert result.metrics["planned_test_case_count"] == 3.0
    assert result.metrics["generated_test_count"] == 1.0
    assert result.metrics["missing_generation_count"] == 2.0


def test_generation_completeness_detects_missing_scenario_type() -> None:
    task = make_task()

    run = make_run(
        {
            "test_cases": [
                {"id": "case-1"},
            ],
            "generated_tests": [
                {},
            ],
        }
    )

    result = evaluate_generation_completeness(task, run)

    assert result.passed is False
    assert result.metrics["missing_scenario_count"] == 1.0