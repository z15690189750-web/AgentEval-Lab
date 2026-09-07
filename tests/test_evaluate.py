from agent_eval_lab.core.models import Task
from agent_eval_lab.evaluate import evaluate_output, evaluate_schema
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