from agent_eval_lab.core.models import Task
from agent_eval_lab.evaluate import evaluate_output
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