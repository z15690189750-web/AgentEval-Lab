from agent_eval_lab.core.models import Task
from agent_eval_lab.run import Adapter, Runner


def fake_agent(data: dict) -> dict:
    return {
        "result": "passed",
        "input": data,
    }


def failing_agent(data: dict) -> dict:
    raise RuntimeError("agent failed")


def make_task() -> Task:
    return Task(
        id="task-001",
        name="basic task",
        description="A simple evaluation task.",
        input={
            "value": 1,
        },
    )


def test_adapter_calls_agent() -> None:
    adapter = Adapter(fake_agent)
    task = make_task()

    result = adapter.run(task)

    assert result["result"] == "passed"
    assert result["input"] == {"value": 1}


def test_runner_returns_successful_run() -> None:
    runner = Runner(Adapter(fake_agent))
    task = make_task()

    run = runner.run(task)

    assert run.task_id == "task-001"
    assert run.status == "success"
    assert run.output["result"] == "passed"
    assert len(run.trace.events) == 2


def test_runner_records_agent_error() -> None:
    runner = Runner(Adapter(failing_agent))
    task = make_task()

    run = runner.run(task)

    assert run.status == "error"
    assert run.error == "agent failed"
    assert len(run.trace.events) == 2
    assert run.trace.events[1].event_type == "error"