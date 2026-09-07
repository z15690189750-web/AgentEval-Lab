from agent_eval_lab.core.models import Task
from agent_eval_lab.run import Adapter, Runner


def make_task() -> Task:
    return Task(
        id="task-001",
        name="runner task",
        description="Runner test task.",
        input={"value": "ok"},
    )


def agent(input_data: dict[str, str]) -> dict[str, str]:
    return {"result": input_data["value"]}


def failing_agent(_: dict[str, str]) -> None:
    raise RuntimeError("agent failed")


def test_runner_executes_agent() -> None:
    task = make_task()

    run = Runner(
        Adapter(agent)
    ).run(task)

    assert run.task_id == "task-001"
    assert run.id == "run-task-001"
    assert run.status == "success"
    assert run.output == {"result": "ok"}
    assert run.error is None


def test_runner_records_error() -> None:
    task = make_task()

    run = Runner(
        Adapter(failing_agent)
    ).run(task)

    assert run.status == "error"
    assert run.output is None
    assert run.error == "agent failed"


def test_runner_records_trace_events() -> None:
    task = make_task()

    run = Runner(
        Adapter(agent)
    ).run(task)

    assert [event.event_type for event in run.trace.events] == [
        "agent_start",
        "agent_end",
    ]


def test_runner_accepts_custom_run_id() -> None:
    task = make_task()

    run = Runner(
        Adapter(agent)
    ).run(
        task,
        run_id="experiment-001-run-003",
    )

    assert run.id == "experiment-001-run-003"


def test_runner_records_duration() -> None:
    task = make_task()

    run = Runner(
        Adapter(agent)
    ).run(task)

    assert run.finished_at is not None
    assert run.duration_ms is not None
    assert run.duration_ms >= 0.0