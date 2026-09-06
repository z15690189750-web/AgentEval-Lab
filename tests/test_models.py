from agent_eval_lab.core.models import Run, Task, Trace


def test_task_can_be_created() -> None:
    task = Task(
        id="task-001",
        name="login test",
        description="Test successful login.",
        input={
            "username": "alice",
            "password": "correct-password",
        },
    )

    assert task.id == "task-001"
    assert task.name == "login test"


def test_trace_can_record_events() -> None:
    trace = Trace()

    trace.add_event(
        event_type="agent_start",
        name="agent_started",
    )

    trace.add_event(
        event_type="tool_call",
        name="pytest",
        data={
            "command": "pytest tests/test_login.py",
        },
    )

    assert len(trace.events) == 2
    assert trace.events[0].event_type == "agent_start"
    assert trace.events[1].event_type == "tool_call"


def test_run_can_finish_successfully() -> None:
    run = Run(
        id="run-001",
        task_id="task-001",
    )

    assert run.status == "running"
    assert run.finished_at is None

    run.finish(
        status="success",
        output={
            "result": "passed",
        },
    )

    assert run.status == "success"
    assert run.output == {
        "result": "passed",
    }
    assert run.finished_at is not None


def test_run_can_record_error() -> None:
    run = Run(
        id="run-002",
        task_id="task-001",
    )

    run.finish(
        status="error",
        error="Agent execution failed.",
    )

    assert run.status == "error"
    assert run.error == "Agent execution failed."
    assert run.finished_at is not None