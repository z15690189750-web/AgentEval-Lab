import pytest

from agent_eval_lab.core.models import Task
from agent_eval_lab.repeated_runner import RepeatedRunner
from agent_eval_lab.run import Adapter, Runner


def make_task() -> Task:
    return Task(
        id="task-001",
        name="repeated task",
        description="Repeated runner test task.",
        input={"value": "ok"},
        expected_output={"result": "ok"},
    )


def agent(input_data: dict[str, str]) -> dict[str, str]:
    return {"result": input_data["value"]}


def test_repeated_runner_executes_task_multiple_times() -> None:
    task = make_task()

    repeated_runner = RepeatedRunner(
        Runner(Adapter(agent))
    )

    runs = repeated_runner.run(
        task,
        repetitions=5,
    )

    assert len(runs) == 5
    assert all(run.status == "success" for run in runs)


def test_repeated_runner_generates_unique_run_ids() -> None:
    task = make_task()

    repeated_runner = RepeatedRunner(
        Runner(Adapter(agent))
    )

    runs = repeated_runner.run(
        task,
        repetitions=3,
    )

    assert [run.id for run in runs] == [
        "run-task-001-001",
        "run-task-001-002",
        "run-task-001-003",
    ]

    assert len({run.id for run in runs}) == 3


def test_repeated_runner_preserves_outputs() -> None:
    task = make_task()

    repeated_runner = RepeatedRunner(
        Runner(Adapter(agent))
    )

    runs = repeated_runner.run(
        task,
        repetitions=3,
    )

    assert [run.output for run in runs] == [
        {"result": "ok"},
        {"result": "ok"},
        {"result": "ok"},
    ]


def test_repeated_runner_preserves_failed_runs() -> None:
    task = make_task()

    def failing_agent(_: dict[str, str]) -> None:
        raise RuntimeError("temporary failure")

    repeated_runner = RepeatedRunner(
        Runner(Adapter(failing_agent))
    )

    runs = repeated_runner.run(
        task,
        repetitions=3,
    )

    assert len(runs) == 3
    assert all(run.status == "error" for run in runs)
    assert all(
        run.error == "temporary failure"
        for run in runs
    )


def test_repeated_runner_rejects_invalid_repetitions() -> None:
    task = make_task()

    repeated_runner = RepeatedRunner(
        Runner(Adapter(agent))
    )

    with pytest.raises(
        ValueError,
        match="repetitions must be at least 1",
    ):
        repeated_runner.run(
            task,
            repetitions=0,
        )