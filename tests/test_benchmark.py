import pytest

from agent_eval_lab.benchmark import Benchmark
from agent_eval_lab.core.models import Task


def make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        name=f"task {task_id}",
        description="Benchmark test task.",
        input={"value": task_id},
    )


def make_benchmark() -> Benchmark:
    return Benchmark(
        name="basic",
        tasks=(
            make_task("task-001"),
            make_task("task-002"),
        ),
    )


def test_benchmark_contains_tasks() -> None:
    benchmark = make_benchmark()

    assert len(benchmark) == 2
    assert benchmark.tasks[0].id == "task-001"
    assert benchmark.tasks[1].id == "task-002"


def test_benchmark_get_task() -> None:
    benchmark = make_benchmark()

    task = benchmark.get_task("task-002")

    assert task.id == "task-002"


def test_benchmark_get_task_raises_for_unknown_id() -> None:
    benchmark = make_benchmark()

    with pytest.raises(KeyError):
        benchmark.get_task("unknown")


def test_benchmark_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="Benchmark name"):
        Benchmark(
            name="",
            tasks=(make_task("task-001"),),
        )


def test_benchmark_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValueError, match="task IDs must be unique"):
        Benchmark(
            name="duplicate",
            tasks=(
                make_task("task-001"),
                make_task("task-001"),
            ),
        )