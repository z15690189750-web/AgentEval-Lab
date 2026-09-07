from agent_eval_lab.benchmark import Benchmark
from agent_eval_lab.benchmark_runner import BenchmarkRunner
from agent_eval_lab.core.models import Task
from agent_eval_lab.run import Adapter, Runner


def make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        name=f"task {task_id}",
        description="Benchmark runner test task.",
        input={"value": task_id},
        expected_output={"result": task_id},
    )


def make_benchmark() -> Benchmark:
    return Benchmark(
        name="basic",
        tasks=(
            make_task("task-001"),
            make_task("task-002"),
            make_task("task-003"),
        ),
    )


def agent(input_data: dict[str, str]) -> dict[str, str]:
    return {"result": input_data["value"]}


def test_benchmark_runner_executes_all_tasks() -> None:
    benchmark = make_benchmark()

    runner = Runner(Adapter(agent))
    benchmark_runner = BenchmarkRunner(runner)

    runs = benchmark_runner.run(benchmark)

    assert len(runs) == 3
    assert [run.task_id for run in runs] == [
        "task-001",
        "task-002",
        "task-003",
    ]


def test_benchmark_runner_preserves_run_results() -> None:
    benchmark = make_benchmark()

    runner = Runner(Adapter(agent))
    benchmark_runner = BenchmarkRunner(runner)

    runs = benchmark_runner.run(benchmark)

    assert all(run.status == "success" for run in runs)
    assert [run.output for run in runs] == [
        {"result": "task-001"},
        {"result": "task-002"},
        {"result": "task-003"},
    ]


def test_benchmark_runner_preserves_failed_runs() -> None:
    benchmark = Benchmark(
        name="mixed",
        tasks=(
            make_task("task-001"),
            make_task("task-002"),
        ),
    )

    def failing_agent(input_data: dict[str, str]) -> dict[str, str]:
        if input_data["value"] == "task-002":
            raise RuntimeError("agent failed")

        return {"result": input_data["value"]}

    runner = Runner(Adapter(failing_agent))
    benchmark_runner = BenchmarkRunner(runner)

    runs = benchmark_runner.run(benchmark)

    assert len(runs) == 2
    assert runs[0].status == "success"
    assert runs[1].status == "error"
    assert runs[1].error == "agent failed"