import pytest

from agent_eval_lab.benchmark import Benchmark
from agent_eval_lab.core.models import Task
from agent_eval_lab.evaluate import evaluate_output
from agent_eval_lab.evaluation_pipeline import EvaluationPipeline
from agent_eval_lab.experiment import Experiment
from agent_eval_lab.repeated_runner import RepeatedRunner
from agent_eval_lab.run import Adapter, Runner


def make_benchmark() -> Benchmark:
    return Benchmark(
        name="basic-benchmark",
        tasks=(
            Task(
                id="task-001",
                name="first task",
                description="First experiment task.",
                input={"value": "one"},
                expected_output={"result": "one"},
            ),
            Task(
                id="task-002",
                name="second task",
                description="Second experiment task.",
                input={"value": "two"},
                expected_output={"result": "two"},
            ),
        ),
    )


def agent(input_data: dict[str, str]) -> dict[str, str]:
    return {
        "result": input_data["value"],
    }


def test_experiment_produces_complete_result() -> None:
    benchmark = make_benchmark()

    experiment = Experiment(
        repeated_runner=RepeatedRunner(
            Runner(Adapter(agent))
        ),
        evaluation_pipeline=EvaluationPipeline(
            evaluators=[evaluate_output],
        ),
    )

    result = experiment.run(
        benchmark=benchmark,
        repetitions=2,
        experiment_id="experiment-001",
        experiment_name="basic experiment",
    )

    assert result.id == "experiment-001"
    assert result.name == "basic experiment"
    assert result.benchmark_name == "basic-benchmark"

    assert len(result.runs) == 4
    assert len(result.evaluations) == 4

    assert all(
        run.status == "success"
        for run in result.runs
    )

    assert all(
        evaluation.passed
        for evaluation in result.evaluations
    )


def test_experiment_calculates_metrics() -> None:
    benchmark = make_benchmark()

    experiment = Experiment(
        repeated_runner=RepeatedRunner(
            Runner(Adapter(agent))
        ),
        evaluation_pipeline=EvaluationPipeline(
            evaluators=[evaluate_output],
        ),
    )

    result = experiment.run(
        benchmark=benchmark,
        repetitions=3,
        experiment_id="experiment-001",
        experiment_name="metric experiment",
    )

    assert result.metrics["task_success_rate"] == 1.0
    assert result.metrics["average_score"] == 1.0
    assert result.metrics["failure_rate"] == 0.0
    assert result.metrics["average_latency_ms"] >= 0.0


def test_experiment_evaluates_every_run() -> None:
    benchmark = make_benchmark()

    experiment = Experiment(
        repeated_runner=RepeatedRunner(
            Runner(Adapter(agent))
        ),
        evaluation_pipeline=EvaluationPipeline(
            evaluators=[evaluate_output],
        ),
    )

    result = experiment.run(
        benchmark=benchmark,
        repetitions=4,
        experiment_id="experiment-001",
        experiment_name="coverage experiment",
    )

    run_ids = {run.id for run in result.runs}
    evaluation_run_ids = {
        evaluation.run_id
        for evaluation in result.evaluations
    }

    assert len(run_ids) == 8
    assert evaluation_run_ids == run_ids


def test_experiment_rejects_invalid_repetitions() -> None:
    experiment = Experiment(
        repeated_runner=RepeatedRunner(
            Runner(Adapter(agent))
        ),
        evaluation_pipeline=EvaluationPipeline(
            evaluators=[evaluate_output],
        ),
    )

    with pytest.raises(
        ValueError,
        match="repetitions must be at least 1",
    ):
        experiment.run(
            benchmark=make_benchmark(),
            repetitions=0,
            experiment_id="experiment-001",
            experiment_name="invalid experiment",
        )