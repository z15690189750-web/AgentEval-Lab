import pytest

from agent_eval_lab.core.models import EvaluationResult, Run, Task
from agent_eval_lab.evaluate import (
    evaluate_output,
    evaluate_schema,
    evaluate_trace,
)
from agent_eval_lab.evaluation_pipeline import EvaluationPipeline
from agent_eval_lab.run import Adapter, Runner


def make_task() -> Task:
    return Task(
        id="task-001",
        name="pipeline task",
        description="Evaluation pipeline test.",
        input={"value": "ok"},
        expected_output={"status": "ok"},
        expected_schema={
            "required": ["status"],
            "types": {"status": "str"},
        },
        required_tools=["search"],
    )


def agent(input_data: dict[str, str]) -> dict[str, str]:
    return {"status": input_data["value"]}


def tool_agent(input_data: dict[str, str]) -> dict[str, str]:
    del input_data

    return {"status": "ok"}


def make_tool_run() -> Run:
    task = make_task()

    runner = Runner(Adapter(tool_agent))
    run = runner.run(task)

    run.trace.add_event(
        event_type="tool_call",
        name="search",
        data={"query": "test"},
    )

    return run


def test_pipeline_runs_all_evaluators() -> None:
    task = make_task()

    run = Runner(Adapter(agent)).run(task)

    pipeline = EvaluationPipeline(
        evaluators=(
            evaluate_output,
            evaluate_schema,
        )
    )

    results = pipeline.evaluate(task, run)

    assert len(results) == 2
    assert [result.evaluator for result in results] == [
        "output",
        "schema",
    ]
    assert all(result.passed for result in results)


def test_pipeline_can_evaluate_trace() -> None:
    task = make_task()
    run = make_tool_run()

    pipeline = EvaluationPipeline(
        evaluators=(evaluate_trace,)
    )

    results = pipeline.evaluate(task, run)

    assert len(results) == 1
    assert results[0].evaluator == "trace"
    assert results[0].passed is True


def test_pipeline_preserves_evaluator_results() -> None:
    task = make_task()

    run = Runner(
        Adapter(lambda _: {"status": "wrong"})
    ).run(task)

    def custom_evaluator(
        task: Task,
        run: Run,
    ) -> EvaluationResult:
        return EvaluationResult(
            task_id=task.id,
            run_id=run.id,
            passed=False,
            score=0.25,
            reason="Custom evaluation failed.",
            evaluator="custom",
        )

    pipeline = EvaluationPipeline(
        evaluators=(custom_evaluator,)
    )

    results = pipeline.evaluate(task, run)

    assert results[0].score == 0.25
    assert results[0].reason == "Custom evaluation failed."
    assert results[0].evaluator == "custom"


def test_pipeline_rejects_empty_evaluator_list() -> None:
    with pytest.raises(
        ValueError,
        match="At least one evaluator is required",
    ):
        EvaluationPipeline(evaluators=())