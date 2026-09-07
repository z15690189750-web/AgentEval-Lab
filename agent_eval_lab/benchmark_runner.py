from __future__ import annotations

from agent_eval_lab.benchmark import Benchmark
from agent_eval_lab.core.models import Run
from agent_eval_lab.run import Runner


class BenchmarkRunner:
    """
    Execute every task in a benchmark through a Runner.

    This class is responsible only for batch execution.
    It does not evaluate results or calculate metrics.
    """

    def __init__(self, runner: Runner) -> None:
        self.runner = runner

    def run(self, benchmark: Benchmark) -> list[Run]:
        """Execute all tasks in the benchmark and return their runs."""

        return [
            self.runner.run(task)
            for task in benchmark.tasks
        ]