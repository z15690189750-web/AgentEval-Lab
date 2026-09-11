
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent_eval_lab.core.models import Run, Task


class AITestAgentAdapter:
    """
    Adapt the AI Test Agent CLI to the AgentEval Lab Runner.

    One complete AI Test Agent CLI invocation is treated as one
    AgentEval Lab Run.

    The AI Test Agent remains an independent project. This adapter
    communicates with it only through its public CLI and JSON report.
    """

    def __init__(
        self,
        executable: str = "ai-test-agent",
        *,
        cwd: str | Path | None = None,
        timeout: float = 300.0,
    ) -> None:
        if not executable.strip():
            raise ValueError(
                "executable must not be empty."
            )

        if timeout <= 0:
            raise ValueError(
                "timeout must be greater than 0."
            )

        self.executable = executable
        self.cwd = (
            Path(cwd)
            if cwd is not None
            else None
        )
        self.timeout = timeout

    def run(
        self,
        task: Task,
        run_id: str | None = None,
    ) -> Run:
        """
        Execute the AI Test Agent for one evaluation task.

        Expected task.input structure:

            {
                "spec": "path/to/spec.md",
                "base_url": "http://127.0.0.1:8000",
                "mock": true
            }

        ``spec`` and ``base_url`` are required.
        ``mock`` is optional and defaults to False.

        The complete AI Test Agent JSON report is preserved
        in Run.output.
        """

        config = self._parse_input(task)

        run = Run(
            id=run_id or f"run-{task.id}",
            task_id=task.id,
        )

        run.trace.add_event(
            event_type="agent_start",
            name="ai_test_agent_started",
            data={
                "executable": self.executable,
                "spec": str(config["spec"]),
                "base_url": config["base_url"],
                "mock": config["mock"],
            },
        )

        try:
            with tempfile.TemporaryDirectory(
                prefix="agent-eval-",
            ) as temp_dir:
                report_path = (
                    Path(temp_dir)
                    / "agent-report.json"
                )

                command = self._build_command(
                    spec=config["spec"],
                    base_url=config["base_url"],
                    mock=config["mock"],
                    report_path=report_path,
                )

                completed = subprocess.run(
                    command,
                    cwd=self.cwd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    check=False,
                )

                report = self._load_report(
                    report_path,
                )

                self._add_execution_events(
                    run=run,
                    report=report,
                )

                output = {
                    "report": report,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "exit_code": completed.returncode,
                }

                run.trace.add_event(
                    event_type="agent_end",
                    name="ai_test_agent_finished",
                    data={
                        "exit_code": completed.returncode,
                        "generated_tests": len(
                            report.get(
                                "generated_tests",
                                [],
                            )
                        ),
                        "executions": len(
                            report.get(
                                "executions",
                                [],
                            )
                        ),
                    },
                )

                if completed.returncode == 0:
                    run.finish(
                        status="success",
                        output=output,
                    )
                else:
                    run.finish(
                        status="error",
                        output=output,
                        error=(
                            "AI Test Agent exited with "
                            f"code {completed.returncode}."
                        ),
                    )

        except subprocess.TimeoutExpired as exc:
            run.trace.add_event(
                event_type="error",
                name="ai_test_agent_timeout",
                data={
                    "timeout_seconds": self.timeout,
                    "error": str(exc),
                },
            )

            run.finish(
                status="error",
                error=(
                    "AI Test Agent timed out after "
                    f"{self.timeout} seconds."
                ),
            )

        except (
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
            ValueError,
        ) as exc:
            run.trace.add_event(
                event_type="error",
                name="ai_test_agent_error",
                data={
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

            run.finish(
                status="error",
                error=str(exc),
            )

        return run

    def _build_command(
        self,
        *,
        spec: Path,
        base_url: str,
        mock: bool,
        report_path: Path,
    ) -> list[str]:
        """Build the AI Test Agent CLI command."""

        command = [
            self.executable,
            str(spec),
            "--base-url",
            base_url,
            "--report",
            str(
                report_path.with_suffix(".md")
            ),
            "--json-report",
            str(report_path),
        ]

        if mock:
            command.append("--mock")

        return command

    @staticmethod
    def _parse_input(
        task: Task,
    ) -> dict[str, Any]:
        """Validate and normalize Task input."""

        if not isinstance(task.input, dict):
            raise ValueError(
                "AI Test Agent task input must be a dictionary."
            )

        spec_value = task.input.get("spec")
        base_url = task.input.get("base_url")
        mock = task.input.get("mock", False)

        if not isinstance(spec_value, str):
            raise ValueError(
                "Task input 'spec' must be a string."
            )

        if not spec_value.strip():
            raise ValueError(
                "Task input 'spec' must not be empty."
            )

        if not isinstance(base_url, str):
            raise ValueError(
                "Task input 'base_url' must be a string."
            )

        if not base_url.strip():
            raise ValueError(
                "Task input 'base_url' must not be empty."
            )

        if not isinstance(mock, bool):
            raise ValueError(
                "Task input 'mock' must be a boolean."
            )

        return {
            "spec": Path(spec_value),
            "base_url": base_url,
            "mock": mock,
        }

    @staticmethod
    def _load_report(
        report_path: Path,
    ) -> dict[str, Any]:
        """Load and validate the AI Test Agent JSON report."""

        if not report_path.exists():
            raise FileNotFoundError(
                "AI Test Agent did not produce the expected "
                f"JSON report: {report_path}"
            )

        data = json.loads(
            report_path.read_text(
                encoding="utf-8",
            )
        )

        if not isinstance(data, dict):
            raise ValueError(
                "AI Test Agent JSON report must contain "
                "a top-level object."
            )

        if "generated_tests" not in data:
            raise ValueError(
                "AI Test Agent report is missing "
                "'generated_tests'."
            )

        if "executions" not in data:
            raise ValueError(
                "AI Test Agent report is missing "
                "'executions'."
            )

        if not isinstance(
            data["generated_tests"],
            list,
        ):
            raise ValueError(
                "AI Test Agent report field "
                "'generated_tests' must be a list."
            )

        if not isinstance(
            data["executions"],
            list,
        ):
            raise ValueError(
                "AI Test Agent report field "
                "'executions' must be a list."
            )

        return data

    @staticmethod
    def _add_execution_events(
        *,
        run: Run,
        report: dict[str, Any],
    ) -> None:
        """
        Preserve target-agent execution information in the trace.

        The execution records remain raw data inside Run.output.
        TraceEvent is used only as a lightweight observation layer,
        so target-agent execution records are represented as
        ``message`` events rather than introducing a new event type.
        """

        for execution in report["executions"]:
            if not isinstance(execution, dict):
                continue

            run.trace.add_event(
                event_type="message",
                name=str(
                    execution.get(
                        "test_name",
                        "unknown_test",
                    )
                ),
                data={
                    "trace_id": execution.get(
                        "trace_id"
                    ),
                    "status": execution.get(
                        "status"
                    ),
                    "duration_seconds": execution.get(
                        "duration_seconds"
                    ),
                    "exit_code": execution.get(
                        "exit_code"
                    ),
                    "http_status_code": execution.get(
                        "http_status_code"
                    ),
                },
            )

