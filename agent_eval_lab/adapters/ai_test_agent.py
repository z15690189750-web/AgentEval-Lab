from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from agent_eval_lab.core.models import Task


class AITestAgentAdapter:
    """
    Adapt the AI Test Agent CLI to AgentEval Lab.

    The adapter invokes the real AI Test Agent and returns its
    target-agent output together with evidence that can be mapped
    into the AgentEval Lab trace.

    The adapter does not create Run objects.
    Runner owns the Run lifecycle.
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
        self.cwd = Path(cwd) if cwd is not None else None
        self.timeout = timeout

    def run(
        self,
        task: Task,
    ) -> dict[str, Any]:
        """
        Execute the AI Test Agent for one evaluation task.

        The returned dictionary preserves the complete raw target-agent
        report and process-level evidence.

        It also contains ``trace_events`` describing evidence that
        AgentEval Lab can attach to its execution trace.

        These events represent observed target-agent evidence only.
        They do not invent semantic tool calls.
        """

        config = self._parse_input(task)

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

            try:
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
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    "AI Test Agent timed out after "
                    f"{self.timeout} seconds."
                ) from exc
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to execute AI Test Agent: {exc}"
                ) from exc

            report = self._load_report(report_path)

            self._validate_report(report)

            output = {
                "report": report,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "exit_code": completed.returncode,
                "executable": self.executable,
                "spec": str(config["spec"]),
                "base_url": config["base_url"],
                "mock": config["mock"],
            }

            output["trace_events"] = self._build_trace_events(
                report=report,
                stdout=completed.stdout,
                stderr=completed.stderr,
                exit_code=completed.returncode,
            )

            return output

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
            str(report_path.with_suffix(".md")),
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
        """Validate and normalize task input."""

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
        """Load the AI Test Agent JSON report."""

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

        return data

    @staticmethod
    def _validate_report(
        report: dict[str, Any],
    ) -> None:
        """Validate the minimum report contract required by the adapter."""

        if "generated_tests" not in report:
            raise ValueError(
                "AI Test Agent report is missing "
                "'generated_tests'."
            )

        if "executions" not in report:
            raise ValueError(
                "AI Test Agent report is missing "
                "'executions'."
            )

        if not isinstance(
            report["generated_tests"],
            list,
        ):
            raise ValueError(
                "AI Test Agent report field "
                "'generated_tests' must be a list."
            )

        if not isinstance(
            report["executions"],
            list,
        ):
            raise ValueError(
                "AI Test Agent report field "
                "'executions' must be a list."
            )

    @staticmethod
    def _build_trace_events(
        *,
        report: dict[str, Any],
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> list[dict[str, Any]]:
        """
        Convert real AI Test Agent evidence into trace events.

        The mapping is deliberately conservative:

        - each target execution becomes a ``message`` event;
        - target errors become ``error`` events;
        - process completion becomes an ``agent_end`` event.

        We do not label API-test executions as tool calls because the
        AI Test Agent report does not establish that semantic contract.
        """

        events: list[dict[str, Any]] = []

        executions = report.get("executions", [])

        for index, execution in enumerate(
            executions,
            start=1,
        ):
            if not isinstance(execution, dict):
                continue

            data = {
                "source": "ai_test_agent",
                "evidence_type": "test_execution",
                "index": index,
            }

            for key in (
                "trace_id",
                "test_name",
                "status",
                "exit_code",
                "http_status_code",
                "duration_seconds",
            ):
                if key in execution:
                    data[key] = execution[key]

            events.append(
                {
                    "event_type": "message",
                    "name": "ai_test_agent_execution",
                    "data": data,
                }
            )

            error = execution.get("error")

            if error:
                events.append(
                    {
                        "event_type": "error",
                        "name": "ai_test_agent_execution_error",
                        "data": {
                            "source": "ai_test_agent",
                            "evidence_type": "test_execution",
                            "index": index,
                            "error": str(error),
                        },
                    }
                )

        events.append(
            {
                "event_type": "message",
                "name": "ai_test_agent_process_result",
                "data": {
                    "source": "ai_test_agent",
                    "evidence_type": "process_result",
                    "exit_code": exit_code,
                    "generated_tests": len(
                        report.get("generated_tests", [])
                    ),
                    "executions": len(executions),
                    "stdout_present": bool(stdout),
                    "stderr_present": bool(stderr),
                },
            }
        )

        return events