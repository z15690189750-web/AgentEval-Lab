
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent_eval_lab.adapters.ai_test_agent import (
    AITestAgentAdapter,
)
from agent_eval_lab.core.models import Task


def make_task() -> Task:
    """Create a minimal task for adapter tests."""

    return Task(
        id="login",
        name="Login API",
        description="Test the login API.",
        input={
            "spec": "login.md",
            "base_url": "http://127.0.0.1:8000",
            "mock": True,
        },
    )


def make_report() -> dict:
    """Create a valid fake AI Test Agent report."""

    return {
        "spec": "login.md",
        "test_cases": [
            {
                "name": "test_login_success",
            }
        ],
        "generated_tests": [
            {
                "name": "test_login_success",
            }
        ],
        "executions": [
            {
                "trace_id": "trace-001",
                "test_name": "test_login_success",
                "status": "passed",
                "started_at": "2026-09-09T00:00:00",
                "finished_at": "2026-09-09T00:00:01",
                "duration_seconds": 1.0,
                "stdout": "",
                "stderr": "",
                "exit_code": 0,
                "http_status_code": 200,
            }
        ],
    }


def write_fake_executable(
    tmp_path: Path,
    *,
    report: dict,
    exit_code: int = 0,
) -> Path:
    """
    Create a Windows .cmd fake executable.

    The command invokes the current Python interpreter so the
    tests behave consistently on Windows.
    """

    script_path = tmp_path / "fake_ai_test_agent.py"

    script_path.write_text(
        (
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n"
            "\n"
            "args = sys.argv[1:]\n"
            "index = args.index('--json-report')\n"
            "report_path = Path(args[index + 1])\n"
            f"report = {report!r}\n"
            "\n"
            "report_path.write_text(\n"
            "    json.dumps(report),\n"
            "    encoding='utf-8',\n"
            ")\n"
            f"raise SystemExit({exit_code})\n"
        ),
        encoding="utf-8",
    )

    executable_path = tmp_path / "fake_ai_test_agent.cmd"

    executable_path.write_text(
        (
            "@echo off\n"
            f'"{sys.executable}" '
            f'"{script_path}" %*\n'
        ),
        encoding="utf-8",
    )

    return executable_path


def test_adapter_executes_target_agent(
    tmp_path: Path,
) -> None:
    """Adapter should execute the target agent and return a successful Run."""

    executable = write_fake_executable(
        tmp_path,
        report=make_report(),
    )

    adapter = AITestAgentAdapter(
        executable=str(executable),
    )

    run = adapter.run(make_task())

    assert run.status == "success"
    assert run.error is None

    assert isinstance(run.output, dict)

    assert run.output["report"]["spec"] == "login.md"
    assert len(run.output["report"]["generated_tests"]) == 1
    assert len(run.output["report"]["executions"]) == 1

    assert run.output["exit_code"] == 0

    assert run.trace.events
    assert run.trace.events[0].event_type == "agent_start"
    assert run.trace.events[-1].event_type == "agent_end"


def test_adapter_preserves_raw_report_evidence(
    tmp_path: Path,
) -> None:
    """Adapter should preserve the complete target-agent report."""

    report = make_report()

    executable = write_fake_executable(
        tmp_path,
        report=report,
    )

    adapter = AITestAgentAdapter(
        executable=str(executable),
    )

    run = adapter.run(make_task())

    assert run.status == "success"

    assert run.output["report"] == report

    assert run.output["stdout"] == ""
    assert run.output["stderr"] == ""

    assert run.output["exit_code"] == 0


def test_adapter_records_execution_events(
    tmp_path: Path,
) -> None:
    """Adapter should preserve target execution information in the trace."""

    report = make_report()

    report["executions"].append(
        {
            "trace_id": "trace-002",
            "test_name": "test_login_invalid",
            "status": "failed",
            "started_at": "2026-09-09T00:00:01",
            "finished_at": "2026-09-09T00:00:02",
            "duration_seconds": 1.0,
            "stdout": "",
            "stderr": "assertion failed",
            "exit_code": 1,
            "http_status_code": 401,
        }
    )

    executable = write_fake_executable(
        tmp_path,
        report=report,
    )

    adapter = AITestAgentAdapter(
        executable=str(executable),
    )

    run = adapter.run(make_task())

    assert run.status == "success"

    message_events = [
        event
        for event in run.trace.events
        if event.event_type == "message"
    ]

    assert len(message_events) == 2

    assert message_events[0].data["trace_id"] == "trace-001"
    assert message_events[0].data["status"] == "passed"

    assert message_events[1].data["trace_id"] == "trace-002"
    assert message_events[1].data["status"] == "failed"
    assert message_events[1].data["exit_code"] == 1
    assert message_events[1].data["http_status_code"] == 401


def test_adapter_builds_expected_command(
    tmp_path: Path,
) -> None:
    """Adapter should build the expected AI Test Agent CLI command."""

    adapter = AITestAgentAdapter(
        executable="ai-test-agent",
    )

    report_path = tmp_path / "report.json"

    command = adapter._build_command(
        spec=Path("login.md"),
        base_url="http://127.0.0.1:8000",
        mock=True,
        report_path=report_path,
    )

    assert command == [
        "ai-test-agent",
        "login.md",
        "--base-url",
        "http://127.0.0.1:8000",
        "--report",
        str(tmp_path / "report.md"),
        "--json-report",
        str(report_path),
        "--mock",
    ]


def test_adapter_builds_command_without_mock(
    tmp_path: Path,
) -> None:
    """Adapter should omit --mock when mock mode is disabled."""

    adapter = AITestAgentAdapter(
        executable="ai-test-agent",
    )

    report_path = tmp_path / "report.json"

    command = adapter._build_command(
        spec=Path("login.md"),
        base_url="http://127.0.0.1:8000",
        mock=False,
        report_path=report_path,
    )

    assert command == [
        "ai-test-agent",
        "login.md",
        "--base-url",
        "http://127.0.0.1:8000",
        "--report",
        str(tmp_path / "report.md"),
        "--json-report",
        str(report_path),
    ]

    assert "--mock" not in command


def test_adapter_rejects_invalid_task_input() -> None:
    """Adapter should reject malformed task input."""

    adapter = AITestAgentAdapter(
        executable="ai-test-agent",
    )

    task = Task(
        id="invalid",
        name="Invalid",
        description="Invalid task.",
        input="not-a-dict",
    )

    with pytest.raises(ValueError, match="input"):
        adapter.run(task)


def test_adapter_rejects_missing_spec() -> None:
    """Adapter should reject task input without a specification."""

    adapter = AITestAgentAdapter(
        executable="ai-test-agent",
    )

    task = Task(
        id="invalid",
        name="Invalid",
        description="Invalid task.",
        input={
            "base_url": "http://127.0.0.1:8000",
            "mock": True,
        },
    )

    with pytest.raises(ValueError, match="spec"):
        adapter.run(task)


def test_adapter_rejects_missing_base_url() -> None:
    """Adapter should reject task input without a base URL."""

    adapter = AITestAgentAdapter(
        executable="ai-test-agent",
    )

    task = Task(
        id="invalid",
        name="Invalid",
        description="Invalid task.",
        input={
            "spec": "login.md",
            "mock": True,
        },
    )

    with pytest.raises(ValueError, match="base_url"):
        adapter.run(task)


def test_adapter_rejects_invalid_mock_value() -> None:
    """Adapter should reject a non-boolean mock value."""

    adapter = AITestAgentAdapter(
        executable="ai-test-agent",
    )

    task = Task(
        id="invalid",
        name="Invalid",
        description="Invalid task.",
        input={
            "spec": "login.md",
            "base_url": "http://127.0.0.1:8000",
            "mock": "yes",
        },
    )

    with pytest.raises(ValueError, match="mock"):
        adapter.run(task)


def test_adapter_handles_nonzero_exit_code(
    tmp_path: Path,
) -> None:
    """
    A non-zero target-agent exit should produce an error Run.

    The subprocess itself completed normally and produced a
    valid report, so the adapter records agent_end rather than
    an adapter error trace event.
    """

    report = make_report()

    executable = write_fake_executable(
        tmp_path,
        report=report,
        exit_code=1,
    )

    adapter = AITestAgentAdapter(
        executable=str(executable),
    )

    run = adapter.run(make_task())

    assert run.status == "error"
    assert run.error == "AI Test Agent exited with code 1."

    assert run.output is not None
    assert run.output["exit_code"] == 1
    assert run.output["report"] == report

    assert run.trace.events
    assert run.trace.events[0].event_type == "agent_start"
    assert run.trace.events[-1].event_type == "agent_end"

    assert not any(
        event.event_type == "error"
        for event in run.trace.events
    )


def test_adapter_handles_timeout(
    tmp_path: Path,
) -> None:
    """Adapter should convert subprocess timeout into Run error."""

    script_path = tmp_path / "slow_ai_test_agent.py"

    script_path.write_text(
        """
import time

time.sleep(10)
""".strip(),
        encoding="utf-8",
    )

    executable_path = tmp_path / "slow_ai_test_agent.cmd"

    executable_path.write_text(
        (
            "@echo off\n"
            f'"{sys.executable}" '
            f'"{script_path}" %*\n'
        ),
        encoding="utf-8",
    )

    adapter = AITestAgentAdapter(
        executable=str(executable_path),
        timeout=0.1,
    )

    run = adapter.run(make_task())

    assert run.status == "error"
    assert run.error is not None
    assert "timed out" in run.error.lower()

    assert run.trace.events
    assert run.trace.events[0].event_type == "agent_start"

    assert any(
        event.event_type == "error"
        for event in run.trace.events
    )

    assert run.trace.events[-1].event_type == "error"


def test_adapter_handles_missing_report(
    tmp_path: Path,
) -> None:
    """Missing JSON report should produce an adapter error."""

    script_path = tmp_path / "no_report.py"

    script_path.write_text(
        """
raise SystemExit(0)
""".strip(),
        encoding="utf-8",
    )

    executable_path = tmp_path / "no_report.cmd"

    executable_path.write_text(
        (
            "@echo off\n"
            f'"{sys.executable}" '
            f'"{script_path}" %*\n'
        ),
        encoding="utf-8",
    )

    adapter = AITestAgentAdapter(
        executable=str(executable_path),
    )

    run = adapter.run(make_task())

    assert run.status == "error"
    assert run.error is not None
    assert "report" in run.error.lower()

    assert run.trace.events
    assert run.trace.events[0].event_type == "agent_start"

    assert any(
        event.event_type == "error"
        for event in run.trace.events
    )

    assert run.trace.events[-1].event_type == "error"


def test_adapter_handles_invalid_report(
    tmp_path: Path,
) -> None:
    """Malformed JSON report should produce an adapter error."""

    script_path = tmp_path / "invalid_report.py"

    script_path.write_text(
        """
from pathlib import Path
import sys

args = sys.argv[1:]
index = args.index("--json-report")

Path(args[index + 1]).write_text(
    "{invalid-json",
    encoding="utf-8",
)

raise SystemExit(0)
""".strip(),
        encoding="utf-8",
    )

    executable_path = tmp_path / "invalid_report.cmd"

    executable_path.write_text(
        (
            "@echo off\n"
            f'"{sys.executable}" '
            f'"{script_path}" %*\n'
        ),
        encoding="utf-8",
    )

    adapter = AITestAgentAdapter(
        executable=str(executable_path),
    )

    run = adapter.run(make_task())

    assert run.status == "error"
    assert run.error is not None

    assert run.trace.events
    assert run.trace.events[0].event_type == "agent_start"

    assert any(
        event.event_type == "error"
        for event in run.trace.events
    )

    assert run.trace.events[-1].event_type == "error"


def test_adapter_rejects_invalid_report_structure(
    tmp_path: Path,
) -> None:
    """A JSON report with invalid top-level structure should fail."""

    report = {
        "spec": "login.md",
        "test_cases": [],
        "generated_tests": {},
        "executions": [],
    }

    executable = write_fake_executable(
        tmp_path,
        report=report,
    )

    adapter = AITestAgentAdapter(
        executable=str(executable),
    )

    run = adapter.run(make_task())

    assert run.status == "error"
    assert run.error is not None
    assert "generated_tests" in run.error

    assert run.trace.events
    assert run.trace.events[0].event_type == "agent_start"

    assert any(
        event.event_type == "error"
        for event in run.trace.events
    )

    assert run.trace.events[-1].event_type == "error"


def test_adapter_preserves_stderr(
    tmp_path: Path,
) -> None:
    """Adapter should preserve subprocess stderr output."""

    script_path = tmp_path / "stderr_ai_test_agent.py"

    report = make_report()

    script_path.write_text(
        f"""
from pathlib import Path
import json
import sys

args = sys.argv[1:]
index = args.index("--json-report")
report_path = Path(args[index + 1])

print("target stderr", file=sys.stderr)

report = {report!r}

report_path.write_text(
    json.dumps(report),
    encoding="utf-8",
)

raise SystemExit(0)
""".strip(),
        encoding="utf-8",
    )

    executable_path = tmp_path / "stderr_ai_test_agent.cmd"

    executable_path.write_text(
        (
            "@echo off\n"
            f'"{sys.executable}" '
            f'"{script_path}" %*\n'
        ),
        encoding="utf-8",
    )

    adapter = AITestAgentAdapter(
        executable=str(executable_path),
    )

    run = adapter.run(make_task())

    assert run.status == "success"
    assert run.error is None
    assert run.output is not None

    assert "target stderr" in run.output["stderr"]

    assert run.output["exit_code"] == 0

    assert run.trace.events
    assert run.trace.events[0].event_type == "agent_start"
    assert run.trace.events[-1].event_type == "agent_end"


def test_adapter_uses_custom_working_directory(
    tmp_path: Path,
) -> None:
    """Adapter should pass a custom working directory to subprocess."""

    report = make_report()

    executable = write_fake_executable(
        tmp_path,
        report=report,
    )

    working_directory = tmp_path / "working"
    working_directory.mkdir()

    adapter = AITestAgentAdapter(
        executable=str(executable),
        cwd=working_directory,
    )

    run = adapter.run(make_task())

    assert run.status == "success"
    assert run.error is None


def test_adapter_uses_custom_timeout(
    tmp_path: Path,
) -> None:
    """Adapter should honor its configured timeout."""

    script_path = tmp_path / "slow_agent.py"

    script_path.write_text(
        """
import time

time.sleep(1)
""".strip(),
        encoding="utf-8",
    )

    executable_path = tmp_path / "slow_agent.cmd"

    executable_path.write_text(
        (
            "@echo off\n"
            f'"{sys.executable}" '
            f'"{script_path}" %*\n'
        ),
        encoding="utf-8",
    )

    adapter = AITestAgentAdapter(
        executable=str(executable_path),
        timeout=0.1,
    )

    run = adapter.run(make_task())

    assert run.status == "error"
    assert run.error is not None
    assert "timed out" in run.error.lower()


def test_adapter_rejects_nonexistent_executable(
    tmp_path: Path,
) -> None:
    """Adapter should convert missing executable errors into Run errors."""

    adapter = AITestAgentAdapter(
        executable=str(tmp_path / "does-not-exist.cmd"),
    )

    run = adapter.run(make_task())

    assert run.status == "error"
    assert run.error is not None

    assert run.trace.events
    assert run.trace.events[0].event_type == "agent_start"
    assert run.trace.events[-1].event_type == "error"


def test_adapter_report_contains_generated_tests_and_executions(
    tmp_path: Path,
) -> None:
    """Adapter should expose both generated tests and executions."""

    report = make_report()

    report["generated_tests"].append(
        {
            "name": "test_login_invalid",
        }
    )

    report["executions"].append(
        {
            "trace_id": "trace-002",
            "test_name": "test_login_invalid",
            "status": "passed",
            "started_at": "2026-09-09T00:00:02",
            "finished_at": "2026-09-09T00:00:03",
            "duration_seconds": 1.0,
            "stdout": "",
            "stderr": "",
            "exit_code": 0,
            "http_status_code": 401,
        }
    )

    executable = write_fake_executable(
        tmp_path,
        report=report,
    )

    adapter = AITestAgentAdapter(
        executable=str(executable),
    )

    run = adapter.run(make_task())

    assert run.status == "success"

    stored_report = run.output["report"]

    assert len(stored_report["generated_tests"]) == 2
    assert len(stored_report["executions"]) == 2


def test_adapter_output_is_json_serializable(
    tmp_path: Path,
) -> None:
    """Adapter output should remain JSON serializable."""

    executable = write_fake_executable(
        tmp_path,
        report=make_report(),
    )

    adapter = AITestAgentAdapter(
        executable=str(executable),
    )

    run = adapter.run(make_task())

    json.dumps(run.output)

    assert run.status == "success"
