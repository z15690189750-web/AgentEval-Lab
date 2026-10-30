from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from agent_eval_lab.adapters.ai_test_agent import (
    AITestAgentAdapter,
)
from agent_eval_lab.core.models import Task
from agent_eval_lab.run import Runner


def resolve_ai_test_agent_root() -> Path:
    """Locate AI Test Agent without hardcoding an absolute path."""

    configured = os.getenv("AI_TEST_AGENT_ROOT")

    if configured:
        return Path(configured)

    workspace_root = Path(__file__).resolve().parents[2]

    return workspace_root / "AI-Test-Agent1"


def resolve_executable() -> str:
    """Prefer the console script next to the running interpreter."""

    candidate = Path(sys.executable).with_name(
        "ai-test-agent.exe"
        if os.name == "nt"
        else "ai-test-agent"
    )

    if candidate.exists():
        return str(candidate)

    return "ai-test-agent"


AI_TEST_AGENT_ROOT = resolve_ai_test_agent_root()

LOGIN_SPEC = (
    AI_TEST_AGENT_ROOT
    / "examples"
    / "login.md"
)

LOCAL_API_MODULE = (
    AI_TEST_AGENT_ROOT
    / "tests"
    / "support"
    / "local_api.py"
)


def load_local_api_server():
    """Load the AI Test Agent's existing local API fixture."""

    if not LOCAL_API_MODULE.exists():
        raise FileNotFoundError(
            f"AI Test Agent local API fixture not found: "
            f"{LOCAL_API_MODULE}"
        )

    spec = importlib.util.spec_from_file_location(
        "ai_test_agent_local_api",
        LOCAL_API_MODULE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            "Unable to load AI Test Agent local API fixture."
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.LocalAPIServer


def test_ai_test_agent_real_e2e() -> None:
    """
    Verify the real integration boundary.

    AI Test Agent is invoked through its public CLI.
    Adapter returns raw target-agent output.
    Runner creates the AgentEval Lab Run.
    """

    LocalAPIServer = load_local_api_server()

    server = LocalAPIServer()
    server.start()

    try:
        task = Task(
            id="ai-test-agent-login-e2e",
            name="AI Test Agent login API E2E",
            description=(
                "Evaluate the AI Test Agent's ability to "
                "generate and execute API tests from the "
                "login API specification."
            ),
            input={
                "spec": str(LOGIN_SPEC),
                "base_url": server.base_url,
                "mock": True,
            },
        )

        adapter = AITestAgentAdapter(
            executable=resolve_executable(),
            cwd=AI_TEST_AGENT_ROOT,
            timeout=60.0,
        )

        runner = Runner(adapter)
        run = runner.run(
            task,
            run_id="ai-test-agent-login-e2e-run",
        )

    finally:
        server.stop()

    assert run.status == "success"
    assert run.error is None

    assert isinstance(run.output, dict)

    assert "report" in run.output
    assert "stdout" in run.output
    assert "stderr" in run.output
    assert "exit_code" in run.output

    assert run.output["exit_code"] == 0

    report = run.output["report"]

    assert isinstance(report, dict)

    assert "generated_tests" in report
    assert "executions" in report

    assert isinstance(
        report["generated_tests"],
        list,
    )

    assert isinstance(
        report["executions"],
        list,
    )

    assert len(report["generated_tests"]) > 0
    assert len(report["executions"]) > 0

    assert run.trace.events

    event_types = [
        event.event_type
        for event in run.trace.events
    ]

    assert event_types[0] == "agent_start"
    assert event_types[-1] == "agent_end"