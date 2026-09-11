
from __future__ import annotations

import importlib.util
from pathlib import Path

from agent_eval_lab.adapters.ai_test_agent import AITestAgentAdapter
from agent_eval_lab.core.models import Task


AI_TEST_AGENT_ROOT = Path(
    r"E:\code\1\AI-Agent-Portfolio\AI-Test-Agent1"
)

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
    """
    Load the LocalAPIServer directly from the AI Test Agent
    project's existing test fixture.

    The fixture is intentionally loaded by file path instead
    of copying it into AgentEval Lab.
    """

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
    Run the real AI Test Agent against the real local API.

    This test verifies the integration boundary between
    AgentEval Lab and the independent AI Test Agent project.

    The AI Test Agent is executed through its public CLI.
    No fake executable, fake report, or mocked subprocess is used.
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
            executable="ai-test-agent",
            cwd=AI_TEST_AGENT_ROOT,
            timeout=60.0,
        )

        run = adapter.run(task)

    finally:
        server.stop()

    assert run.status == "success"
    assert run.output is not None
    assert isinstance(run.output, dict)

    assert "report" in run.output
    assert "stdout" in run.output
    assert "stderr" in run.output
    assert "exit_code" in run.output

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

    event_names = [
        event.name
        for event in run.trace.events
    ]

    assert "ai_test_agent_started" in event_names
    assert "ai_test_agent_finished" in event_names

