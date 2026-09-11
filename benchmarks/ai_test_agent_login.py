
from __future__ import annotations

from pathlib import Path

from agent_eval_lab.benchmark import Benchmark
from agent_eval_lab.core.models import Task


AI_TEST_AGENT_ROOT = Path(
    r"E:\code\1\AI-Agent-Portfolio\AI-Test-Agent1"
)

LOGIN_SPEC = (
    AI_TEST_AGENT_ROOT
    / "examples"
    / "login.md"
)


def create_ai_test_agent_login_benchmark(
    base_url: str,
) -> Benchmark:
    """
    Create the real AI Test Agent login benchmark.

    The benchmark uses the existing login specification from
    the independent AI Test Agent project.

    The target API URL is supplied when the benchmark is created
    because the integration test starts the local API server
    dynamically and therefore its port is not known in advance.

    The benchmark defines the evaluation task. It does not
    execute the target agent.
    """

    if not base_url.strip():
        raise ValueError(
            "base_url must not be empty."
        )

    if not LOGIN_SPEC.exists():
        raise FileNotFoundError(
            f"AI Test Agent login specification not found: "
            f"{LOGIN_SPEC}"
        )

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
            "base_url": base_url,
            "mock": True,
        },
        expected_schema={
            "required": [
                "report",
                "stdout",
                "stderr",
                "exit_code",
            ],
            "types": {
                "report": "dict",
                "stdout": "str",
                "stderr": "str",
                "exit_code": "int",
            },
        },
    )

    return Benchmark(
        name="ai-test-agent-login",
        tasks=(task,),
    )

