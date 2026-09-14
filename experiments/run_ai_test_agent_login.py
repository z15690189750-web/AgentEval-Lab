from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from time import sleep
from urllib.parse import parse_qs, urlparse

from agent_eval_lab.adapters.ai_test_agent import AITestAgentAdapter
from agent_eval_lab.benchmark import Benchmark
from agent_eval_lab.core.models import Task
from agent_eval_lab.evaluate import (
    evaluate_execution_outcome,
    evaluate_generation_completeness,
    evaluate_schema,
)
from agent_eval_lab.evaluation_pipeline import EvaluationPipeline
from agent_eval_lab.experiment import Experiment
from agent_eval_lab.repeated_runner import RepeatedRunner
from agent_eval_lab.run import Runner


AI_TEST_AGENT_ROOT = Path(
    r"E:\code\1\AI-Agent-Portfolio\AI-Test-Agent1"
)

LOGIN_SPEC = AI_TEST_AGENT_ROOT / "examples" / "login.md"


class LocalAPIHandler(BaseHTTPRequestHandler):
    """Minimal local API used by the real AI Test Agent E2E experiment."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._send_response(200, "ok")
            return

        if parsed.path == "/users":
            self._send_response(200, "users")
            return

        if parsed.path == "/server-error":
            self._send_response(500, "internal server error")
            return

        if parsed.path == "/slow":
            sleep(11)
            self._send_response(200, "slow response")
            return

        self._send_response(404, "not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path != "/login":
            self._send_response(404, "not found")
            return

        content_length = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        body = self.rfile.read(
            content_length
        ).decode("utf-8")

        parameters = parse_qs(body)

        username = parameters.get("username")
        password = parameters.get("password")

        if not username or not password:
            self._send_response(
                400,
                "missing credentials",
            )
            return

        self._send_response(
            200,
            "login success",
        )

    def log_message(
        self,
        format: str,
        *args,
    ) -> None:
        """Keep experiment output focused on evaluation results."""

    def _send_response(
        self,
        status_code: int,
        body: str,
    ) -> None:
        encoded_body = body.encode("utf-8")

        try:
            self.send_response(status_code)

            self.send_header(
                "Content-Type",
                "text/plain; charset=utf-8",
            )

            self.send_header(
                "Content-Length",
                str(len(encoded_body)),
            )

            self.end_headers()

            self.wfile.write(encoded_body)

        except (
            BrokenPipeError,
            ConnectionResetError,
        ):
            pass


class LocalAPIServer:
    """Start and stop an ephemeral local HTTP server for the experiment."""

    def __init__(self) -> None:
        self.server = HTTPServer(
            ("127.0.0.1", 0),
            LocalAPIHandler,
        )

        self.thread = Thread(
            target=self.server.serve_forever,
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        """Return the dynamically assigned server base URL."""

        host, port = self.server.server_address

        return f"http://{host}:{port}"

    def start(self) -> None:
        """Start serving requests in a background thread."""

        self.thread.start()

    def stop(self) -> None:
        """Stop the server and wait briefly for the thread to exit."""

        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def __enter__(self) -> LocalAPIServer:
        self.start()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.stop()


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Run a real AgentEval Lab evaluation experiment "
            "against the AI Test Agent."
        )
    )

    parser.add_argument(
        "--id",
        required=True,
        help="Experiment ID.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name.",
    )

    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of repetitions per benchmark task.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the JSON experiment result.",
    )

    parser.add_argument(
        "--markdown",
        type=Path,
        help="Optional path to the Markdown experiment report.",
    )

    return parser


def create_benchmark(base_url: str) -> Benchmark:
    """Create the real AI Test Agent login benchmark."""

    if not base_url.strip():
        raise ValueError("base_url must not be empty.")

    if not LOGIN_SPEC.exists():
        raise FileNotFoundError(
            f"AI Test Agent login specification not found: {LOGIN_SPEC}"
        )

    task = Task(
        id="ai-test-agent-login-e2e",
        name="AI Test Agent login API E2E",
        description=(
            "Evaluate the AI Test Agent's ability to generate and execute "
            "API tests from the login API specification."
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


def write_markdown(
    path: Path,
    result,
) -> None:
    """Write a human-readable experiment report."""

    markdown = [
        "# AgentEval Experiment Report",
        "",
        f"**Experiment:** `{result.name}`",
        f"**ID:** `{result.id}`",
        f"**Benchmark:** `{result.benchmark_name}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]

    for name, value in result.metrics.items():
        markdown.append(
            f"| {name} | {value} |"
        )

    markdown.extend(
        [
            "",
            "## Evaluations",
            "",
            "| Run | Evaluator | Passed | Score | Reason |",
            "|---|---|---|---:|---|",
        ]
    )

    for evaluation in result.evaluations:
        reason = evaluation.reason.replace(
            "|",
            "\\|",
        )

        markdown.append(
            f"| `{evaluation.run_id}` "
            f"| `{evaluation.evaluator}` "
            f"| {evaluation.passed} "
            f"| {evaluation.score} "
            f"| {reason} |"
        )

    path.write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )


def run_experiment(args: argparse.Namespace) -> None:
    """Run the complete real evaluation experiment."""

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.markdown is not None:
        args.markdown.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    with LocalAPIServer() as server:
        benchmark = create_benchmark(
            server.base_url
        )

        adapter = AITestAgentAdapter(
            executable="ai-test-agent",
            cwd=AI_TEST_AGENT_ROOT,
            timeout=60.0,
        )

        runner = RepeatedRunner(
            runner=Runner(adapter),
        )

        evaluation_pipeline = EvaluationPipeline(
            evaluators=[
                evaluate_schema,
                evaluate_execution_outcome,
                evaluate_generation_completeness,
            ],
        )

        experiment = Experiment(
            repeated_runner=runner,
            evaluation_pipeline=evaluation_pipeline,
        )

        result = experiment.run(
            benchmark=benchmark,
            repetitions=args.repetitions,
            experiment_id=args.id,
            experiment_name=args.name,
        )

    args.output.write_text(
        result.model_dump_json(
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.markdown is not None:
        write_markdown(
            path=args.markdown,
            result=result,
        )

    print(
        result.model_dump_json(
            indent=2,
        )
    )


def main() -> None:
    """Run the experiment CLI."""

    parser = build_parser()
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error(
            "--repetitions must be at least 1."
        )

    run_experiment(args)


if __name__ == "__main__":
    main()