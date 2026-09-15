# AgentEval Lab

> A lightweight Agent Evaluation, Benchmark, and Regression Framework for testing the reliability of AI Agents.

AgentEval Lab is a Python-based evaluation framework designed to answer a practical question:

> **An AI Agent can run successfully — but how do we know whether its behavior is actually reliable?**

Instead of evaluating an Agent only by its final output, AgentEval Lab introduces a structured evaluation pipeline based on **Tasks, Runs, Traces, Evaluators, Metrics, Benchmarks, and Regression Analysis**.

The project also integrates **AI Test Agent** as a real Target Agent, forming an end-to-end engineering loop from Agent execution to measurable quality evaluation.

---

## Why AgentEval Lab?

A working Agent is not necessarily a reliable Agent.

For an Agent-based system, evaluation may need to consider:

* Did the Agent complete the task?
* Did the output satisfy the expected structure?
* Did it follow the required execution behavior?
* Did the execution actually succeed?
* Did the Agent regress compared with a previous baseline?
* Can the result be reproduced and traced back to raw execution evidence?

AgentEval Lab provides a lightweight framework for answering these questions without introducing unnecessary enterprise infrastructure.

---

## Core Architecture

```text
                         Target Agent
                              │
                              ▼
                           Adapter
                              │
                              ▼
                             Task
                              │
                              ▼
                            Runner
                              │
                              ▼
                         Run / Trace
                              │
                              ▼
                    Evaluator Pipeline
                              │
              ┌───────────────┼────────────────┐
              │               │                │
              ▼               ▼                ▼
           Output           Schema           Trace
          Evaluator        Evaluator        Evaluator
              │               │                │
              └───────────────┼────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
            Execution Outcome   Generation Completeness
               Evaluator             Evaluator
                    │                   │
                    └─────────┬─────────┘
                              ▼
                     Evaluation Results
                              │
                              ▼
                           Metrics
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
             Benchmark              Repeated Runs
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    Regression Comparator
                              │
                              ▼
                       JSON / Markdown
                           Reports
```

---

## Key Features

### 1. Adapter-Based Target Agent Integration

Agents are integrated through an adapter abstraction rather than coupling the evaluation framework to one specific implementation.

Current integration:

```text
AI Test Agent
      │
      ▼
AITestAgentAdapter
      │
      ▼
AgentEval Lab
```

The adapter preserves relevant execution evidence including:

* raw reports
* stdout
* stderr
* exit code
* configuration
* observed execution information

The framework avoids assigning semantic meanings that cannot be supported by execution evidence.

---

### 2. Structured Task Model

Tasks can define:

* input
* expected output
* expected schema
* required tools
* forbidden tools
* metadata

This provides a consistent contract between the evaluation framework and the Target Agent.

---

### 3. Execution Runs and Traces

The framework models Agent execution explicitly through `Run` and `Trace`.

Supported Run states:

```text
running
success
failure
error
```

Supported trace event types:

```text
agent_start
agent_end
tool_call
tool_result
error
message
```

This allows evaluation to use execution evidence rather than relying only on final output.

---

### 4. Deterministic Evaluators

The current implementation includes five deterministic evaluator types:

```text
OutputEvaluator
SchemaEvaluator
TraceEvaluator
ExecutionOutcomeEvaluator
GenerationCompletenessEvaluator
```

They cover different dimensions of Agent behavior:

| Evaluator               | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| Output                  | Validate expected output behavior                    |
| Schema                  | Validate structured output                           |
| Trace                   | Validate execution lifecycle and behavioral evidence |
| Execution Outcome       | Validate actual execution results                    |
| Generation Completeness | Validate expected generation completion              |

If Agent execution itself fails, evaluators explicitly handle the failed execution state rather than incorrectly treating downstream checks as successful.

---

### 5. Metrics

The framework currently calculates:

```text
task_success_rate
average_score
failure_rate
average_latency_ms
```

`task_success_rate` is calculated at the Run level: a Run is considered successful only when its required evaluation results pass.

This prevents a single successful sub-check from incorrectly turning an unsuccessful Agent run into a successful task.

---

### 6. Repeated Evaluation

`RepeatedRunner` supports executing the same Task multiple times with independent Run IDs.

This enables experiments that move beyond a single execution and provide a basis for:

* repeated evaluation
* benchmark experiments
* metric aggregation
* regression analysis

---

### 7. Benchmarking

Benchmark experiments provide a fixed set of Tasks and evaluation rules that can be executed repeatedly.

The goal is not to produce an arbitrary score, but to create a reproducible basis for comparing Agent behavior.

---

### 8. Regression Detection

The framework includes:

```text
MetricConfig
RegressionResult
RegressionSummary
RegressionComparator
```

Regression comparison supports:

* baseline vs current experiment
* direction-aware metrics
* threshold-based comparison
* regression detection
* improvement detection

Example:

```text
Baseline
    │
    ▼
Current Experiment
    │
    ▼
Metric Comparison
    │
    ├── Improved
    ├── Unchanged
    └── Regression
```

---

### 9. Evidence-Preserving Reports

The framework supports:

```text
JSON
Markdown
```

reports containing evaluation and experiment information.

Raw execution evidence can be retained alongside evaluation results so that a metric can be traced back to the underlying Agent execution.

---

# AI Test Agent Integration

AgentEval Lab uses **AI Test Agent** as a real Target Agent.

This creates a relationship between the two portfolio projects:

```text
┌─────────────────────────────────────┐
│           AI Test Agent             │
│                                     │
│  Spec → Generate → Execute          │
│              → Analyze → Self-Heal  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          AgentEval Lab              │
│                                     │
│  Evaluate → Measure → Benchmark     │
│              → Regression           │
└─────────────────────────────────────┘
```

In other words:

**AI Test Agent answers:**

> How can an Agent perform API testing?

**AgentEval Lab answers:**

> How can we evaluate whether that Agent is reliable?

---

# Experimental Evidence

The framework has been validated against a real AI Test Agent execution workflow.

One experiment produced:

```text
Generated tests: 11
Passed:           0 / 11
```

The resulting metrics were:

```text
task_success_rate:    0.0
average_score:        0.6666666666666666
failure_rate:        0.0
average_latency_ms:  4638.323666666666
```

Compared with the baseline:

```text
                         Baseline       Current
task_success_rate          1.0            0.0
average_score              1.0         0.666667
failure_rate               0.0            0.0
average_latency_ms      4636.058      4638.324
```

Observed changes:

```text
task_success_rate:   -1
average_score:       -0.333333
failure_rate:         0
latency:             +2.266 ms
```

The framework correctly classified the experiment as:

```text
REGRESSION
```

The important result is not that the Target Agent produced a poor run.

The important result is that the evaluation framework **detected, quantified, and reported the regression from execution evidence**.

---

# Engineering Validation

The complete automated test suite currently passes:

```text
119 passed in 17.14s
```

The project intentionally keeps the architecture small.

During development, an unused `benchmark_runner.py` implementation and its tests were removed after verifying that no production code depended on them.

The full test suite remained:

```text
119 passed
```

This reflects the project's design goal:

> **Small but complete — not enterprise-scale infrastructure for its own sake.**

---

# Project Structure

```text
AgentEval-Lab/
│
├── agent_eval_lab/
│   ├── adapters/
│   │   ├── ai_test_agent.py
│   │   └── __init__.py
│   │
│   ├── core/
│   │   ├── models.py
│   │   └── __init__.py
│   │
│   ├── benchmark.py
│   ├── cli.py
│   ├── evaluate.py
│   ├── evaluation_pipeline.py
│   ├── experiment.py
│   ├── experiment_io.py
│   ├── metrics.py
│   ├── regression.py
│   ├── repeated_runner.py
│   ├── report.py
│   └── run.py
│
├── benchmarks/
│   └── ai_test_agent_login.py
│
├── experiments/
│   └── run_ai_test_agent_login.py
│
├── tests/
│   ├── test_ai_test_agent_adapter.py
│   ├── test_ai_test_agent_evaluators.py
│   ├── test_ai_test_agent_integration.py
│   ├── test_benchmark.py
│   ├── test_cli.py
│   ├── test_evaluate.py
│   ├── test_evaluation_pipeline.py
│   ├── test_experiment.py
│   ├── test_experiment_io.py
│   ├── test_experiment_result.py
│   ├── test_metrics.py
│   ├── test_models.py
│   ├── test_regression.py
│   ├── test_repeated_runner.py
│   ├── test_report.py
│   ├── test_run.py
│   └── test_trace_evaluator.py
│
├── README.md
├── PROJECT_FACTS.md
├── pyproject.toml
└── .gitignore
```

---

# Tech Stack

* Python
* Pydantic
* pytest
* subprocess
* JSON
* Markdown
* CLI
* Git

The project deliberately avoids unnecessary infrastructure such as:

* Kubernetes
* Kafka
* Elasticsearch
* Redis Cluster
* microservices
* multi-tenancy
* large frontend applications
* enterprise SaaS infrastructure

The goal is to demonstrate **Agent evaluation engineering**, not infrastructure complexity.

---

# Quick Start

## 1. Clone

```bash
git clone https://github.com/z15690189750-web/AgentEval-Lab.git
cd AgentEval-Lab
```

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install the project

```bash
pip install -e .
```

## 4. Run the test suite

```bash
pytest -q
```

Expected result for the current repository state:

```text
119 passed
```

---

# Design Principles

## Deterministic First

Evaluation logic should be deterministic whenever deterministic verification is sufficient.

This improves:

* reproducibility
* debuggability
* regression detection
* interview explainability

---

## Evidence Before Interpretation

Evaluation results should be grounded in actual execution evidence.

The framework therefore preserves:

```text
Run
Trace
stdout
stderr
exit code
raw reports
evaluation results
```

rather than reducing an Agent execution to a single opaque score.

---

## Small but Complete

The project intentionally avoids unnecessary abstractions and infrastructure.

The target is:

> **A complete, testable, explainable evaluation framework — not a miniature enterprise platform.**

---

## Fail Explicitly

Agent execution failures should not silently become successful evaluation results.

The evaluation pipeline explicitly handles:

```text
execution error
evaluation failure
metric degradation
regression
```

---

# Current Scope

Implemented:

* Target Agent adapters
* Task definitions
* Agent execution
* Run lifecycle
* Trace collection
* deterministic evaluators
* metric calculation
* repeated runs
* benchmark experiments
* regression comparison
* JSON reports
* Markdown reports
* AI Test Agent integration

Not implemented:

* LLM-as-a-Judge
* distributed evaluation infrastructure
* enterprise SaaS
* multi-tenant architecture
* Kubernetes-based execution
* large-scale distributed storage

---

# Project Status

**Feature-complete and portfolio-frozen.**

Current focus is documentation, reproducibility, resume presentation, and interview preparation rather than adding unnecessary infrastructure.

---

# Related Project

### AI Test Agent

AI-assisted API testing agent providing:

```text
API Spec
   ↓
Test Planning
   ↓
Test Generation
   ↓
Pytest Execution
   ↓
Failure Analysis
   ↓
Guarded Self-Healing
```

Repository:

https://github.com/z15690189750-web/AI-Test-Agent

---

# Author

Personal AI Agent Engineering Portfolio

Focus:

```text
Python
AI Agent Engineering
LLM Applications
Automated Testing
Agent Evaluation
```
