# AgentEval Lab

AgentEval Lab is a lightweight evaluation and regression framework for AI agents.

The project focuses on a practical question:

> How do we know whether an AI agent actually works, why it fails, and whether a new version is better or worse than the previous version?

## Core Idea

AgentEval Lab evaluates AI agents through reproducible tasks, execution traces, deterministic evaluators, benchmarks, repeated runs, and regression comparison.

The project is designed to work with real agents rather than only toy examples.

## Target Agent

The primary target agent is the AI Test Agent developed as the first portfolio project.

The relationship between the two projects is:

```text
AI Test Agent
     ↓
execute testing tasks
     ↓
produce results and traces
     ↓
AgentEval Lab
     ↓
evaluate
     ↓
benchmark
     ↓
compare versions
     ↓
detect regression