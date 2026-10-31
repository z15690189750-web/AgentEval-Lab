# AgentEval Regression Report

**Overall:** `REGRESSION`

## Experiments

| | ID | Name | Benchmark |
|---|---|---|---|
| Baseline | `final-exp-001` | AI Test Agent Final Evaluation | ai-test-agent-login |
| Current | `e2e-008` | ai-test-agent-login-e2e-008 | ai-test-agent-login |

## Metric Comparison

| Metric | Baseline | Current | Delta | Threshold | Direction | Status |
|---|---:|---:|---:|---:|---|---|
| task_success_rate | 0 | 1 | +1 | 0 | higher is better | PASS |
| average_score | 0.666667 | 1 | +0.333333 | 0 | higher is better | PASS |
| failure_rate | 0 | 0 | 0 | 0 | lower is better | PASS |
| average_latency_ms | 4638.323667 | 5301.185 | +662.861333 | 0 | lower is better | REGRESSED |

## Summary

- Regression detected: `True`
- Regressed metrics: `average_latency_ms`
- Improved metrics: `task_success_rate`, `average_score`
