# AgentEval Regression Report

**Overall:** `REGRESSION`

## Experiments

| | ID | Name | Benchmark |
|---|---|---|---|
| Baseline | `e2e-008` | ai-test-agent-login-e2e-008 | ai-test-agent-login |
| Current | `e2e-007` | ai-test-agent-login-e2e-007 | ai-test-agent-login |

## Metric Comparison

| Metric | Baseline | Current | Delta | Threshold | Direction | Status |
|---|---:|---:|---:|---:|---|---|
| task_success_rate | 1 | 0 | -1 | 0 | higher is better | REGRESSED |
| average_score | 1 | 0.909091 | -0.090909 | 0 | higher is better | REGRESSED |
| failure_rate | 0 | 0 | 0 | 0 | lower is better | PASS |
| average_latency_ms | 5301.185 | 5656.686333 | +355.501333 | 0 | lower is better | REGRESSED |

## Summary

- Regression detected: `True`
- Regressed metrics: `task_success_rate`, `average_score`, `average_latency_ms`
- Improved metrics: None
