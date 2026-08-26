# Lesson 39 — Secure AI Deployment: Guardrails and Sandboxing

Objectives:

- Identify the three categories of input guardrail.
- Configure NeMo Guardrails using COLANG.
- Describe the Llama Guard model’s category taxonomy.
- Deploy a sandboxed execution environment using Docker with seccomp and cgroup limits.
- Configure per-user API rate limits.
- Measure the false positive rate of the configured guardrails.

## Main Concepts

- Input guardrails: topic classification, injection detection, PII scrubbing before LLM processing
- Output guardrails: harmful content filtering, policy compliance checking, structured output validation
- NeMo Guardrails and Llama Guard: architecture and configuration
- Code execution sandboxing: container isolation, resource limits, network restrictions for code-generating LLMs
- API rate limiting and monitoring: per-user quotas, anomaly detection, abuse telemetry

## Learning Outcomes

- Student can configure NeMo Guardrails or equivalent to block a defined set of disallowed topics and injection patterns.
- Student can deploy a sandboxed code execution environment for an LLM code assistant with defined resource limits.
- Student can demonstrate that the guardrails block known attacks from m36/m37 and measure false positive rate on benign inputs.
