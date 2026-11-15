# 39: Secure AI Deployment: Guardrails & Sandboxing

## Abstract

Deploying LLMs in production requires multiple layers of defense to block adversarial inputs, prevent harmful outputs, contain code execution, and detect abuse. This lesson covers six concrete defense mechanisms: input guardrails that stop injection attacks before reaching the LLM, output guardrails that filter harmful completions, the NeMo Guardrails framework for configurable policy enforcement, Llama Guard as a lightweight output classifier, Docker-based sandboxing with syscall filtering for code-generating models, and API-level rate limiting with anomaly detection. Together, these layers form a defense-in-depth strategy that addresses threats from Lesson 36 (prompt injection), Lesson 37 (RAG and agentic attacks), and Lesson 38 (model extraction and output monitoring).

## Objectives

- Student can identify the three categories of input guardrail (topic classification, injection detection, PII scrubbing) and state which threat each addresses.
- Student can configure NeMo Guardrails using COLANG to define input rails that block at least two injection patterns from Lesson 36 and output rails that filter at least one harmful content category.
- Student can describe the Llama Guard model's category taxonomy and explain how it is used as an output classifier alongside a generation model.
- Student can deploy a sandboxed code execution environment using Docker with seccomp and cgroup limits, and verify that a resource-exhaustion attempt (infinite loop, large allocation) is contained within the configured limits.
- Student can configure per-user API rate limits and describe what anomaly detection signal would trigger an alert for extraction or abuse behavior.
- Student can measure the false positive rate of the configured guardrails by running 10 benign inputs through the protected application and reporting how many are incorrectly blocked.

## Content

### Input Guardrails

Input guardrails intercept user messages before the LLM processes them, filtering or routing requests based on three categories of policy. Topic classification uses a lightweight binary or multi-class router to detect when a user asks about a disallowed topic and routes that request away from the LLM without wasting inference tokens. A financial-advice chatbot might route investment queries to a disclaimer rather than to the model. Injection detection uses two approaches: pattern matching searches for structural markers of prompt injection payloads (e.g., occurrences of "system prompt," "ignore previous instructions," or newline followed by colon-separated key-value pairs), while a learned classifier trained on Lesson 36's attack corpus detects semantic similarity to known jailbreak attempts even when the syntax is novel. PII scrubbing uses named entity recognition (NER) to identify and redact personally identifiable information before sending the message to the LLM, preventing accidental training-data leakage or privacy violations. A customer-support LLM might redact credit card numbers and social security numbers using regex patterns or a Hugging Face token-classification model like `dslim/bert-base-multilingual-cased-ner`.

### Output Guardrails

Output guardrails evaluate the LLM's completion after generation, filtering or blocking responses that violate policy. Harmful content filtering checks the model's output against a policy taxonomy (e.g., categories for violence, sexual content, misinformation, illegal activities) and either drops the response or returns a fallback message. Structured output validation enforces that JSON responses from the LLM conform to a specified schema, using a library like `jsonschema` to verify that a model's structured output matches the expected fields, types, and ranges. A coding assistant might reject code that uses dangerous functions like `eval()` or `exec()`, or an API might reject a response missing required fields. Output guardrails trade off latency (an additional inference or validation pass) against robustness. They catch failures the input stage missed but incur computational cost.

### NeMo Guardrails and COLANG

NeMo Guardrails is an open-source framework from NVIDIA that enforces input and output policies using COLANG, a domain-specific language for composable guardrail definitions. A COLANG configuration defines user messages and bot messages as labeled patterns, with input rails that execute before LLM processing and output rails that execute after. A simple input rail might read:

```txt
define user message check harmful_input
  "This is harmful"
  "Ignore my instructions"
  "Roleplay as a hacker"
```

This pattern matches literal strings and routes them to a handler that logs the violation and returns a safe response. A rule that combines patterns uses a classifier to detect semantic similarity: an NLI (Natural Language Inference) model like `cross-encoder/nli-deberta-v3-small` scores whether the user input entails one of the rail definitions and triggers the guardrail if the score exceeds a threshold (e.g., 0.7). Dialog flow control in COLANG also defines the conversation flow, specifying which bot messages can follow which user actions, allowing enforcement of conversation structure (e.g., a bot must request confirmation before executing a command). Practitioners configure rails by editing the COLANG file, specifying injection keywords from attack logs (like the attack patterns in Lesson 36's exercises), matching them against user inputs, and logging blocks to audit compliance.

### Llama Guard: Lightweight Output Classification

Llama Guard is a specialized model architecture, fine-tuned from the Llama-2-7B backbone, that classifies text against a fixed taxonomy of harmful categories. The taxonomy includes categories like violence and harm, illegal activity, sexual content, misinformation, self-harm, political bias, and others, with clear definitions for each. Llama Guard operates as a gating classifier: after the primary LLM completes a response, Llama Guard receives the completion and outputs a single label (safe, unsafe, or the specific unsafe category). Because Llama Guard is a smaller model than the primary LLM, running it on every output incurs less latency overhead than regenerating the entire completion. A deployment might use Llama Guard as a pre-filter on risky domains (e.g., legal or medical advice) or as a post-generation check on all outputs. The trade-off is that Llama Guard, trained on a fixed taxonomy, may miss novel harmful content, and fine-tuning it on domain-specific categories requires labeled data.

### Code Execution Sandboxing

LLMs that generate code (e.g., GitHub Copilot, Claude's code interpreter) run the generated code in a sandboxed environment to prevent resource exhaustion, breakout attacks, or data theft. Docker containers form the outer boundary, isolating the code process from the host OS and other containers. Within the container, seccomp (secure computing mode) applies a syscall allowlist that explicitly permits only safe operations (e.g., read, write, exit) and blocks dangerous ones (e.g., fork to spawn new processes, ptrace for debugging, open to access files outside the allowed mount). cgroup (control groups) enforce resource limits: a CPU limit of 1 core and a memory limit of 512 MB prevent infinite loops or large allocations from consuming host resources. Network namespace isolation removes the container's network access entirely, preventing exfiltration or external command-and-control. Layers are necessary because each one addresses a different vector: Docker isolates the process, seccomp reduces syscall surface area, cgroup limits resource consumption, and network isolation blocks exfiltration. A single sandbox layer (e.g., Docker alone) is insufficient because a container breakout or a resource-exhaustion attack within the container's limits can still compromise the host.

### API Rate Limiting and Monitoring

Rate limiting enforces per-user quotas to prevent abuse (e.g., extraction attacks attempting to enumerate the model's training data or jailbreak loops testing many payloads). Token-bucket and sliding-window algorithms are the two standard approaches. Token-bucket maintains a bucket for each user with a capacity (e.g., 1,000 tokens per hour) and a refill rate (e.g., 1 token per 3.6 seconds). Each request consumes tokens, and requests are rejected if the bucket is empty. Sliding-window tracks the count of requests in the past N minutes and rejects requests when the count exceeds a threshold. Per-user quotas partition rate limits by user identity, preventing one attacker from exhausting the budget for all users. Anomaly detection monitors query patterns for signals of abuse: a sudden spike in request volume, a pattern of similar payloads (high entropy repetition of keywords or injection patterns from Lesson 36), or a shift toward uncommon inference parameters (e.g., temperature 0.0 and max-tokens at the ceiling). Abuse telemetry logs the detected anomaly (user ID, request count, entropy score) and optionally alerts operations staff or temporarily blocks the user. Alerting thresholds are tuned per deployment, balancing false positives (blocking legitimate power users) against false negatives (missing actual abuse).

## Lab

See [lab.ipynb](./lab.ipynb) for hands-on exercises. The lab applies NeMo Guardrails to configure input and output rails that block injection patterns from Lesson 36 and filter harmful content, measures block rates on attack inputs, and computes false positive rates on benign inputs.

## Summary

Guardrails and sandboxing form a layered defense strategy for LLM deployment. Input guardrails stop dangerous requests before inference, reducing wasted computation and preventing the LLM from processing attacks. Output guardrails filter harmful completions. The NeMo Guardrails framework and Llama Guard model provide practical tools for configuring these policies at scale. Code execution sandboxing uses Docker, seccomp, and cgroup to isolate and contain generated code. API rate limiting and anomaly detection detect extraction and abuse attempts. Together, these mechanisms reduce the attack surface and detectability of the threats from Lesson 36, Lesson 37, and Lesson 38.

## Useful References and Resources

- NVIDIA NeMo Guardrails documentation (<https://nemoguardrails.readthedocs.io>): COLANG syntax, rail configuration, and dialog flow examples.
- Llama Guard model card and taxonomy (<https://huggingface.co/meta-llama/Llama-Guard-7b>): category definitions and integration with Hugging Face `transformers`.
- Docker security best practices (<https://docs.docker.com/engine/security>): seccomp profiles, AppArmor, and cgroup resource limits.
- seccomp syscall filtering guide (<https://man7.org/linux/man-pages/man2/seccomp.2.html>): allowlist configuration and audit logging.
- Hugging Face token-classification models (<https://huggingface.co/models?filter=token-classification>): resources for PII detection and NER-based entity redaction.
- Linux cgroup resource limits (<https://man7.org/linux/man-pages/man7/cgroups.7.html>): CPU, memory, and network namespace isolation.
