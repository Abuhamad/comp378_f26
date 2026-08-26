# Lesson 23 — Data Provenance and AI Supply Chain Security

Objectives:

- Enumerate AI supply chain attack vectors across the ML lifecycle.
- Describe what a DBOM records and explain how it enables provenance tracing.
- Audit a provided checkpoint using static opcode disassembly.
- Convert a PyTorch checkpoint to safetensors format and verify tensor value equivalence.
- Write a CI pipeline gate script that rejects any checkpoint file failing a static safety check.

## Main Concepts

- Data Bill of Materials (DBOM): provenance tracking for datasets and preprocessing steps
- AI supply chain risks: third-party model weights, untrusted dataset sources, dependency injection
- `pickle` deserialization exploits: arbitrary code execution via malicious model checkpoints
- `safetensors` format: safe serialization without arbitrary code execution

## Learning Outcomes

- Student can enumerate AI supply chain attack vectors from data ingestion through model deployment.
- Student can audit a model checkpoint file for deserialization exploit indicators using static analysis.
- Student can convert an insecure `pickle`-based checkpoint to `safetensors` format and verify equivalence.
