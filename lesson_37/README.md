# Lesson 37 — RAG and AI Agent Security

Objectives:

- Enumerate the attack surfaces introduced by each RAG pipeline component.
- Describe a vector database poisoning attack.
- Exploit an indirect prompt injection in a local RAG pipeline.
- Document the full exploit chain step-by-step.
- Propose a retrieval-time mitigation strategy.
- Define the Principle of Least Authority (POLA) for agentic systems.

## Main Concepts

- Retrieval-Augmented Generation (RAG) architecture: retriever, vector store, generator pipeline
- Vector database poisoning: injecting adversarial documents to hijack retrieved context
- Tool and plugin misuse in agentic systems: unauthorized API calls via injected instructions
- Agentic privilege escalation: agent with elevated tool access manipulated by malicious context

## Learning Outcomes

- Student can enumerate the unique attack surfaces introduced by RAG pipelines beyond standard LLM attacks.
- Student can exploit an indirect prompt injection in a RAG pipeline to trigger an unauthorized external API call.
- Student can describe a vector database poisoning attack and propose a retrieval-time mitigation strategy.
