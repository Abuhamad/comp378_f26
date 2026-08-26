# Lesson 36 — LLM Security: Prompt Injection and Jailbreaking

Objectives:

- Map the LLM trust boundary architecture.
- Demonstrate a direct prompt injection attack that overrides a system prompt.
- Demonstrate an indirect prompt injection attack.
- Explain DAN-style jailbreaking.
- Describe the GCG attack.
- Implement a defensive prompt wrapper and compute its bypass rate.

## Main Concepts

- Direct prompt injection: user input overrides system prompt instructions
- Indirect prompt injection: malicious instructions embedded in external content retrieved by the LLM
- Jailbreak mechanics: DAN-style role-play, gradient-based adversarial suffix generation (GCG)
- System prompt extraction and context leakage

## Learning Outcomes

- Student can demonstrate a direct prompt injection that overrides a system prompt and exfiltrates instructions.
- Student can demonstrate an indirect prompt injection embedded in a retrieved document in a simulated RAG context.
- Student can implement and evaluate a defensive prompt wrapper and measure its bypass rate against a defined jailbreak set.
