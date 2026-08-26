# Lesson 16 — Empirical vs. Certified Robustness Fundamentals

Objectives:

- Define empirical robustness and explain why evaluation bias arises with weak attacks.
- Identify the three gradient masking subtypes.
- Detect gradient masking using random search and BPDA.
- Articulate how BPDA bypasses non-differentiable preprocessing.
- Describe a historical falsely secure defense and the attack breaking it.
- Articulate why empirical robustness cannot replace certified robustness.

## Main Concepts

- Limitations of empirical robustness: only as strong as the strongest attack tested
- Gradient masking and obfuscation: false security signal, detection via BPDA and random search
- Falsely secure defenses: case studies from post-Madry literature (thermometer encoding, etc.)
- Motivation for certified robustness: formal guarantees independent of attack strategy

## Learning Outcomes

- Student can define gradient masking and obfuscation and state the indicators that a defense is masking.
- Student can detect gradient masking in a published defense using optimization-free (random search) and BPDA attacks.
- Student can articulate why empirical robustness cannot replace certified robustness.
