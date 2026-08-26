# Lesson 12 — Adversarial Attacks Beyond Vision: NLP and Audio

Objectives:

- Articulate why gradient-based image attacks do not directly apply to NLP.
- Describe the TextFooler word importance ranking and semantic similarity constraint.
- Describe how BERT-Attack uses masked language model to generate adversarial substitutions.
- Explain the masking mechanism that makes audio adversarial perturbations imperceptible.
- Run TextAttack to generate word-substitution adversarial examples and report ASR.

## Main Concepts

- Discrete perturbation spaces: character-level and word-level substitutions
- NLP attack algorithms: TextFooler (word importance ranking + semantic similarity constraint) and BERT-Attack (masked-LM word replacement)
- Acoustic adversarial examples: psychoacoustic masking, CTC-loss-based audio perturbations
- Attack evaluation metrics for NLP: semantic similarity, grammar preservation, human imperceptibility

## Learning Outcomes

- Student can articulate why gradient-based image attacks do not directly transfer to NLP and what makes text perturbation discrete.
- Student can run `TextAttack` to generate word-substitution adversarial examples against a sentiment classifier.
- Student can report attack success rate, average word substitution rate, and USE semantic similarity for generated NLP adversarial examples.
