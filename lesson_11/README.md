# Lesson 11 — Adversarial Transferability and Substitute Models

Objectives:

- Explain why adversarial examples transfer across model architectures.
- Train a substitute model by querying a black-box target API.
- Generate transferable adversarial examples using PGD applied to the trained substitute.
- Measure the cross-architecture transfer fooling rate between two distinct model families.
- Articulate how MI-FGSM and input diversity improve transfer success rate over vanilla FGSM.

## Main Concepts

- Transferability phenomenon: why adversarial examples crafted on one model fool others
- Architectural and decision-boundary explanations of transferability
- Substitute model training: query-target model to collect synthetic labels; train surrogate
- Transferability boosting: input diversity, translation invariance, momentum iterative FGSM (MI-FGSM)

## Learning Outcomes

- Student can explain the two main hypotheses for adversarial transferability (shared features, aligned decision boundaries).
- Student can train a substitute model by querying a black-box target and generate transferable adversarial examples.
- Student can measure cross-architecture transfer rate between at least two distinct model families.
