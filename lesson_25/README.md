# Lesson 25 — Model Extraction and Stealing Attacks

Objectives:

- Describe the model stealing threat model.
- Execute a KnockoffNets extraction attack using a proxy dataset.
- Explain why active learning query strategies reduce the number of queries needed.
- Describe how model fingerprinting and watermarking can detect or deter model stealing.

## Main Concepts

- Model stealing via black-box API queries: using output probabilities to reconstruct model behavior
- Hyperparameter extraction: using query patterns to infer architecture and training choices
- Decision boundary reconstruction: active learning-style query strategies
- Intellectual property implications and detection countermeasures (fingerprinting, watermarking)

## Learning Outcomes

- Student can describe the model stealing threat model and what an attacker gains from a stolen clone.
- Student can execute a model extraction attack using the KnockoffNets strategy to build a functional surrogate.
- Student can measure fidelity (agreement rate) between the clone and the target on a held-out set.
