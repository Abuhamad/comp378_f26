# Lesson 26 — Membership and Attribute Inference Attacks

Objectives:

- Explain why the training/test loss gap enables membership inference.
- Implement the Yeom loss-threshold attack and report TPR and FPR.
- Train shadow models and a binary meta-classifier to execute a full shadow-model MIA.
- Distinguish membership inference from attribute inference.

## Main Concepts

- Membership Inference Attack (MIA): determining whether a record was in the training set
- Shadow model training: training multiple shadow models to learn member vs. non-member loss distributions
- Loss threshold attack and advanced metric-based attacks (Yeom et al.)
- Attribute inference: inferring sensitive attributes from model output distributions

## Learning Outcomes

- Student can explain the shadow model methodology and why training/test loss gap enables membership inference.
- Student can implement a shadow-model-based MIA and compute attack accuracy, precision, recall, and AUC.
- Student can distinguish membership inference from attribute inference by attacker goal and required access level.
