# Lesson 29 — Federated Learning Fundamentals

Objectives:

- State the FedAvg algorithm step by step.
- Distinguish cross-device federated learning from cross-silo federated learning.
- Implement a FedAvg simulation using Flower.
- Explain why non-IID label skew degrades FedAvg convergence.

## Main Concepts

- Federated Learning setting: clients, server, local training, global aggregation
- FedAvg algorithm: weighted averaging of local model updates
- Cross-device vs. cross-silo FL: scale, trust, and data heterogeneity differences
- Non-IID data distribution: label skew, feature skew, and their impact on convergence

## Learning Outcomes

- Student can implement FedAvg and explain how client participation, local epochs, and batch size affect convergence.
- Student can build a Federated Learning simulation using `Flower` (flwr) with configurable numbers of virtual clients.
- Student can demonstrate the degraded convergence caused by Non-IID label skew and quantify the accuracy gap vs. IID training.
