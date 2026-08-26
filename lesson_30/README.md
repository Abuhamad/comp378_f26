# Lesson 30 — Attacks on Federated Learning

Objectives:

- State the Byzantine fault model for federated learning.
- Implement a malicious Flower client that scales its gradient update.
- Explain the DLG gradient inversion attack mechanics.
- Relate batch size to gradient inversion success rate.

## Main Concepts

- Byzantine attack: a malicious client sending arbitrary gradients to disrupt aggregation
- Model poisoning: scaling malicious updates to dominate FedAvg aggregation
- Gradient inversion attacks: reconstructing training images from shared gradients (DLG, R-GAP)
- Scaling attack amplification: boosting malicious update magnitude to survive averaging

## Learning Outcomes

- Student can describe the Byzantine threat model in FL and why standard FedAvg is vulnerable.
- Student can simulate a malicious client that uses gradient scaling to derail global model accuracy.
- Student can explain the DLG gradient inversion attack and the conditions required for successful image reconstruction.
